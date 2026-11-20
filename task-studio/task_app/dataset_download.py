"""Small, bounded downloads of independent 2D scalar examples.

The Well stores many samples per HDF5 file. A seekable HTTP Range reader lets
h5py read one scalar snapshot without downloading its containing archive.
"""
import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import re
import time
from urllib.parse import quote
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image, ImageDraw

from .resampling import downsample

HERE = Path(__file__).resolve().parent
SOURCES = json.loads((HERE / 'dataset_sources.json').read_text())
MIB = 1024 * 1024
USER_AGENT = 'ScalarTaskStudio-dataset-preparation/1'


class DownloadLimitError(ValueError):
    pass


class Budget:
    def __init__(self, limit):
        self.limit, self.used = limit, 0

    def check(self, count):
        if self.used + count > self.limit:
            raise DownloadLimitError(f'Transfer cap of {self.limit / MIB:g} MiB reached.')


def bounded_get(url, limit, budget):
    request = Request(url, headers={'User-Agent': USER_AGENT, 'Accept-Encoding': 'identity'})
    with urlopen(request, timeout=20) as response:
        declared = response.headers.get('Content-Length')
        if declared is not None and int(declared) > limit:
            raise DownloadLimitError(f'Response exceeds the {limit / MIB:g} MiB limit: {url}')
        pieces, size = [], 0
        while True:
            allowance = min(65536, limit - size, budget.limit - budget.used)
            if allowance <= 0:
                if declared is not None and size == int(declared):
                    break
                raise DownloadLimitError('Response reached its transfer limit.')
            piece = response.read(allowance)
            if not piece:
                break
            size += len(piece)
            budget.used += len(piece)
            pieces.append(piece)
        if declared is not None and size != int(declared):
            raise ValueError('Truncated download.')
        return b''.join(pieces)


class HTTPRangeFile(io.RawIOBase):
    """Read-only, block-cached file; refuses a server that ignores Range.

    Budgets count payload bytes, not HTTP headers. No full-file fallback exists.
    Signed redirect URLs are not retained, so each request can obtain a fresh one.
    """
    block_size = 65536

    def __init__(self, url, size, budget, limit=10 * MIB, opener=urlopen):
        super().__init__()
        self.url, self.size, self.budget = url, size, budget
        self.limit, self.used, self.position = limit, 0, 0
        self.blocks, self.requests, self.started = {}, 0, time.monotonic()
        self.opener = opener

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.position

    def seek(self, offset, whence=0):
        position = offset + (0 if whence == 0 else self.position if whence == 1 else self.size if whence == 2 else -1)
        if whence not in (0, 1, 2) or position < 0:
            raise ValueError('Invalid seek.')
        self.position = position
        return position

    def _block(self, index):
        if index in self.blocks:
            return self.blocks[index]
        start = index * self.block_size
        end = min(start + self.block_size, self.size) - 1
        count = end - start + 1
        if self.requests >= 160 or time.monotonic() - self.started > 120:
            raise DownloadLimitError('Stopped after 160 requests or two minutes for this source.')
        if self.used + count > self.limit:
            raise DownloadLimitError(f'Snapshot exceeded its {self.limit / MIB:g} MiB cap; no full archive was downloaded.')
        self.budget.check(count)
        request = Request(self.url, headers={'Range': f'bytes={start}-{end}',
                         'Accept-Encoding': 'identity', 'User-Agent': USER_AGENT})
        self.requests += 1
        with self.opener(request, timeout=20) as response:
            if response.status != 206:
                raise ValueError('Server did not honor HTTP Range; refusing a full-file download.')
            expected = f'bytes {start}-{end}/{self.size}'
            if response.headers.get('Content-Range') != expected:
                raise ValueError('Unexpected Content-Range; refusing ambiguous source bytes.')
            # Reserve before reading, so failed or truncated requests cannot reset a cap.
            self.used += count
            self.budget.used += count
            data = response.read(count)
            if len(data) != count:
                raise ValueError('Truncated byte-range response.')
        self.blocks[index] = data
        return data

    def read(self, size=-1):
        count = max(0, self.size - self.position) if size < 0 else min(size, max(0, self.size - self.position))
        # HDF5 must not allocate an archive-sized buffer before the cap is checked.
        if count > self.limit:
            raise DownloadLimitError('Requested HDF5 read exceeds the snapshot transfer cap.')
        if count:
            first = self.position // self.block_size
            last = (self.position + count - 1) // self.block_size
            needed = sum(min(self.block_size, self.size - i * self.block_size)
                         for i in range(first, last + 1) if i not in self.blocks)
            self.budget.check(needed)
            if self.used + needed > self.limit:
                raise DownloadLimitError('Requested HDF5 read exceeds the remaining snapshot transfer cap.')
        pieces = []
        while count:
            index, offset = divmod(self.position, self.block_size)
            block = self._block(index)
            take = min(count, len(block) - offset)
            pieces.append(block[offset:offset + take])
            self.position += take
            count -= take
        return b''.join(pieces)

    def readinto(self, target):
        data = self.read(len(target))
        target[:len(data)] = data
        return len(data)


def fetch_json(url, budget):
    return json.loads(bounded_get(url, 2 * MIB, budget))


def resolve_well_file(source, budget):
    repo = source['repository']
    info = fetch_json(f'https://huggingface.co/api/datasets/{repo}/revision/main', budget)
    revision = info['sha']
    if not re.fullmatch(r'[a-f0-9]{40}', revision):
        raise ValueError('Repository did not supply an immutable commit identifier.')
    listing = fetch_json(f'https://huggingface.co/api/datasets/{repo}/tree/{revision}/data/test?recursive=true&limit=1000', budget)
    candidates = [entry for entry in listing if entry.get('type') == 'file'
                  and entry['path'].endswith(('.h5', '.hdf5')) and entry.get('size', 0) > 0]
    if not candidates:
        raise ValueError('No published HDF5 test file found; source layout may have changed.')
    preferred = source.get('prefer_filename', [])
    candidates.sort(key=lambda entry: (not all(token.lower() in entry['path'].lower() for token in preferred),
                                        entry['size'], entry['path']))
    chosen = candidates[0]
    url = f'https://huggingface.co/datasets/{repo}/resolve/{revision}/{quote(chosen["path"], safe="/")}'
    return url, chosen['size'], dict(repository=repo, revision=revision, file=chosen['path'])


def text_value(value):
    return value.decode('utf-8') if isinstance(value, bytes) else str(value)


def scalar_snapshot(file, source):
    """Select one native 2D scalar; no inferred vector components or 3D slices."""
    if int(file.attrs.get('n_spatial_dims', -1)) != 2:
        raise ValueError('Source is not explicitly a two-spatial-dimensional dataset.')
    fields = file['t0_fields']
    by_lower = {name.lower(): name for name in fields.keys()}
    name = next((by_lower[c.lower()] for c in source['field_candidates'] if c.lower() in by_lower), None)
    if name is None:
        raise ValueError(f'Expected scalar missing; available scalar fields: {list(fields.keys())}')
    field = fields[name]
    sample_varying = bool(field.attrs['sample_varying'])
    time_varying = bool(field.attrs['time_varying'])
    if not np.asarray(field.attrs['dim_varying'], dtype=bool).all():
        raise ValueError('Chosen scalar is not defined over both spatial dimensions.')
    if field.ndim != 2 + int(sample_varying) + int(time_varying):
        raise ValueError('Unexpected scalar dimensions; refusing to guess an axis.')
    indices = [0] if sample_varying else []
    time_index = None
    if time_varying:
        time_index = (field.shape[int(sample_varying)] - 1) // 2
        indices.append(time_index)
    values = np.asarray(field[tuple(indices) + (slice(None), slice(None))])
    if list(values.shape) != source['expected_shape']:
        raise ValueError(f'Unexpected spatial shape {values.shape}; expected {source["expected_shape"]}.')
    dimensions = file['dimensions']
    axes = [text_value(v) for v in dimensions.attrs['spatial_dims']]
    if len(axes) != 2:
        raise ValueError('Expected two spatial coordinate arrays.')
    coordinates = []
    for axis, length in zip(axes, values.shape):
        dataset = dimensions[axis]
        if bool(dataset.attrs.get('time_varying', False)):
            raise ValueError('Time-varying coordinates are not supported.')
        coords = np.asarray(dataset[0] if bool(dataset.attrs.get('sample_varying', False)) else dataset[:])
        if coords.ndim != 1 or len(coords) != length or not np.isfinite(coords).all():
            raise ValueError('Invalid coordinate array.')
        delta = np.diff(coords)
        if not (np.all(delta > 0) or np.all(delta < 0)) or not np.allclose(delta, delta[0], rtol=1e-4, atol=1e-9):
            raise ValueError('Source coordinates must be regular and monotone.')
        coordinates.append(coords)
    # Published ordering is (x, y); the application uses (row=y, column=x).
    values = values.T
    if coordinates[0][0] > coordinates[0][-1]: values = values[:, ::-1]
    if coordinates[1][0] > coordinates[1][-1]: values = values[::-1]
    metadata = dict(field=name, trajectory=0 if sample_varying else None, time_index=time_index,
                    scalar_units=text_value(field.attrs.get('units', 'not specified in source')),
                    source_spatial_axes=axes,
                    source_coordinates={axis: coords.tolist() for axis, coords in zip(axes, coordinates)},
                    source_array_shape=list(field.shape),
                    orientation='source (axis0,axis1) transposed to rows=axis1; increasing coordinates')
    return values, metadata


def get_well(source, budget):
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError('Install the datasets extra with uv: uv sync --package svlat-colormapping --extra retrieve --extra datasets') from exc
    url, size, metadata = resolve_well_file(source, budget)
    print(f'  Reading one scalar snapshot from {size / MIB:,.1f} MiB container (cap {source["transfer_limit_mib"]} MiB).', flush=True)
    with HTTPRangeFile(url, size, budget, source['transfer_limit_mib'] * MIB) as remote:
        with h5py.File(remote, 'r') as file:
            values, selection = scalar_snapshot(file, source)
        metadata.update(selection, source_url=url, source_file_bytes=size,
                        range_payload_bytes=remote.used, range_requests=remote.requests)
    return values, metadata


def get_image(source, budget):
    data = bounded_get(source['url'], source['transfer_limit_mib'] * MIB, budget)
    actual = hashlib.sha256(data).hexdigest()
    if actual != source['sha256']:
        raise ValueError('Image checksum does not match the published scikit-image registry.')
    with Image.open(io.BytesIO(data)) as im:
        if getattr(im, 'n_frames', 1) != 1 or im.mode not in ('L', 'I', 'I;16', 'F'):
            raise ValueError('Expected a single grayscale scalar image; RGB conversion is not inferred.')
        values = np.asarray(im).copy()
    if list(values.shape) != source['expected_shape']:
        raise ValueError(f'Unexpected image shape {values.shape}.')
    return values[::-1], dict(source_url=source['url'], source_sha256=actual, field=source['field'],
                              orientation='source top row flipped to highest y', scalar_units='stored image intensity')


def write_prepared(source, values, provenance, output, max_side=100):
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError('This application currently needs a square raster; no cropping or stretching is performed.')
    prepared = downsample(values, max_side)
    if np.ptp(prepared) <= max(1., float(np.max(np.abs(prepared)))) * 1e-12:
        raise ValueError('Chosen snapshot is effectively constant after reduction; not adding an uninformative field.')
    if np.max(np.abs(prepared)) > 1e100:
        raise ValueError('Scalar values exceed the application limit.')
    ident = source['id']
    raw_hash = hashlib.sha256(np.ascontiguousarray(values, dtype='<f8').tobytes()).hexdigest()
    provenance = dict(provenance, source_page=source['source_page'], citation=source['citation'],
                      license=source['license'], source_shape=list(values.shape), source_scalar_sha256=raw_hash,
                      prepared_shape=list(prepared.shape),
                      resampling='area-weighted cell averaging before color mapping; no value normalization',
                      axes='normalized raster coordinates [0,1] on each axis; physical coordinates retained only in provenance')
    output.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO(); np.save(buf, prepared, allow_pickle=False)
    data = buf.getvalue()
    metadata = dict(id=ident, name=source['name'], file=f'{ident}.npy', extent=[0., 1.],
                    file_sha256=hashlib.sha256(data).hexdigest(), provenance=provenance)
    tmp = output / f'.{ident}.npy.tmp'; tmp.write_bytes(data); tmp.replace(output / metadata['file'])
    tmp = output / f'.{ident}.json.tmp'
    tmp.write_text(json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(output / f'{ident}.json')
    return metadata


def contact_sheet(output):
    entries = []
    for source in SOURCES:
        path = output / f'{source["id"]}.json'
        if path.exists():
            try:
                meta, values = read_prepared(output, source['id'], 100)
                entries.append((meta, values))
            except (OSError, ValueError, KeyError, TypeError):
                continue
    if not entries:
        return
    sheet = Image.new('RGB', (3 * 240, math.ceil(len(entries) / 3) * 270), 'white')
    draw = ImageDraw.Draw(sheet)
    for index, (meta, values) in enumerate(entries):
        x, y = (index % 3) * 240, (index // 3) * 270
        gray = np.round(255 * (values - values.min()) / np.ptp(values)).astype(np.uint8)[::-1]
        tile = Image.fromarray(gray).resize((220, 220), Image.Resampling.BILINEAR)
        sheet.paste(tile, (x + 10, y + 10))
        draw.text((x + 10, y + 236), meta['id'], fill='black')
        draw.text((x + 10, y + 251), f'{values.shape[1]} x {values.shape[0]}', fill='black')
    sheet.save(output / 'preview.png')


def read_prepared(output, ident, max_side):
    meta = json.loads((output / f'{ident}.json').read_text())
    data = (output / f'{ident}.npy').read_bytes()
    if meta['id'] != ident or meta['file'] != f'{ident}.npy' or meta['file_sha256'] != hashlib.sha256(data).hexdigest():
        raise ValueError('Cached data identity or checksum mismatch.')
    values = np.load(io.BytesIO(data), allow_pickle=False)
    if (values.dtype.kind not in 'fiu' or values.ndim != 2 or
        values.shape[0] != values.shape[1] or not 3 <= values.shape[0] <= max_side or
        not np.isfinite(values).all() or np.max(np.abs(values)) > 1e100 or np.ptp(values) == 0):
        raise ValueError('Cached data need rebuilding.')
    return meta, values


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--list', action='store_true', help='Show the independent sources and hard transfer caps; no network.')
    parser.add_argument('--only', nargs='+', choices=[s['id'] for s in SOURCES])
    parser.add_argument('--output', type=Path, default=HERE / 'datasets')
    parser.add_argument('--max-side', type=int, default=100, choices=range(3, 101), metavar='3..100')
    parser.add_argument('--max-total-mib', type=int, default=64)
    args = parser.parse_args(argv)
    if not 1 <= args.max_total_mib <= 256:
        parser.error('--max-total-mib must be between 1 and 256.')
    sources = [s for s in SOURCES if args.only is None or s['id'] in args.only]
    if args.list:
        for source in sources:
            size = ' x '.join(map(str, source['expected_shape']))
            print(f'{source["id"]}: {size} -> at most {args.max_side} x {args.max_side}; transfer cap {source["transfer_limit_mib"]} MiB\n  {source["name"]}\n  {source["source_page"]}')
        print(f'Total payload cap: {args.max_total_mib} MiB (package installation is separate).')
        return 0
    budget = Budget(args.max_total_mib * MIB)
    failures, successes = [], []
    for source in sources:
        ident = source['id']
        path = args.output / f'{ident}.json'
        if path.exists():
            try:
                read_prepared(args.output, ident, args.max_side)
                print(f'{ident}: already prepared; no download.', flush=True)
                successes.append(ident)
                continue
            except (OSError, ValueError, KeyError, TypeError):
                pass
        print(f'{ident}: fetching…', flush=True)
        before = budget.used
        try:
            values, metadata = get_image(source, budget) if source['kind'] == 'image' else get_well(source, budget)
            metadata['download_payload_bytes'] = budget.used - before
            write_prepared(source, values, metadata, args.output, args.max_side)
            successes.append(ident)
            print(f'  Saved; transferred {(budget.used - before) / MIB:.2f} MiB.', flush=True)
        except (OSError, ValueError, KeyError, RuntimeError, TypeError) as exc:
            failures.append(dict(id=ident, error=str(exc)))
            print(f'  Skipped: {exc}', flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    report = dict(prepared=successes, failures=failures, downloaded_payload_bytes=budget.used,
                  payload_limit_bytes=budget.limit, max_side=args.max_side)
    (args.output / 'download-report.json').write_text(json.dumps(report, indent=2) + '\n')
    contact_sheet(args.output)
    print(f'{len(successes)}/{len(sources)} sources ready; {budget.used / MIB:.2f} MiB transferred. Restart Scalar Task Studio to load them.\n{args.output}')
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
