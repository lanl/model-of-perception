"""Prepare the three supplied CCC screenshots as image-intensity scalar fields.

Run from the repository root:
uv run --package svlat-colormapping --extra retrieve python code/prepare_ccc_fields.py
"""
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from task_app.resampling import downsample

CODE = Path(__file__).resolve().parent
SOURCE = CODE / 'data/ccc'
OUTPUT = CODE / 'datasets'


def prepare(spec, source_dir=SOURCE, output_dir=OUTPUT):
    """Keep screen geometry and orientation; do not infer original physical data."""
    path = source_dir / spec['file']
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != spec['sha256']:
        raise ValueError(f'Source screenshot changed: {path.name}')
    with Image.open(io.BytesIO(raw)) as image:
        rgba = np.asarray(image.convert('RGBA'))
    if not np.all(rgba[:, :, 3] == 255):
        raise ValueError('Transparent pixels require an explicit compositing policy.')
    rgb = rgba[:, :, :3].astype(np.float64)
    if np.max(np.ptp(rgb, axis=2)) > 2:
        raise ValueError('This preparation only supports the supplied near-gray screenshots.')
    values = rgb.mean(axis=2)
    for left, top, right, bottom in spec.get('background_repairs_xyxy', []):
        values[top:bottom, left:right] = 0
    height, width = values.shape
    side = max(height, width)
    top, left = (side-height)//2, (side-width)//2
    padding = ((top, side-height-top), (left, side-width-left))
    square = np.pad(values, padding, constant_values=0)
    prepared = downsample(square)[::-1].copy()
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / (spec['id']+'.npy')
    np.save(target, prepared, allow_pickle=False)
    provenance = dict(
        source_page='https://ccctool.com/',
        citation='Nardini et al., A Testing Environment for Continuous Colormaps, '
                 'IEEE TVCG 27(2), 1043–1053 (2021; online 2020), doi:10.1109/TVCG.2020.3028955.',
        reference_scope='CCC test-suite reference; does not identify the original source of this screenshot.',
        source_file='code/data/ccc/'+spec['file'], source_sha256=spec['sha256'],
        source_shape=[height, width], prepared_shape=list(prepared.shape),
        field='arithmetic mean of the three stored near-gray RGB code values; no gamma decoding',
        scalar_units='screenshot intensity (0–255), not original physical values',
        acquisition='User-supplied screenshot of CCC-Tool, 2026-11-06',
        original_dataset=spec['original_dataset'],
        license='Original image/data reuse terms not established from the supplied screenshot.',
        background_repairs_xyxy=spec.get('background_repairs_xyxy', []),
        background_repair_reason='Remove white screenshot corner marks in the black globe background.' if spec.get('background_repairs_xyxy') else 'None.',
        padding_top_bottom_left_right=[*padding[0], *padding[1]],
        padding_value=0, padding_reason='Square app grids; retain image proportions and all image content.',
        resampling='Area-weighted cell averaging to 100 × 100 after square padding; no value normalization.',
        orientation='Source top row flipped to highest y.',
        axes='Normalized screen coordinates [0,1] on each axis; not geographic or physical coordinates.',
        background_policy='Black background and padding are ordinary zero-valued cells; no scientific-domain mask.',
        interpretation='Tasks concern the prepared screenshot intensity, including rendering and quantization effects. '
                       'Globe views are not asserted to be independent simulation datasets.')
    meta = dict(id=spec['id'], name=spec['name'], file=target.name, extent=[0.0,1.0],
                file_sha256=hashlib.sha256(target.read_bytes()).hexdigest(), provenance=provenance)
    target.with_suffix('.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False)+'\n')
    return prepared


def main():
    specs = json.loads((SOURCE/'sources.json').read_text())
    sheet = Image.new('RGB', (900,330), 'white')
    draw = ImageDraw.Draw(sheet)
    for index, spec in enumerate(specs):
        values = prepare(spec)
        preview = Image.fromarray(np.rint(values[::-1]).astype(np.uint8)).resize((280,280))
        sheet.paste(preview, (index*300+10,40))
        draw.text((index*300+10,12), spec['id'], fill='black')
        print(f"{spec['id']}: {values.shape}, intensity [{values.min():.3f}, {values.max():.3f}]")
    sheet.save(OUTPUT/'ccc-preview.png')


if __name__ == '__main__':
    main()
