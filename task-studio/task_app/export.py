"""Portable exports; relative image links are identical in CSV and Markdown."""
import csv
import hashlib
import io
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from .contracts import combinations
from .engine import sample_task, render

VERSION = '3.0.0'
MAX_UNIQUE_DRAWS = 100
FIELDS = ['task_id', 'dataset', 'colormap', 'operation', 'selection', 'aggregation', 'action',
          'parameters', 'image_ref', 'prompt', 'answer_spec', 'ground_truth', 'context', 'provenance']


def encoded(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':')) if isinstance(value,(dict,list)) else str(value)


def md(value):
    return encoded(value).replace('&','&amp;').replace('<','&lt;').replace('>','&gt;').replace('|','&#124;').replace('\n','<br>').replace('`','&#96;')


def task_fingerprint(dataset, colormap, stages, record):
    """Task content, not its random seed, generated ID, filename, or asset name."""
    content = dict(dataset_sha256=dataset['sha256'], colormap_sha256=colormap['sha256'],
                   stages=stages, **{key:record[key] for key in
                   ('parameters','context','prompt','answer_spec','ground_truth')})
    return hashlib.sha256(json.dumps(content,sort_keys=True,allow_nan=False,
                                    separators=(',',':')).encode()).hexdigest()


def generate(config, assets, output, progress=lambda *_: None, *, check=lambda: None):
    count, seed = config.get('count'), config.get('seed')
    if type(count) is not int or not 1 <= count <= 200: raise ValueError('Task count must be 1–200.')
    if type(seed) is not int or not 0 <= seed < 2**53: raise ValueError('Seed must be an integer from 0 to 2^53−1.')
    combos = combinations(config)
    if not combos: raise ValueError('No included compatible stage combinations remain.')
    for key, source in [('datasets',assets.datasets), ('colormaps',assets.maps)]:
        if not isinstance(config.get(key), list) or not config[key] or any(v not in source for v in config[key]):
            raise ValueError(f'Select valid {key}.')
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    (output/'images').mkdir(); (output/'inputs').mkdir()
    records, failures = [], []
    seen = set()
    duplicates_rejected = 0
    rng = np.random.default_rng(seed)
    for slot in range(count):
        check()
        record = None
        last_error = ''
        failure_kind = 'validation'
        for draw in range(1,MAX_UNIQUE_DRAWS+1):
            check()
            stages = combos[int(rng.integers(len(combos)))]
            dataset = assets.datasets[str(rng.choice(config['datasets']))]
            cmap_id = str(rng.choice(config['colormaps']))
            task_seed = int(rng.integers(0,2**53)); task_rng = np.random.default_rng(task_seed)
            for attempt in range(1,51):
                check()
                try:
                    candidate, _, overlays = sample_task(dataset, stages, task_rng)
                    break
                except ValueError as exc:
                    last_error = str(exc)
            else:
                failure_kind = 'validation'
                break
            fingerprint = task_fingerprint(dataset,assets.maps[cmap_id],stages,candidate)
            if fingerprint in seen:
                duplicates_rejected += 1
                failure_kind = 'uniqueness'
                last_error = f'No new distinct task found within {MAX_UNIQUE_DRAWS} candidate draws. Broaden the selections or request fewer tasks.'
                continue
            record = candidate
            break
        if record is None:
            failures.append(dict(slot=slot+1, dataset=dataset['id'], **stages, reason=last_error,
                                 kind=failure_kind, candidate_draws=draw, parameter_attempts=attempt))
            progress(slot+1,count); continue
        seen.add(fingerprint)
        task_id = f'task_{slot+1:04d}'
        image_ref = f'images/{task_id}.png'
        png = render(dataset['values'],overlays,dataset['extent'],assets.cmap(cmap_id))
        (output/image_ref).write_bytes(png)
        record.update(task_id=task_id, dataset=dataset['name'], colormap=assets.maps[cmap_id]['name'], **stages,
                      image_ref=image_ref, provenance=dict(generator_version=VERSION, task_seed=task_seed,
                      parameter_attempt=attempt, dataset_id=dataset['id'], dataset_sha256=dataset['sha256'],
                      candidate_draw=draw, task_fingerprint=fingerprint,
                      colormap_id=cmap_id, colormap_sha256=assets.maps[cmap_id]['sha256'],
                      image_sha256=hashlib.sha256(png).hexdigest()))
        if 'source' in dataset:
            record['provenance']['dataset_source'] = dataset['source']
        records.append(record); progress(slot+1,count)
    for ident in config['datasets']:
        asset=assets.datasets[ident]
        np.save(output/'inputs'/f'{ident}.npy',asset['values'],allow_pickle=False)
        (output/'inputs'/f'{ident}.json').write_text(json.dumps({k:v for k,v in asset.items() if k!='values'},indent=2))
    for ident in config['colormaps']:
        (output/'inputs'/f'{ident}.json').write_text(json.dumps(assets.maps[ident],indent=2))
    manifest = dict(generator_version=VERSION, created_utc=datetime.now(timezone.utc).isoformat(), configuration=config,
                    requested_count=count, actual_count=len(records), failures=failures,
                    compatible_combination_count=len(combinations(config,include_excluded=True)),
                    included_combination_count=len(combos),
                    duplicates_rejected=duplicates_rejected,
                    uniqueness='Within this batch, identical dataset and colormap contents, stages, parameters, context, prompt and answer are emitted once. Different colormaps are distinct tasks. Seeds, task IDs and asset names do not make a task distinct.',
                    sampling='Uniform candidate draws over included compatible stage tuples; explicitly excluded tuples are never sampled. Dataset and colormap independently uniform. Parameters uniform on their stated domains, conditioned on validity and uniqueness. Duplicate candidates are redrawn, up to 100 candidate draws per slot. Up to 50 parameter attempts within a tuple and dataset; validation failures and uniqueness shortfalls are reported.',
                    parameter_domains={'rectangles':'Independent uniform pairs of distinct grid boundaries per axis',
                    'points':'Uniform distinct cell indices', 'windows':'Uniform odd widths 3 through grid width',
                    'window_size_cue':'Uniform axis_scale / reference_square; cue changes presentation, not the statistic or window support',
                    'above_and_below_threshold':'One uniform draw between transformed field minimum and maximum; strict comparison',
                    'between_thresholds':'Two independent uniform draws between transformed field minimum and maximum, sorted as lower/upper; both bounds inclusive',
                    'extremum':'Uniform minimum/maximum', 'scope':'Uniform all/non_global/global',
                    'interval_measure':'Uniform minimum/maximum/width for Range and Extent X/Y with two groups or window search',
                    'correlation_direction':'Uniform x/y/x+y/y-x', 'find_extremum':'Uniform minimum/maximum'})
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (output/'tasks.json').write_text(json.dumps(records,indent=2,ensure_ascii=False,allow_nan=False))
    with (output/'tasks.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=FIELDS); writer.writeheader()
        for record in records:
            row={key:encoded(record[key]) for key in FIELDS}
            # Guard spreadsheet formula interpretation of uploaded names.
            for key in ('dataset','colormap'):
                if row[key].startswith(('=','+','-','@','\t','\r')): row[key]="'"+row[key]
            writer.writerow(row)
    columns=['task_id','dataset','colormap','operation','selection','aggregation','action','parameters','image_ref','prompt','ground_truth']
    markdown='# Generated tasks\n\n'+f'{len(records)} of {count} requested tasks · uniform sampling · seed {seed}.\n\n'
    markdown+='Extract images.zip beside this file, or extract the complete bundle. Answers are included for review. See manifest.json for constraints and failures.\n\n'
    markdown+='Sampled parameters show only independently sampled choices. Derived values and fixed settings are in the context field of tasks.csv and tasks.json.\n\n'
    markdown+='| '+' | '.join('Image' if c=='image_ref' else 'Sampled parameters' if c=='parameters' else c.replace('_',' ').title() for c in columns)+' |\n'
    markdown+='| '+' | '.join('---' for _ in columns)+' |\n'
    for record in records:
        markdown+='| '+' | '.join(f'![{record["task_id"]}]({record[c]})' if c=='image_ref' else
                                 '<br>'.join(md(line) for line in record['parameter_summary']) if c=='parameters' else
                                 md(record[c]) for c in columns)+' |\n'
    if failures:
        markdown+='\n## Ungenerated slots\n\n'+'\n'.join(f'- Slot {f["slot"]}: {md(f["reason"])}' for f in failures)+'\n'
    (output/'tasks.md').write_text(markdown,encoding='utf-8')
    with zipfile.ZipFile(output/'images.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for path in sorted((output/'images').glob('*.png')): archive.write(path,path.relative_to(output))
    with zipfile.ZipFile(output/'everything.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.rglob('*')):
            if path.is_file() and path.suffix!='.zip': archive.write(path,path.relative_to(output))
    return dict(records=records,manifest=manifest)
