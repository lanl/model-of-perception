"""Readable sampled choices shared by the preview and illustrated Markdown."""


def summarize_parameters(params, context):
    def number(value):
        return format(value, '.10g')

    lines = []
    for label, (row, col) in zip(('P', 'Q'), params.get('cells', [])):
        step = context['cell_spacing']
        lo = context['extent'][0]
        lines.append(f'Point {label}: x = {number(lo+(col+.5)*step)}, y = {number(lo+(row+.5)*step)}')
    for label, (x0, x1, y0, y1) in zip(('P', 'Q'), params.get('rectangles', [])):
        lines.append(f'Box {label} bounds: x = {number(x0)} to {number(x1)}; y = {number(y0)} to {number(y1)}')
    if 'window_cells' in params:
        width = params['window_cells']
        lines.append(f'Window width: {width} cells')
        cue = {'axis_scale': 'axis scale and ruler', 'reference_square': 'reference square R'}
        lines.append(f'Size cue: {cue[params["window_size_cue"]]}')
    for key, label in (('threshold', 'Threshold'), ('lower_threshold', 'Lower threshold'),
                       ('upper_threshold', 'Upper threshold')):
        if key in params:
            lines.append(f'{label}: {number(params[key])}')
    if 'extremum' in params:
        lines.append(f'Extremum: {params["extremum"]}')
        scopes = {'all': 'all local extrema, including global',
                  'non_global': 'local extrema excluding global', 'global': 'global only'}
        lines.append(f'Scope: {scopes[params["scope"]]}')
    if 'interval_measure' in params:
        lines.append(f'Use: {params["interval_measure"]}')
    if 'find_extremum' in params:
        lines.append(f'Find: {params["find_extremum"]}')
    if 'correlation_direction' in params:
        lines.append(f'Correlation direction: {params["correlation_direction"]}')
    return lines or ['None beyond the selected stages.']
