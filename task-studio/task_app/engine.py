"""Execute compatible stages, using the project's scalar analysis primitives."""
import io
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parent / '.cache'))
import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
import numpy as np

from .numerics.search import coordinates, derived_map
from .numerics.extrema import objects
from .numerics.threshold import area, components
from .numerics.correlation import coordinate, pearson, category
from .contracts import compatibility, INTERVAL_AGGREGATIONS, INTERVAL_MEASURES
from .interpolation import INTERPOLATION, scalar_at, scalar_raster
from .critical_points import bilinear_saddles
from .parameters import summarize_parameters


def sample_task(dataset, stages, rng):
    """Return the record, analysis field, and overlays; display the original dataset field."""
    ok, reason = compatibility(**stages)
    if not ok: raise ValueError(reason)
    field = dataset['values']
    n = len(field)
    extent = dataset['extent']
    lo, hi = extent
    step = (hi-lo)/n
    xy = coordinates(n, extent)
    op, selection, agg, action = (stages[k] for k in ('operation', 'selection', 'aggregation', 'action'))
    values = derived_map(field, 'gradient', step)[0] if op == 'gradient' else field
    params = {}
    context = {'extent': extent, 'grid_shape': [n, n], 'row_zero': 'lowest y', 'cell_spacing': step,
              'normalization': 'linear', 'color_range': [float(field.min()), float(field.max())],
              'display_field': 'original_scalar_field',
              'analysis_field': 'gradient_magnitude' if op == 'gradient' else 'original_scalar_field',
              'analysis_range': [float(values.min()), float(values.max())],
              'interpolation': dict(INTERPOLATION), 'statistics_support': 'original_grid_samples'}
    overlays = []
    masks = []
    positions = []
    descriptions = []
    is_objects = selection in ('extrema', 'saddles')
    is_interval = agg in INTERVAL_AGGREGATIONS
    groups_objects = []

    def value_bounds(samples):
        return [float(np.min(samples)), float(np.max(samples))]

    def interval_width(pair):
        width = pair[1]-pair[0]
        # Grid-coordinate spans are integer multiples of spacing. Preserve ties
        # between equal-width windows despite endpoint subtraction roundoff.
        if agg in ('extent_x','extent_y') and selection != 'saddles':
            return float(round(width/step)*step)
        return float(width)

    def box():
        # Every pair of distinct grid boundaries is equally likely on each axis.
        r0, r1 = sorted(rng.choice(n+1, 2, replace=False).tolist())
        c0, c1 = sorted(rng.choice(n+1, 2, replace=False).tolist())
        mask = np.zeros_like(values, bool); mask[r0:r1, c0:c1] = True
        return mask, [lo+c0*step, lo+c1*step, lo+r0*step, lo+r1*step]

    if selection in ('point', 'points'):
        indices = rng.choice(n*n, 1 if selection == 'point' else 2, replace=False)
        params['cells'] = []
        for label, index in zip(('P', 'Q'), indices):
            r, c = divmod(int(index), n)
            m = np.zeros_like(values, bool); m[r, c] = True; masks.append(m)
            overlays.append(('point', label, xy[r,c].tolist()))
            params['cells'].append([r, c]); positions.append([[r,c]])
            descriptions.append(f'the marked point {label}')
    elif selection in ('region', 'regions', 'inside_outside'):
        params['rectangles'] = []
        for label in ('P', 'Q') if selection == 'regions' else ('P',):
            mask, bounds = box(); masks.append(mask); overlays.append(('box', label, bounds))
            params['rectangles'].append(bounds); descriptions.append(f'cells inside rectangle {label}')
        if selection == 'inside_outside':
            masks.append(~masks[0]); descriptions.append('cells outside rectangle P (group Q)')
    elif selection == 'windows':
        widths = list(range(3, n+1, 2)); w = int(rng.choice(widths)); params['window_cells'] = w
        side = float(w*step)
        cue = str(rng.choice(['axis_scale','reference_square']))
        params['window_size_cue'] = cue
        context['window_side_length'] = side
        if cue == 'axis_scale':
            overlays.append(('window_scale','Window side',side))
        else:
            margin = min((hi-lo)*.05,max(0.,(hi-lo-side)/2))
            bounds = [lo+margin,lo+margin+side,lo+margin,lo+margin+side]
            context['window_reference_bounds'] = bounds
            overlays.append(('reference_box','R',bounds))
        def windows():
            for r in range(n-w+1):
                for c in range(n-w+1):
                    m = np.zeros_like(values, bool); m[r:r+w,c:c+w] = True
                    yield m
        masks = windows()
        positions = [[[r+w//2, c+w//2]] for r in range(n-w+1) for c in range(n-w+1)]
        descriptions = [f'each fully contained {w} × {w} cell square window']
    elif selection in ('threshold', 'below_threshold', 'between_thresholds'):
        if selection == 'between_thresholds':
            lower, upper = sorted(float(v) for v in rng.uniform(values.min(), values.max(), size=2))
            masks = [(values >= lower) & (values <= upper)]
            params.update(lower_threshold=lower, upper_threshold=upper)
            context['threshold_bounds'] = 'inclusive'
            descriptions = [f'cells with values between {lower:.10g} and {upper:.10g} (inclusive)']
        else:
            level = float(rng.uniform(values.min(), values.max()))
            above = selection == 'threshold'
            masks = [values > level if above else values < level]
            params['threshold'] = level
            context['threshold_comparison'] = '>' if above else '<'
            descriptions = [f'cells with values strictly {"above" if above else "below"} {level:.10g}']
    elif selection == 'extrema':
        kind = str(rng.choice(['minimum', 'maximum'])); scope = str(rng.choice(['all', 'non_global', 'global']))
        params.update(extremum=kind, scope=scope)
        context['connectivity'] = 8
        found = objects(values, kind=kind, connectivity=8)
        found = [o for o in found if scope == 'all' or o['global_extremum'] == (scope == 'global')]
        groups_objects = [found]
        descriptions = [f'{scope.replace("_", " ")} {kind} plateaus (8-connected, each plateau counted once; field boundaries included)']
    elif selection == 'saddles':
        found = bilinear_saddles(values,extent)
        groups_objects = [found]
        context['statistics_support'] = 'interior_bilinear_saddle_points'
        context['saddle_definition'] = 'Nondegenerate stationary points strictly inside bilinear patches between original grid samples; excludes patch edges, grid vertices, and degenerate stationary sets.'
        descriptions = ['saddle points in the interpolated field']
    else:
        masks = [np.ones_like(values, bool)]; descriptions = ['all cells in the field']

    numbers, candidates, collections, intervals = [], [], [], []
    if agg == 'pearson':
        direction = str(rng.choice(['x', 'y', 'x+y', 'y-x']))
        params['correlation_direction'] = direction
        posfield = coordinate(xy[:,:,0], xy[:,:,1], extent, direction)
    if is_objects:
        selected_objects = groups_objects[0]
        if agg == 'mean' or is_interval:
            if len(selected_objects) < 2:
                raise ValueError('Aggregation requires at least two samples per group (distinct selected objects).')
        if agg == 'mean':
            numbers = [float(np.mean([obj['value'] for obj in selected_objects]))]
            context.update(aggregation_input='one_scalar_per_object',object_weighting='equal',
                          selected_object_count=len(selected_objects))
        elif is_interval:
            if agg == 'range':
                samples = [obj['value'] for obj in selected_objects]
            else:
                axis = 0 if agg == 'extent_x' else 1
                samples = ([obj['xy'][axis] for obj in selected_objects] if selection == 'saddles' else
                           [xy[r,c,axis] for obj in selected_objects for r,c in obj['cells']])
            intervals.append(value_bounds(samples))
            context['selected_object_count'] = len(selected_objects)
        else:
            collections = groups_objects
        if selection == 'extrema':
            candidates = [[cell for obj in selected_objects for cell in obj['cells']]]
    else:
        for index, mask in enumerate(masks):
            cells = np.argwhere(mask).tolist()
            if agg != 'identity' and len(cells) == 1:
                raise ValueError('Aggregation requires at least two samples per group.')
            if agg == 'identity':
                collections.append(cells); candidates.append(cells)
                if len(cells) == 1:
                    r,c = cells[0]
                    numbers.append(float(scalar_at(values,extent,xy[r,c])))
            elif is_interval:
                if not cells: raise ValueError('This statistic requires a nonempty selection.')
                samples = values[mask] if agg == 'range' else xy[:,:,0 if agg == 'extent_x' else 1][mask]
                intervals.append(value_bounds(samples))
                candidates.append(positions[index] if selection == 'windows' else [])
            elif agg == 'components':
                context['connectivity'] = 4
                labels = components(mask, 4)
                group = [np.argwhere(labels == i).tolist() for i in range(1, int(labels.max())+1)]
                collections.append(group); candidates.append(cells)
            else:
                a = values[mask]
                if not a.size and agg != 'area':
                    raise ValueError('This statistic requires a nonempty selection.')
                if agg == 'mean': value = float(a.mean())
                elif agg == 'std': value = float(a.std(ddof=0))
                elif agg == 'minimum': value = float(a.min())
                elif agg == 'maximum': value = float(a.max())
                elif agg == 'area': value = area(mask, step*step)
                elif agg == 'pearson': value = pearson(values, posfield, mask)[0]
                else: raise ValueError('Unsupported statistic.')
                numbers.append(value)
                if selection == 'windows': candidates.append(positions[index])
                elif agg in ('minimum', 'maximum'): candidates.append(np.argwhere(mask & (values == value)).tolist())
                else: candidates.append([])

    label = {'identity': 'value', 'mean': 'arithmetic mean', 'std': 'population standard deviation',
             'range': 'minimum and maximum values', 'extent_x': 'minimum and maximum x coordinates',
             'extent_y': 'minimum and maximum y coordinates', 'minimum': 'minimum value', 'maximum': 'maximum value',
             'area': 'area in squared coordinate units',
             'pearson': f'Pearson correlation with normalized {params.get("correlation_direction", "x")} coordinates',
             'components': '4-connected components (shared edges only; boundaries included)'}[agg]
    subject = '; '.join(descriptions)
    prefix = 'Use the gradient magnitude of the displayed scalar field for this task. ' if op == 'gradient' else ''
    if op == 'gradient':
        context['gradient_definition'] = 'Largest absolute difference to an existing horizontal/vertical neighbor, divided by spacing; computed before selection.'
    if selection in ('region', 'regions', 'inside_outside', 'windows'):
        context['region_inclusion'] = 'Sample centers inside the rectangle; equal sample weights.'
    quantity = {'mean': 'mean value', 'std': 'population standard deviation',
                'range': 'value range', 'area': 'area'}.get(agg,label)
    multiple_intervals = is_interval and selection in ('regions','inside_outside','windows')
    if multiple_intervals:
        measure = str(rng.choice(INTERVAL_MEASURES))
        params['interval_measure'] = measure
        numbers = [interval_width(pair) if measure == 'width' else pair[0 if measure == 'minimum' else 1]
                   for pair in intervals]
        if agg == 'range':
            quantity = {'minimum':'minimum value','maximum':'maximum value',
                        'width':'value span (maximum minus minimum)'}[measure]
        else:
            axis = 'x' if agg == 'extent_x' else 'y'
            quantity = (('horizontal' if axis == 'x' else 'vertical')+' span' if measure == 'width'
                        else f'{measure} {axis} coordinate')
    if is_interval:
        context['interval_order'] = ['minimum','maximum']
        if agg != 'range':
            context['extent_support'] = ('saddle_point_coordinates' if selection == 'saddles' else
                                         'all_selected_plateau_cell_centers' if selection == 'extrema' else
                                         'selected_cell_centers')
    if is_interval and not multiple_intervals:
        lower, upper = intervals[0]
        dimension = {'range':'value','extent_x':'x coordinate','extent_y':'y coordinate'}[agg]
        if action == 'retrieve':
            answer = [lower,upper]; response = 'number_pair'
            question = f'What are the {label} for {subject}? Return [minimum, maximum].'
        elif action == 'difference':
            answer = interval_width(intervals[0]); response = 'number'
            question = f'What is the maximum {dimension} minus the minimum {dimension} for {subject}? Return one number.'
        else:
            response = 'number_or_undefined'
            answer = 'undefined' if lower == 0 else float(upper/lower)
            if lower == 0: context['undefined_reason'] = 'minimum_is_zero'
            question = (f'What is the maximum {dimension} divided by the minimum {dimension} for {subject}? '
                        'Return one number, or undefined if the minimum is zero.')
    elif agg == 'area' and action in ('difference','ratio') and len(numbers) == 1:
        total_area = area(np.ones_like(values, dtype=bool), step*step)
        context.update(comparison_reference='whole_field_area', reference_area=total_area)
        response = 'number'
        area_subject = 'region P' if selection == 'region' else subject
        if action == 'ratio':
            answer = numbers[0]/total_area
            question = f'What fraction of the whole field area is covered by {area_subject}? Return one number.'
        else:
            answer = abs(total_area-numbers[0])
            question = (f'What is the whole field area minus the area of {area_subject}? '
                        'Return one number in squared coordinate units.')
    elif action == 'retrieve':
        answer = numbers[0]; response = 'number'
        if selection == 'extrema':
            plural = 'maxima' if kind == 'maximum' else 'minima'
            extrema_subject = {'all':f'all local {plural}, including global {plural}',
                               'non_global':f'local {plural} that are not global {plural}',
                               'global':f'global {plural}'}[scope]
            question = f'What is the mean value of {extrema_subject}? Give each 8-connected plateau equal weight.'
        elif selection == 'saddles': question = 'What is the mean value at the saddle points? Give each saddle point equal weight.'
        elif selection == 'point': question = 'What is the value at point P?'
        elif selection == 'region': question = f'What is the {quantity} in region P?'
        elif selection == 'field': question = f'What is the {quantity} across the whole field?'
        else: question = f'What is the {label} for {subject}?'
        question += ' Return one number.'
    elif action in ('difference','ratio') and is_objects:
        object_values = [obj['value'] for obj in selected_objects]
        context.update(selected_object_count=len(object_values),comparison_values='highest_and_lowest_scalar_values')
        response = 'number_or_undefined'
        if len(object_values) < 2:
            answer = 'undefined'
            context['undefined_reason'] = 'fewer_than_two_objects'
        else:
            lowest,highest = min(object_values),max(object_values)
            if action == 'ratio' and lowest == 0:
                answer = 'undefined'
                context['undefined_reason'] = 'lowest_value_is_zero'
            else:
                answer = float(highest-lowest if action == 'difference' else highest/lowest)
        object_subject = 'the saddle points' if selection == 'saddles' else subject
        if action == 'difference':
            question = f'What is the difference between the highest and lowest scalar values at {object_subject}? Return one number, or undefined if fewer than two exist.'
        else:
            question = f'What is the highest scalar value divided by the lowest at {object_subject}? Return one number, or undefined if fewer than two exist or the lowest value is zero.'
    elif action in ('compare', 'difference', 'ratio'):
        p, q = numbers
        pair = ('the values at points P and Q' if selection == 'points' else
                f'the {quantity} inside and outside rectangle P' if selection == 'inside_outside' else
                f'the {quantity} in regions P and Q')
        if action == 'compare':
            epsilon = float(max(abs(p), abs(q), 1)*1e-10)
            context['comparison_equality_epsilon'] = epsilon
            answer = 'Equal' if abs(p-q) <= epsilon else 'P' if p > q else 'Q'; response = 'category'
            if selection == 'points': question = 'Which point has the higher value, P or Q?'
            elif selection == 'inside_outside': question = f'Is the {quantity} higher inside or outside rectangle P?'
            else:
                adjective = 'higher' if agg in ('mean','minimum','maximum','pearson') else 'larger'
                question = f'Which region has the {adjective} {quantity}, P or Q?'
            question += (' Answer P (inside), Q (outside), or Equal.' if selection == 'inside_outside'
                         else ' Answer P, Q, or Equal.')
        elif action == 'difference':
            answer = abs(p-q); response = 'number'
            question = f'What is the absolute difference between {pair}? Return one number.'
        else:
            if min(p,q) <= 0: raise ValueError('Larger-to-smaller ratios require two positive numbers.')
            answer = max(p,q)/min(p,q); response = 'number'
            question = f'What is the ratio of the larger to the smaller of {pair}? Return one number.'
    elif action in ('count', 'exists'):
        count = len(collections[0]); answer = count if action == 'count' else bool(count)
        response = 'integer' if action == 'count' else 'boolean'
        entity = label if agg == 'components' else 'objects' if is_objects else 'cells'
        question = f'For {subject}, ' + (f'how many {entity} are there? Return an integer.' if action == 'count' else f'do any {entity} exist? Return true or false.')
        if selection == 'saddles':
            question = ('How many saddle points are in the field? Return an integer.' if action == 'count'
                        else 'Does the field contain any saddle points? Return true or false.')
    elif action == 'classify':
        answer = category(numbers[0], .1); context['correlation_band'] = .1; response = 'category'
        question = f'For {subject}, classify the {label}. Return positive (r > 0.1), negative (r < −0.1), or uncorrelated (−0.1 ≤ r ≤ 0.1).'
    elif selection == 'saddles':
        if not selected_objects: raise ValueError('A location task requires at least one valid location.')
        answer = {'accepted_xy':[obj['xy'] for obj in selected_objects],
                  'patches_row_col':[obj['patch_row_col'] for obj in selected_objects]}
        response = 'coordinates' if action == 'coordinates' else 'point'
        question = ('What are the x and y coordinates of a saddle point? Return one [x, y] pair. Any saddle point is accepted.'
                    if action == 'coordinates' else 'Click a saddle point in the field.')
    elif action == 'coordinates':
        response = 'coordinates'
        if selection == 'extrema':
            cells = candidates[0]
            if not cells: raise ValueError('A location task requires at least one valid location.')
            answer = {'accepted_cells_row_col': cells, 'accepted_xy': [xy[r,c].tolist() for r,c in cells]}
            plural = 'maxima' if kind == 'maximum' else 'minima'
            location_subject = {'all':f'a local {kind} (global {plural} included)',
                                'non_global':f'a local {kind} that is not global',
                                'global':f'a global {kind}'}[scope]
            question = (f'What are the x and y coordinates of {location_subject}? Return one [x, y] pair. '
                        'For a plateau, any cell center is accepted. Include field boundaries.')
        else:
            r,c = candidates[0][0]; answer = xy[r,c].tolist()
            question = 'What are the x and y coordinates of point P? Return [x, y].'
    else:
        if selection == 'windows':
            extremum = str(rng.choice(['minimum', 'maximum'])); params['find_extremum'] = extremum
            optimum = min(numbers) if extremum == 'minimum' else max(numbers)
            cells = [cell for v, group in zip(numbers,candidates) if v == optimum for cell in group]
            order = 'lowest' if extremum == 'minimum' else 'highest'
            question = f'Click the center of a square window with the {order} {quantity}. '
            question += (f'Use a side length of {context["window_side_length"]:.10g} in the plotted axis units; the ruler shows this length. '
                         if params['window_size_cue'] == 'axis_scale' else 'Use the same size as reference square R. ')
            question += 'The window must fit entirely inside the field.'
        else:
            cells = candidates[0]
            question = f'For {subject}, click the center of any ' + (f'cell attaining the {label}.' if agg in ('minimum','maximum') else 'qualifying cell.') + ' All ties are accepted.'
        if not cells: raise ValueError('A click task requires at least one valid location.')
        answer = {'accepted_cells_row_col': cells, 'accepted_xy': [xy[r,c].tolist() for r,c in cells]}; response = 'point'
    if selection == 'saddles':
        question += ' Consider only isolated saddle points inside interpolation patches; exclude patch edges and grid vertices.'
    if agg in ('extent_x','extent_y') and selection != 'saddles':
        question += ' Use cell-center coordinates.'
    if response in ('number','number_or_undefined') and answer != 'undefined':
        if not np.isfinite(answer): raise ValueError('Answer is not finite.')
    elif response == 'number_pair':
        if not np.isfinite(answer).all(): raise ValueError('Answer is not finite.')
    answer_spec = {'type':response}
    if response == 'number_pair':
        answer_spec.update(length=2,order=['minimum','maximum'],
                           quantity={'range':'scalar_value','extent_x':'x_coordinate','extent_y':'y_coordinate'}[agg])
    if response == 'number_or_undefined': answer_spec['undefined_token'] = 'undefined'
    return dict(parameters=params, parameter_summary=summarize_parameters(params, context), context=context,
                prompt=prefix+question, answer_spec=answer_spec, ground_truth=answer), values, overlays


def render(values, overlays, extent, cmap):
    has_window_cue = any(kind in ('window_scale','reference_box') for kind,_,_ in overlays)
    fig = Figure(figsize=(6.4,5.8 if has_window_cue else 5.1), dpi=110)
    ax = fig.subplots()
    # Rasterize the scalar field first. Never interpolate already-mapped colors:
    # colors between LUT entries need not represent the intermediate scalar.
    display_values = scalar_raster(values, extent)
    im = ax.imshow(display_values, origin='lower', interpolation='nearest',
                   interpolation_stage='data', extent=[*extent,*extent], cmap=cmap,
                   vmin=float(values.min()), vmax=float(values.max()))
    fig.colorbar(im, ax=ax, label='Scalar value')
    ax.set(xlabel='x', ylabel='y')
    import matplotlib.patheffects as pe
    effects = [pe.withStroke(linewidth=3, foreground='white')]
    for kind, label, position in overlays:
        if kind == 'window_scale': continue
        if kind in ('box','reference_box'):
            x0,x1,y0,y1 = position
            patch = Rectangle((x0,y0),x1-x0,y1-y0,fill=False,color='#c026d3',linewidth=1.5)
            if kind == 'reference_box': patch.set_linestyle('--')
            patch.set_path_effects(effects); ax.add_patch(patch); x,y = x0,y1
        else:
            x,y = position
            ax.plot(x,y,marker='o',markersize=9,markerfacecolor='none',markeredgewidth=1.5,
                    markeredgecolor='#c026d3',path_effects=effects)
        ax.annotate(label,(x,y),xytext=(5,5),textcoords='offset points',color='#701a75',weight='bold',path_effects=effects)
    fig.tight_layout()
    if has_window_cue:
        fig.subplots_adjust(bottom=.24)
        for kind,label,position in overlays:
            if kind == 'window_scale':
                side = position
                center = sum(extent)/2
                x0,x1 = center-side/2,center+side/2
                transform = ax.get_xaxis_transform()
                ax.plot([x0,x1],[-.21,-.21],color='#263e32',linewidth=1.7,transform=transform,clip_on=False)
                for x in (x0,x1):
                    ax.plot([x,x],[-.225,-.195],color='#263e32',linewidth=1.7,transform=transform,clip_on=False)
                ax.text(center,-.25,f'Window side: {side:.10g} axis units',ha='center',va='top',fontsize=10,
                        transform=transform,clip_on=False)
            elif kind == 'reference_box':
                ax.text(.5,-.25,'R shows the required window size',ha='center',va='top',fontsize=10,
                        transform=ax.transAxes,clip_on=False)
    buf=io.BytesIO(); fig.savefig(buf,format='png'); return buf.getvalue()
