"""Compatibility is inferred from stage contracts, never a recipe whitelist."""
from dataclasses import dataclass
from itertools import product

INTERVAL_AGGREGATIONS = ('range', 'extent_x', 'extent_y')
INTERVAL_MEASURES = ('minimum', 'maximum', 'width')

OPTIONS = {
    'operation': {'identity': 'Original scalar field', 'gradient': 'Gradient magnitude'},
    'selection': {'field': 'Whole field', 'point': 'One marked cell', 'points': 'Two marked cells',
                  'region': 'One rectangle', 'regions': 'Two rectangles', 'inside_outside': 'Inside / outside rectangle',
                  'threshold': 'Cells above threshold', 'below_threshold': 'Cells below threshold',
                  'between_thresholds': 'Cells between thresholds',
                  'windows': 'All square windows', 'extrema': 'Extrema', 'saddles': 'Saddle points'},
    'aggregation': {'identity': 'No reduction', 'mean': 'Mean', 'std': 'Population SD', 'range': 'Range',
                    'extent_x': 'Extent X', 'extent_y': 'Extent Y',
                    'minimum': 'Minimum', 'maximum': 'Maximum', 'area': 'Area',
                    'pearson': 'Pearson correlation', 'components': 'Connected components'},
    'action': {'retrieve': 'Retrieve values', 'coordinates': 'Retrieve coordinates', 'compare': 'Compare order',
               'difference': 'Absolute difference', 'ratio': 'Ratio', 'find': 'Find + click',
               'count': 'Count', 'exists': 'Test existence', 'classify': 'Classify correlation'},
}

@dataclass(frozen=True)
class Shape:
    kind: str
    groups: str = 'one'
    singleton: bool = False
    locations: bool = True
    scalar_values: bool = False
    reference: str | None = None


def selection_output(selection):
    return Shape('objects' if selection in ('extrema','saddles') else 'cells',
                 'two' if selection in ('points', 'regions', 'inside_outside') else 'many' if selection == 'windows' else 'one',
                 selection in ('point', 'points'), scalar_values=True)


def aggregation_output(shape, aggregation):
    if aggregation == 'identity':
        return Shape('scalar', shape.groups, True, True) if shape.singleton else shape
    if shape.singleton or shape.kind in ('scalar', 'correlation'):
        raise ValueError('Aggregation requires multiple samples per group, not a single number. Choose No reduction.')
    if aggregation in INTERVAL_AGGREGATIONS:
        if aggregation == 'range' and not shape.scalar_values:
            raise ValueError('Range requires scalar values.')
        if aggregation != 'range' and not shape.locations:
            raise ValueError('Extent requires spatial locations.')
        return Shape('interval', shape.groups, locations=shape.groups == 'many', scalar_values=True)
    if aggregation == 'mean':
        if not shape.scalar_values:
            raise ValueError('Mean requires a scalar value for each selected sample or object.')
        return Shape('scalar', shape.groups, True, shape.groups == 'many', scalar_values=True)
    if shape.kind != 'cells':
        raise ValueError('This aggregation requires cell samples, not objects.')
    if aggregation == 'components':
        return Shape('objects', shape.groups)
    if aggregation == 'area':
        return Shape('scalar', shape.groups, True, shape.groups == 'many', reference='whole_field_area')
    return Shape('correlation' if aggregation == 'pearson' else 'scalar', shape.groups,
                 True, aggregation in ('minimum', 'maximum') or shape.groups == 'many')


def compatibility(operation, selection, aggregation, action):
    choices = dict(operation=operation, selection=selection, aggregation=aggregation, action=action)
    if any(v not in OPTIONS[k] for k, v in choices.items()):
        return False, 'Unknown stage option.'
    try:
        shape = aggregation_output(selection_output(selection), aggregation)
    except ValueError as exc:
        return False, str(exc)
    numeric = shape.kind in ('scalar', 'correlation')
    interval = shape.kind == 'interval'
    scalar_objects = shape.kind == 'objects' and shape.scalar_values and shape.groups == 'one'
    if action == 'retrieve': ok = (numeric or interval) and shape.groups == 'one'
    elif action == 'compare': ok = (numeric or interval) and shape.groups == 'two'
    elif action in ('difference', 'ratio'):
        ok = ((numeric and (shape.groups == 'two' or (shape.groups == 'one' and shape.reference is not None)))
              or scalar_objects or (interval and shape.groups in ('one','two')))
    elif action == 'coordinates':
        ok = aggregation == 'identity' and shape.locations and shape.groups == 'one' and (shape.singleton or scalar_objects)
    elif action == 'classify': ok = shape.kind == 'correlation' and shape.groups == 'one'
    elif action in ('count', 'exists'): ok = shape.kind in ('cells', 'objects') and shape.groups == 'one'
    elif action == 'find': ok = shape.locations and (shape.groups == 'one' or ((numeric or interval) and shape.groups == 'many'))
    else: ok = False
    return ok, '' if ok else f'{OPTIONS["action"][action]} cannot consume {shape.groups} group(s) of {shape.kind}' + (' without locations.' if not shape.locations else '.')


def combination_id(stages):
    return '|'.join(stages[key] for key in OPTIONS)


def combinations(selected, *, include_excluded=False):
    for key in OPTIONS:
        if not isinstance(selected.get(key), list) or not selected[key]:
            raise ValueError(f'Select at least one {key}.')
        if any(v not in OPTIONS[key] for v in selected[key]):
            raise ValueError(f'Unknown {key}.')
    exclusions = selected.get('excluded_combinations', [])
    if not isinstance(exclusions, list):
        raise ValueError('Excluded combinations must be a list of combination IDs.')
    for ident in exclusions:
        parts = ident.split('|') if isinstance(ident,str) else []
        if len(parts) != len(OPTIONS) or not compatibility(*parts)[0]:
            raise ValueError('Unknown or incompatible excluded combination.')
    excluded = set(exclusions)
    valid = [dict(zip(OPTIONS, values)) for values in product(*(list(dict.fromkeys(selected[k])) for k in OPTIONS))
             if compatibility(*values)[0]]
    return valid if include_excluded else [stages for stages in valid if combination_id(stages) not in excluded]
