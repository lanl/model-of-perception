"""Nominal parameter-space sizes; data-validity filters can reduce these bounds."""
from math import comb
from .contracts import INTERVAL_AGGREGATIONS, INTERVAL_MEASURES


def describe(stages, dataset):
    n = len(dataset['values'])
    selection, aggregation, action = (stages[k] for k in ('selection','aggregation','action'))
    factors = []
    def discrete(name, count): factors.append(dict(name=name,kind='discrete',choices=str(count)))
    def continuous(name, dimensions=1): factors.append(dict(name=name,kind='continuous',dimensions=dimensions))
    if selection == 'point': discrete('Grid location P',n*n)
    elif selection == 'points': discrete('Ordered distinct grid locations P, Q',n*n*(n*n-1))
    elif selection in ('region','regions','inside_outside'):
        rectangles = comb(n+1,2)**2
        discrete('Rectangle P',rectangles)
        if selection == 'regions': discrete('Rectangle Q',rectangles)
    elif selection == 'windows':
        discrete('Odd window width (3 through grid width)',(n-1)//2)
        discrete('Window size cue: axis scale / reference square',2)
    elif selection in ('threshold','below_threshold'): continuous('Threshold')
    elif selection == 'between_thresholds':
        continuous('Lower threshold'); continuous('Upper threshold')
    elif selection == 'extrema':
        discrete('Minimum / maximum',2); discrete('All / non-global / global',3)
    if aggregation == 'pearson': discrete('Correlation direction',4)
    if aggregation in INTERVAL_AGGREGATIONS and selection in ('regions','inside_outside','windows'):
        discrete('Use interval minimum / maximum / width',len(INTERVAL_MEASURES))
    if selection == 'windows' and action == 'find': discrete('Minimum / maximum window statistic',2)
    size = 1
    for factor in factors:
        if factor['kind'] == 'discrete': size *= int(factor['choices'])
    return dict(continuous_dimensions=sum(f.get('dimensions',0) for f in factors),
                discrete_settings=str(size), factors=factors)


def summarize(combos, selected, assets):
    datasets = list(dict.fromkeys(selected.get('datasets',[])))
    colormaps = list(dict.fromkeys(selected.get('colormaps',[])))
    if not datasets or any(v not in assets.datasets for v in datasets): raise ValueError('Select valid datasets.')
    if not colormaps or any(v not in assets.maps for v in colormaps): raise ValueError('Select valid colormaps.')
    rows=[]
    by_dimension={}
    for dataset_id in datasets:
        dataset=assets.datasets[dataset_id]
        for stages in combos:
            space=describe(stages,dataset)
            rows.append(dict(dataset=dataset['name'],stages=stages,**space))
            d=space['continuous_dimensions']
            by_dimension[d]=by_dimension.get(d,0)+int(space['discrete_settings'])*len(colormaps)
    return dict(rows=rows,colormap_choices=len(colormaps),
                totals=[dict(continuous_dimensions=d,discrete_settings=str(size)) for d,size in sorted(by_dimension.items())],
                note='Upper bounds before data-validity checks, including selected datasets and colormaps. Points and rectangles currently sample grid locations, not continuous coordinates. Fixed settings and derived values are not independent parameters.')
