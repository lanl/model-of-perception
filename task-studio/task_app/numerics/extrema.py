"""Scalar-data definitions shared by truth generation and image-only tools."""
import numpy as np


VARIANTS = (
    ('presence_non_global_maximum', 'Non-global local maximum present', 'maximum', 'non_global', 'exists'),
    ('presence_non_global_minimum', 'Non-global local minimum present', 'minimum', 'non_global', 'exists'),
    ('count_all_maximum', 'Total local and global maxima', 'maximum', 'all', 'count'),
    ('count_all_minimum', 'Total local and global minima', 'minimum', 'all', 'count'),
    ('count_non_global_maximum', 'Non-global local maxima count', 'maximum', 'non_global', 'count'),
    ('count_non_global_minimum', 'Non-global local minima count', 'minimum', 'non_global', 'count'),
    ('count_global_maximum', 'Global maxima count', 'maximum', 'global', 'count'),
    ('count_global_minimum', 'Global minima count', 'minimum', 'global', 'count'),
)


def offsets(connectivity):
    if type(connectivity) is not int or connectivity not in (4, 8):
        raise ValueError('Connectivity must be 4 or 8')
    return [(dy,dx) for dy in (-1,0,1) for dx in (-1,0,1)
            if (dy or dx) and (connectivity == 8 or abs(dy)+abs(dx)==1)]


def objects(values, valid=None, kind='maximum', connectivity=8):
    """Connected equal-valued plateaus; compare their full outside boundary.

    Global is relative to the selected support. Boundary objects use only valid
    neighbors. An isolated/constant connected support counts as one maximum and
    one minimum. Equality is exact; no hidden smoothing or prominence threshold.
    """
    a=np.asarray(values,float)
    m=np.ones_like(a,bool) if valid is None else np.asarray(valid,bool)
    if a.ndim!=2 or a.shape!=m.shape or not m.any() or not np.isfinite(a[m]).all():
        raise ValueError('Expected a nonempty finite 2D scalar support')
    if kind not in ('maximum','minimum'): raise ValueError('Choose maximum or minimum')
    neighbors=offsets(connectivity); seen=np.zeros_like(m); result=[]
    extreme=float(a[m].max() if kind=='maximum' else a[m].min())
    for r,c in np.argwhere(m):
        if seen[r,c]: continue
        value=float(a[r,c]); todo=[(int(r),int(c))]; seen[r,c]=True; cells=[]; is_extreme=True
        while todo:
            y,x=todo.pop(); cells.append([y,x])
            for dy,dx in neighbors:
                yy,xx=y+dy,x+dx
                if not (0<=yy<a.shape[0] and 0<=xx<a.shape[1] and m[yy,xx]): continue
                other=a[yy,xx]
                if other==value:
                    if not seen[yy,xx]: seen[yy,xx]=True; todo.append((yy,xx))
                elif (other>value if kind=='maximum' else other<value): is_extreme=False
        if is_extreme:
            result.append(dict(cells=sorted(cells),value=value,global_extremum=value==extreme))
    return result


def counts(fields, valid, kind='maximum', scope='all', connectivity=8):
    """Joint field counts, vectorized when there are no adjacent equal values.

    Plateau cases use the same component algorithm as scalar ground truth.
    The vectorized route preserves shared-neighbor dependence across extrema.
    """
    a=np.asarray(fields,float); m=np.asarray(valid,bool)
    if a.ndim==2: a=a[None]
    if a.ndim!=3 or not len(a) or a.shape[1:]!=m.shape or not m.any() or not np.isfinite(a[:,m]).all():
        raise ValueError('Invalid joint fields or selection')
    if kind not in ('maximum','minimum') or scope not in ('all','non_global','global'):
        raise ValueError('Invalid extremum kind or scope')
    neighbor_offsets=offsets(connectivity)
    # Bound temporary array sizes for the observer's numerical integration.
    if len(a)>256:
        return np.concatenate([counts(a[i:i+256],m,kind,scope,connectivity) for i in range(0,len(a),256)])
    candidates=np.broadcast_to(m,a.shape).copy(); tied=np.zeros(len(a),bool)
    h,w=m.shape
    for dy,dx in neighbor_offsets:
        sy=slice(max(0,-dy),min(h,h-dy)); sx=slice(max(0,-dx),min(w,w-dx))
        ty=slice(max(0,dy),min(h,h+dy)); tx=slice(max(0,dx),min(w,w+dx))
        pair=m[sy,sx]&m[ty,tx]
        center,other=a[:,sy,sx],a[:,ty,tx]
        candidates[:,sy,sx] &= ~pair | (center>=other if kind=='maximum' else center<=other)
        tied |= np.any(pair & (center==other),axis=(1,2))
    extrema=(a[:,m].max(axis=1) if kind=='maximum' else a[:,m].min(axis=1))[:,None,None]
    if scope=='global': candidates &= a==extrema
    elif scope=='non_global': candidates &= a!=extrema
    result=candidates.sum(axis=(1,2))
    cache={}
    for index in np.flatnonzero(tied):
        key=a[index,m].tobytes()
        if key not in cache:
            found=objects(a[index],m,kind,connectivity)
            cache[key]=sum(scope=='all' or (obj['global_extremum'] if scope=='global' else not obj['global_extremum']) for obj in found)
        result[index]=cache[key]
    return result.astype(int)


def validate_answer(value, response):
    expected=response['type']
    if expected=='boolean':
        if type(value) is not bool: raise ValueError('Presence requires JSON true or false')
    elif expected=='integer':
        if type(value) is not int or not 0<=value<=response['maximum']:
            raise ValueError('Count requires a nonnegative JSON integer within the stated bound')
    else: raise ValueError('Unsupported response type')
    return value


def prompt(kind,scope,operation,connectivity=8):
    offsets(connectivity)
    if kind not in ('maximum','minimum') or scope not in ('all','non_global','global') or operation not in ('count','exists'):
        raise ValueError('Invalid extremum kind, scope or operation')
    if operation=='exists' and scope!='non_global':
        raise ValueError('Presence questions require non_global scope')
    noun='maxima' if kind=='maximum' else 'minima'
    singular='maximum' if kind=='maximum' else 'minimum'
    if operation=='exists':
        question=f'Are there any local {noun} that are not global {noun} in this scalar field? Answer true or false.'
    elif scope=='all':
        question=f'How many local {noun} are there in total, including all global {noun}? Count each extremum once; do not add global extrema again. Return an integer.'
    elif scope=='global':
        question=f'How many separate global {noun} are there in this scalar field? Count disconnected plateaus attaining the global {singular} separately. Return an integer.'
    else:
        question=f'How many local {noun} are not global {noun}? Return an integer.'
    relation='higher' if kind=='maximum' else 'lower'
    directions='horizontal, vertical and diagonal' if connectivity==8 else 'horizontal and vertical'
    return question+(
        f' Use {connectivity}-connectivity: the immediate {directions} neighbors. '
        f'A local {singular} is a cell or connected equal-valued plateau whose value is strictly '
        f'{relation} than every neighboring cell outside that plateau. Count a plateau once, '
        f'not once per cell. Include boundary extrema using only existing neighbors. Global '
        f'means attaining the {"largest" if kind=="maximum" else "smallest"} scalar value anywhere '
        'in the displayed field. A constant whole field counts as one global maximum and one '
        'global minimum, with no non-global extrema. Use the cell values directly, without '
        'smoothing or a prominence threshold.'
    )
