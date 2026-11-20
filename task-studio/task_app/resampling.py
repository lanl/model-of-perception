"""Area averaging of scalar samples, before color mapping."""
import numpy as np


def downsample(values, max_side=100):
    """Average overlapping source cells; never upsample, crop, or normalize values.

    Samples represent equal-area cells for this preparation step. Fractional
    overlap weights also handle sizes that are not multiples of the output size.
    """
    a = np.asarray(values)
    if a.ndim != 2 or a.dtype.kind not in 'fiu' or min(a.shape) < 3:
        raise ValueError('Expected a numeric two-dimensional field, at least 3 × 3.')
    if not np.isfinite(a).all():
        raise ValueError('Missing or infinite values require an explicit mask policy.')
    if type(max_side) is not int or not 3 <= max_side <= 100:
        raise ValueError('Maximum side must be an integer from 3 to 100.')
    a = a.astype(np.float64)
    scale = min(1., max_side / max(a.shape))
    shape = tuple(max(1, round(size * scale)) for size in a.shape)
    if min(shape) < 3:
        raise ValueError('The aspect ratio would leave fewer than three samples on one axis.')
    if shape == a.shape:
        return a.copy()

    def weights(old, new):
        edges = np.linspace(0., old, new + 1)
        left = np.maximum(edges[:-1, None], np.arange(old)[None, :])
        right = np.minimum(edges[1:, None], np.arange(1, old + 1)[None, :])
        return np.maximum(0., right - left) / (old / new)

    return weights(a.shape[0], shape[0]) @ a @ weights(a.shape[1], shape[1]).T
