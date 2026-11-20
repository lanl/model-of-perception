"""Pearson statistics and analytic scoring of uncertain numeric responses.

The coefficient is exact for supplied scalar readings. The observer uses the
first-order delta method, then clips its Gaussian response to [-1, 1]. This is
an approximation to reading uncertainty, not a fitted human correlation model.
"""
import math

import numpy as np


DIRECTIONS = {
    'x': 'the x coordinate (left to right)',
    'y': 'the y coordinate (bottom to top)',
    'x+y': 'position toward the top-right, measured by normalized x + y',
    'y-x': 'position toward the top-left, measured by normalized y − x',
    'x-y': 'position toward the bottom-right, measured by normalized x − y',
}
MODEL = 'pearson_delta_clipped_normal_v1'


def coordinate(xx, yy, extent, direction):
    if direction not in DIRECTIONS:
        raise ValueError('Direction must be x, y, x+y, x-y, or y-x')
    lo, hi = map(float, extent)
    if not math.isfinite(lo + hi) or hi <= lo:
        raise ValueError('Invalid coordinate extent')
    x = (np.asarray(xx, float) - lo) / (hi - lo)
    y = (np.asarray(yy, float) - lo) / (hi - lo)
    return {'x': x, 'y': y, 'x+y': x+y, 'y-x': y-x, 'x-y': x-y}[direction]


def pearson(values, positions, mask):
    """Return r and its derivative with respect to each original scalar cell."""
    a, c, m = np.asarray(values, float), np.asarray(positions, float), np.asarray(mask, bool)
    if a.shape != c.shape or a.shape != m.shape or m.sum() < 2:
        raise ValueError('Correlation requires matching arrays and at least two cells')
    if not np.isfinite(a[m]).all() or not np.isfinite(c[m]).all():
        raise ValueError('Nonfinite correlation input')
    z = a[m] - a[m].mean()
    q = c[m] - c[m].mean()
    # Scale before taking norms for unit invariance and to avoid underflow.
    zs, qs = float(np.max(abs(z))), float(np.max(abs(q)))
    if zs == 0 or qs == 0:
        raise ValueError('Pearson correlation is undefined for a constant input')
    zn, qn = z/zs, q/qs
    zl, ql = np.linalg.norm(zn), np.linalg.norm(qn)
    zu, qu = zn/zl, qn/ql
    r = float(np.clip(zu @ qu, -1, 1))
    j = np.zeros_like(a)
    j[m] = (qu-r*zu)/(zs*zl)
    return r, j


def category(r, tolerance=.1):
    if not math.isfinite(tolerance) or not 0 <= tolerance < 1:
        raise ValueError('Correlation tolerance must be finite and in [0, 1)')
    r = numeric_answer(r)
    return 'positive' if r > tolerance else 'negative' if r < -tolerance else 'uncorrelated'


def prompt(direction, tolerance=.1, numeric=False):
    category(0, tolerance)
    if not numeric:
        return (
            f'Is the scalar value positively correlated, negatively correlated, or not correlated with {DIRECTIONS[direction]}? '
            'Choose Positive, Negative, or Not correlated. Positive means values tend to increase in that direction; '
            'negative means they tend to decrease. '
            f'For this task, Not correlated means −{tolerance:g} ≤ Pearson r ≤ {tolerance:g}. '
            'Use all cells. Normalize each axis to [0, 1] before forming a diagonal coordinate. '
            'This is a linear-correlation category, not a statistical significance test.'
        )
    return (
        f'Does the scalar value correlate positively or negatively with {DIRECTIONS[direction]}, '
        'or is there little or no linear correlation? Estimate the Pearson correlation '
        'coefficient r using all equal-area cells in the image. Return one number from '
        '−1 to +1: positive means values tend to increase in the stated direction, '
        'negative means they tend to decrease, and values near zero indicate little '
        'or no linear correlation. Normalize each axis from 0 at its minimum to 1 '
        'at its maximum before forming a diagonal coordinate. Use the whole field, '
        'not just cells lying on a diagonal line.'
    )


def distribution(mean, sd):
    mean, sd = float(mean), float(sd)
    if not math.isfinite(mean) or not -1 <= mean <= 1 or not math.isfinite(sd) or sd < 0:
        raise ValueError('Invalid Pearson observer parameters')
    return dict(family='clipped_normal', location=mean, scale=sd, lower=-1., upper=1.)


def expected_absolute_error(prediction, truth):
    """Closed-form E|clip(N(mu,sd²),-1,1) - truth|; includes boundary atoms."""
    if set(prediction) != {'family', 'location', 'scale', 'lower', 'upper'}:
        raise ValueError('Invalid prediction schema')
    if prediction['family'] != 'clipped_normal' or (prediction['lower'], prediction['upper']) != (-1, 1):
        raise ValueError('Unsupported numeric distribution')
    p = distribution(prediction['location'], prediction['scale'])
    truth = numeric_answer(truth)
    mu, sd = p['location'], p['scale']
    if sd == 0:
        return abs(mu-truth)
    phi = lambda z: math.exp(-.5*z*z)/math.sqrt(2*math.pi)
    cdf = lambda z: .5*math.erfc(-z/math.sqrt(2))
    delta = mu-truth
    absolute = 2*sd*phi(delta/sd) + delta*(2*cdf(delta/sd)-1)
    lower_tail = (-1-mu)*cdf((-1-mu)/sd) + sd*phi((-1-mu)/sd)
    upper_tail = (mu-1)*cdf((mu-1)/sd) + sd*phi((mu-1)/sd)
    return float(np.clip(absolute-lower_tail-upper_tail, 0, 2))


def numeric_answer(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError('Return r as a JSON number')
    if not math.isfinite(value) or not -1 <= value <= 1:
        raise ValueError('Pearson response must be finite and in [-1, 1]')
    return float(value)
