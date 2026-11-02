"""
Optimized Affiliation metrics — drop-in replacement for pr_from_events.

Optimizations vs original metrics.py
-------------------------------------
1. _partition_fast: binary search (bisect) instead of O(|E_gt|×|Is|) full scan
   Original bug: Is_j = [x for x, y in zip(Is, kept_index)]  ← y never used, no filter
   Fixed + O((|E_gt|+|Is|)·log|Is|) instead of O(|E_gt|×|Is|)

2. _precision_both: one pass over Is computes both distance and proba
   (original iterates Is separately in affiliation_precision_distance + affiliation_precision_proba)

3. _recall_both: filters Is once, shared between distance and proba sub-computations
   (original filters Is independently in two separate function calls)
"""

import bisect
import math

from metrics.affiliation.metrics import test_events
from metrics.affiliation.generics import (
    infer_Trange,
    has_point_anomalies,
    _len_wo_nan,
    _sum_wo_nan,
)
from metrics.affiliation._affiliation_zone import get_all_E_gt_func
from metrics.affiliation._integral_interval import (
    interval_intersection,
    interval_length,
    sum_interval_lengths,
    integral_interval_distance,
    integral_interval_probaCDF_precision,
    integral_interval_probaCDF_recall,
)


# ---------------------------------------------------------------------------
# Core optimization 1: binary-search partition
# ---------------------------------------------------------------------------

def _partition_fast(Is, E_gt):
    """
    Optimized affiliation_partition using binary search.

    Complexity
        Original : O(|E_gt| × |Is|)  — linear scan of all Is for every GT zone
        This     : O((|E_gt| + |Is|) × log|Is|)  — bisect on sorted Is endpoints

    Original also has a bug:
        Is_j = [x for x, y in zip(Is, kept_index)]   # y is never used → no filtering

    Returns
        List of length |E_gt|.  Each entry is a (possibly empty) list of non-None
        interval_intersection results — equivalent to the original's per-zone list
        but with Nones removed.  Downstream functions already handle None-free lists
        (affiliation_recall_* filter out Nones themselves; affiliation_precision_*
        treat None as zero-length → zero contribution to sums).
    """
    if not Is:
        return [[] for _ in E_gt]

    Is_starts = [I[0] for I in Is]
    Is_ends   = [I[1] for I in Is]

    result = []
    for E in E_gt:
        e0, e1 = E
        # I overlaps E non-trivially  ⟺  I[0] < e1  AND  I[1] > e0
        lo = bisect.bisect_right(Is_ends,   e0)   # first I with I[1] > e0
        hi = bisect.bisect_left(Is_starts,  e1)   # first I with I[0] >= e1
        result.append([interval_intersection(I, E) for I in Is[lo:hi]])
    return result


# ---------------------------------------------------------------------------
# Core optimization 2: combined precision (single pass over Is)
# ---------------------------------------------------------------------------

def _precision_both(Is_j, J, E):
    """
    Compute precision distance + precision proba in one pass over Is_j.

    Original: two separate functions each iterating Is_j independently.
    """
    active = [I for I in Is_j if I is not None]
    if not active:
        return math.nan, math.nan
    total_len = sum_interval_lengths(active)
    if total_len == 0:
        return math.nan, math.nan

    d_sum = 0.0
    p_sum = 0.0
    for I in active:
        d_sum += integral_interval_distance(I, J)
        p_sum += integral_interval_probaCDF_precision(I, J, E)

    return d_sum / total_len, p_sum / total_len


# ---------------------------------------------------------------------------
# Core optimization 3: combined recall (filter Is once)
# ---------------------------------------------------------------------------

def _recall_both(Is_j, J, E):
    """
    Compute recall distance + recall proba, filtering Is once and reusing it.

    Original: two separate functions that independently filter Is and call
    get_all_E_gt_func + affiliation_partition each.

    Note: distance uses range (-inf, +inf) for Voronoi zones;
          proba   uses bounded E  — so two separate get_all_E_gt_func calls
          are still needed, but Is filtering and _partition_fast calls are shared.
    """
    Is = [I for I in Is_j if I is not None]
    if not Is:
        return math.inf, 0.0

    J_len = interval_length(J)

    # Recall distance — unbounded Voronoi zones
    E_gt_dist  = get_all_E_gt_func(Is, (-math.inf, math.inf))
    Js_dist    = _partition_fast([J], E_gt_dist)
    d_sum = sum(
        integral_interval_distance(segs[0], I)
        for I, segs in zip(Is, Js_dist) if segs
    )

    # Recall proba — zones bounded by E
    E_gt_proba = get_all_E_gt_func(Is, E)
    Js_proba   = _partition_fast([J], E_gt_proba)
    p_sum = sum(
        integral_interval_probaCDF_recall(I, segs[0], E)
        for I, segs in zip(Is, Js_proba) if segs
    )

    return d_sum / J_len, p_sum / J_len


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def pr_from_events_fast(events_pred, events_gt, Trange):
    """
    Optimized drop-in replacement for pr_from_events.

    Same interface, same output dict format, faster execution.
    """
    # Validation — identical to original
    test_events(events_pred)
    test_events(events_gt)
    minimal_Trange = infer_Trange(events_pred, events_gt)
    if not Trange[0] <= minimal_Trange[0]:
        raise ValueError('`Trange` should include all the events')
    if not minimal_Trange[1] <= Trange[1]:
        raise ValueError('`Trange` should include all the events')
    if len(events_gt) == 0:
        raise ValueError('Input `events_gt` should have at least one event')
    if has_point_anomalies(events_pred) or has_point_anomalies(events_gt):
        raise ValueError('Cannot manage point anomalies currently')
    if Trange is None:
        raise ValueError('Trange should be indicated')

    # Core
    E_gt          = get_all_E_gt_func(events_gt, Trange)
    aff_partition = _partition_fast(events_pred, E_gt)

    d_precision, p_precision = [], []
    d_recall,    p_recall    = [], []

    for Is_j, J, E in zip(aff_partition, events_gt, E_gt):
        dp, pp = _precision_both(Is_j, J, E)
        dr, pr = _recall_both(Is_j, J, E)
        d_precision.append(dp)
        p_precision.append(pp)
        d_recall.append(dr)
        p_recall.append(pr)

    # Aggregate — identical to original
    if _len_wo_nan(p_precision) > 0:
        p_precision_avg = _sum_wo_nan(p_precision) / _len_wo_nan(p_precision)
    else:
        p_precision_avg = p_precision[0]   # math.nan
    p_recall_avg = sum(p_recall) / len(p_recall)

    return {
        'precision':                          p_precision_avg,
        'recall':                             p_recall_avg,
        'individual_precision_probabilities': p_precision,
        'individual_recall_probabilities':    p_recall,
        'individual_precision_distances':     d_precision,
        'individual_recall_distances':        d_recall,
    }
