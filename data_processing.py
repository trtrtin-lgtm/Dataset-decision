import numpy as np
import pandas as pd

AOI_BBOXES = {
    9: {'target1': [436.3, 293.9, 655.3, 465.7], 'target2': [726.8, 294.9, 971.7, 459.3], 'op1a': [215.9, 274.6, 414.6, 375.4], 'op1b': [154.0, 427.2, 417.4, 533.7], 'op2a': [987.4, 234.3, 1309.9, 412.8], 'op2b': [993.2, 432.9, 1309.9, 545.2]},
    12: {'op1a': [218.8, 270.3, 420.3, 379.7], 'op2a': [980.2, 230.0, 1223.5, 345.1], 'op1b': [228.9, 421.1, 423.2, 525.1], 'op2b': [987.4, 424.3, 1154.4, 526.5], 'target1': [447.3, 309.2, 625.2, 454.8], 'target2': [731.8, 305.2, 962.2, 450.8]},
    15: {'target1': [442.7, 307.9, 632.6, 460.0], 'target2': [732.4, 303.1, 922.3, 458.4], 'op2a': [944.3, 251.6, 1248.0, 362.4], 'op2b': [947.1, 422.9, 1252.3, 532.3], 'op1a': [175.6, 274.6, 417.4, 376.8], 'op1b': [112.3, 431.5, 414.6, 529.4]},
    22: {'target1': [441.4, 301.3, 626.3, 450.5], 'target2': [727.2, 307.2, 964.1, 454.5], 'op1a': [156.9, 199.8, 397.3, 361.0], 'op1b': [159.8, 428.6, 401.6, 654.6], 'op2a': [987.4, 182.5, 1269.6, 356.7], 'op2b': [991.8, 425.7, 1275.3, 604.2]},
    25: {'target1': [405.7, 267.8, 686.8, 504.1], 'target2': [729.3, 266.9, 976.5, 506.9], 'op1a': [146.8, 152.3, 365.6, 310.6], 'op1b': [142.5, 461.7, 359.9, 621.5], 'op2a': [1010.5, 152.3, 1276.8, 313.5], 'op2b': [1013.3, 447.3, 1299.8, 633.0]},
    28: {'target1': [411.5, 258.5, 680.2, 501.0], 'target2': [731.0, 261.0, 974.2, 503.6], 'op1a': [95.1, 137.9, 395.8, 382.6], 'op1b': [136.7, 477.6, 395.8, 664.3], 'op2a': [987.4, 139.3, 1268.1, 362.4], 'op2b': [990.3, 401.3, 1307.0, 717.9]},
    33: {'target1': [38.4, 270.6, 305.8, 510.5], 'target2': [1085.1, 280.5, 1345.8, 502.5], 'op1a': [773.0, 313.5, 1073.8, 464.6], 'op1b': [476.4, 499.2, 883.8, 658.9], 'op2b': [305.2, 316.3, 593.0, 473.2], 'op2a': [495.2, 94.7, 855.0, 268.8]},
    36: {'target1': [12.3, 257.5, 292.6, 520.7], 'target2': [1078.7, 274.9, 1340.4, 507.1], 'op2a': [541.2, 113.4, 837.7, 238.6], 'op2b': [309.5, 332.2, 591.6, 451.7], 'op1a': [781.6, 330.7, 1059.4, 447.3], 'op1b': [492.3, 516.4, 882.4, 625.8]}
}

def point_in_bbox(x, y, bbox):
    x1, y1, x2, y2 = bbox
    return (x >= x1) & (x <= x2) & (y >= y1) & (y <= y2)

def get_aoi_labels(tid, x, y):
    labels = np.full(len(x), None, dtype=object)
    if tid not in AOI_BBOXES:
        return labels
    for nm, bb in AOI_BBOXES[tid].items():
        m = (labels == None) & point_in_bbox(x, y, bb)
        labels[m] = nm
    return labels

def map_aoi_to_group(aoi_name):
    if aoi_name is None:
        return None
    name = str(aoi_name).lower()
    if "op1" in name or "target1" in name:
        return 1
    if "op2" in name or "target2" in name:
        return 2
    return None

def clean_pupil_signal(time_ms, px, py, dt_ms=16.667):
    px = np.where(px > 0, px, np.nan)
    py = np.where(py > 0, py, np.nan)
    p = np.where(np.isfinite(px) & np.isfinite(py), (px + py) / 2.0, np.where(np.isfinite(px), px, py))
    ok = np.isfinite(p)
    if ok.sum() < 20:
        return np.full_like(p, np.nan)
    med = np.nanmedian(p)
    mad = np.nanmedian(np.abs(p - med)) * 1.4826
    ok &= np.abs(p - med) <= 4.0 * max(mad, 1e-6)
    d = np.abs(np.diff(p, prepend=np.nan))
    d2 = np.abs(np.diff(p, append=np.nan))
    sp = np.fmax(d, d2)
    sm = np.nanmedian(sp[ok])
    smad = np.nanmedian(np.abs(sp[ok] - sm)) * 1.4826
    ok &= ~(sp > sm + 6.0 * max(smad, 1e-6))
    bad = ~ok
    pad = int(round(100.0 / dt_ms))
    if bad.any():
        k = np.convolve(bad.astype(int), np.ones(2 * pad + 1, int), mode='same') > 0
        ok = ok & ~k
    q = np.where(ok, p, np.nan)
    s = pd.Series(q, index=time_ms)
    gap = s.isna()
    grp = (gap != gap.shift()).cumsum()
    glen = gap.groupby(grp).transform('sum') * dt_ms
    inside = s.ffill().notna() & s.bfill().notna()
    interp = s.interpolate(method='index', limit_area='inside')
    fill = gap & (glen <= 500.0) & inside
    out = s.copy()
    out[fill] = interp[fill]
    out = out.rolling(5, center=True, min_periods=3).mean().where(out.notna())
    return out.to_numpy()

def pfe_correction(pupil, gx, gy):
    m = np.isfinite(pupil) & np.isfinite(gx) & np.isfinite(gy)
    if m.sum() < 30:
        return pupil
    X = np.column_stack([np.ones(m.sum()), gx[m], gy[m], gx[m]**2, gy[m]**2, gx[m]*gy[m]])
    b, *_ = np.linalg.lstsq(X, pupil[m], rcond=None)
    pred = X @ b
    res = pupil[m] - pred
    out = pupil.copy()
    out[m] = res + np.nanmean(pupil[m])
    return out

def response_locked_binning(time_ms, grp_labels, window_ms=3000.0, num_bins=15):
    t_end = time_ms[-1]
    t_start = t_end - window_ms
    bin_width = window_ms / num_bins
    dt = np.diff(time_ms, append=time_ms[-1])
    dt = np.clip(dt, 0.0, 100.0)
    bin_idx = np.floor((time_ms - t_start) / bin_width).astype(int)
    p1 = np.zeros(num_bins)
    p2 = np.zeros(num_bins)
    r = np.zeros(num_bins)
    for b in range(num_bins):
        mask = (bin_idx == b)
        if not np.any(mask):
            continue
        dt_b = dt[mask]
        grp_b = grp_labels[mask]
        t_total = float(np.sum(dt_b))
        if t_total <= 0:
            continue
        p1[b] = float(np.sum(dt_b[grp_b == 1])) / t_total
        p2[b] = float(np.sum(dt_b[grp_b == 2])) / t_total
        r[b] = p1[b] + p2[b]
    return p1, p2, r

def check_quality_gate(r_bins, min_bin_ratio=0.30, min_valid_bins=3):
    valid_bins = (r_bins >= min_bin_ratio)
    trial_ratio = float(np.mean(r_bins))
    is_ok = (valid_bins.sum() >= min_valid_bins) and (trial_ratio >= min_bin_ratio)
    return is_ok, valid_bins
