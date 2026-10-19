import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score

def count_gaze_switches(aoi_sequence):
    switches = 0
    prev_group = None
    for aoi in aoi_sequence:
        if aoi is None:
            continue
        name = str(aoi).lower()
        if "op1" in name or "target1" in name:
            grp = 1
        elif "op2" in name or "target2" in name:
            grp = 2
        else:
            continue
        if prev_group is not None and grp != prev_group:
            switches += 1
        prev_group = grp
    return switches

def calc_late_pupil_dilation(pupil_z, time_ms, window_ms=1000.0):
    t_end = time_ms[-1]
    mask_late = (time_ms >= t_end - window_ms)
    mask_base = (time_ms < t_end - window_ms)
    p_late = pupil_z[mask_late]
    p_base = pupil_z[mask_base]
    if len(p_late) == 0 or len(p_base) == 0:
        return 0.0
    val_late = float(np.nanmean(p_late))
    val_base = float(np.nanmean(p_base))
    return val_late - val_base

def calc_dci(z_switches, z_mean_pupil, z_delta_pupil):
    z_composite = 0.50 * z_switches + 0.30 * z_mean_pupil + 0.20 * z_delta_pupil
    return float(100.0 / (1.0 + np.exp(-z_composite)))

def classify_hesitation_level(dci_score):
    if dci_score <= 40.0:
        return 1
    elif dci_score <= 60.0:
        return 2
    elif dci_score <= 75.0:
        return 3
    else:
        return 4

def evaluate_early_prediction(X, y, n_splits=5, random_state=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    accs = []
    aucs = []
    for train_idx, test_idx in skf.split(X, y):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        clf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=random_state)
        clf.fit(X_tr, y_tr)
        preds = clf.predict(X_te)
        probs = clf.predict_proba(X_te)[:, 1]
        accs.append(accuracy_score(y_te, preds))
        aucs.append(roc_auc_score(y_te, probs))
    return float(np.mean(accs)), float(np.mean(aucs))
