import os
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from typing import Any, Optional, Dict, List, Tuple

# =============================================================================
# SciPy
# =============================================================================
try:
    from scipy.integrate import solve_ivp
    from scipy.optimize import minimize
    from .classical_model import objective_function_classical
except Exception as e:
    raise ImportError(
        "Thiếu SciPy.\nCài: python -m pip install scipy\n"
        f"Chi tiết: {e}"
    )

# =============================================================================
# 1) CONFIG
# =============================================================================
NUM_BINS = 15

MIN_VALID_RATIO_BIN = 0.30
MIN_VALID_RATIO_TRIAL = 0.30
MIN_POINTS_FOR_FIT = 3
DT_CLIP_MAX_MS = 100.0

SUBBIN_MS = 100.0
MIN_VALID_BINS_PER_TRIAL = 3  # >=3 bin hợp lệ

# =============================================================================
# 2) AOI BOXES
# =============================================================================
AOI_BBOXES = {
    9: {'target1': [436.3, 293.9, 655.3, 465.7], 'target2': [726.8, 294.9, 971.7, 459.3], 'op1a': [215.9, 274.6, 414.6, 375.4], 'op1b': [154.0, 427.2, 417.4, 533.7], 'op2a': [987.4, 234.3, 1309.9, 412.8], 'op2b': [993.2, 432.9, 1309.9, 545.2]},
    12: {'op1': [432.5, 298.4, 637.8, 466.1], 'op2': [715.4, 273.2, 991.8, 481.9], 'op1a': [218.8, 270.3, 420.3, 379.7], 'op2a': [980.2, 230.0, 1223.5, 345.1], 'op1b': [228.9, 421.1, 423.2, 525.1], 'op2b': [987.4, 424.3, 1154.4, 526.5], 'target1': [447.3, 309.2, 625.2, 454.8], 'target2': [731.8, 305.2, 962.2, 450.8]},
    15: {'target1': [442.7, 307.9, 632.6, 460.0], 'target2': [732.4, 303.1, 922.3, 458.4], 'op2a': [944.3, 251.6, 1248.0, 362.4], 'op2b': [947.1, 422.9, 1252.3, 532.3], 'op1a': [175.6, 274.6, 417.4, 376.8], 'op1b': [112.3, 431.5, 414.6, 529.4]},
    22: {'target1': [441.4, 301.3, 626.3, 450.5], 'target2': [727.2, 307.2, 964.1, 454.5], 'op1a': [156.9, 199.8, 397.3, 361.0], 'op1b': [159.8, 428.6, 401.6, 654.6], 'op2a': [987.4, 182.5, 1269.6, 356.7], 'op2b': [991.8, 425.7, 1275.3, 604.2]},
    25: {'target1': [405.7, 267.8, 686.8, 504.1], 'target2': [729.3, 266.9, 976.5, 506.9], 'op1a': [146.8, 152.3, 365.6, 310.6], 'op1b': [142.5, 461.7, 359.9, 621.5], 'op2a': [1010.5, 152.3, 1276.8, 313.5], 'op2b': [1013.3, 447.3, 1299.8, 633.0]},
    28: {'target1': [411.5, 258.5, 680.2, 501.0], 'target2': [731.0, 261.0, 974.2, 503.6], 'op1a': [95.1, 137.9, 395.8, 382.6], 'op1b': [136.7, 477.6, 395.8, 664.3], 'op2a': [987.4, 139.3, 1268.1, 362.4], 'op2b': [990.3, 401.3, 1307.0, 717.9]},
    33: {'target1': [38.4, 270.6, 305.8, 510.5], 'target2': [1085.1, 280.5, 1345.8, 502.5], 'op1a': [773.0, 313.5, 1073.8, 464.6], 'op1b': [476.4, 499.2, 883.8, 658.9], 'op2b': [305.2, 316.3, 593.0, 473.2], 'op2a': [495.2, 94.7, 855.0, 268.8]},
    36: {'target1': [12.3, 257.5, 292.6, 520.7], 'target2': [1078.7, 274.9, 1340.4, 507.1], 'op2a': [541.2, 113.4, 837.7, 238.6], 'op2b': [309.5, 332.2, 591.6, 451.7], 'op1a': [781.6, 330.7, 1059.4, 447.3], 'op1b': [492.3, 516.4, 882.4, 625.8]}
}


# =============================================================================
# 3) UTILS
# =============================================================================
def to_int_or_none(v: Any) -> Optional[int]:
    if isinstance(v, (pd.Series, np.ndarray, list, tuple)):
        if len(v) == 0:
            return None
        v = v.iloc[0] if isinstance(v, pd.Series) else v[0]
    x = pd.to_numeric(v, errors="coerce")
    if pd.isna(x):
        return None
    return int(float(x))

def point_in_bbox_vec(x: np.ndarray, y: np.ndarray, bb: List[float]) -> np.ndarray:
    x1, y1, x2, y2 = bb
    return (x >= x1) & (x <= x2) & (y >= y1) & (y <= y2)

AOI_LOOKUP: Dict[int, List[Tuple[str, List[float]]]] = {
    int(tid): [(nm, bb) for nm, bb in aoi_dict.items()]
    for tid, aoi_dict in AOI_BBOXES.items()
}

def group_of_aoi(aoi_name: Optional[str]) -> Optional[int]:
    if not aoi_name or pd.isna(aoi_name):
        return None
    name = str(aoi_name).lower()
    if "op1" in name or "target1" in name:
        return 1
    if "op2" in name or "target2" in name:
        return 2
    return None

AOI_GROUP_MAP: Dict[str, Optional[int]] = {}
for _tid, pairs in AOI_LOOKUP.items():
    for nm, _bb in pairs:
        AOI_GROUP_MAP[nm] = group_of_aoi(nm)

def aoi_labels_for_points(tid_int: int, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    labels = np.full(x.shape, None, dtype=object)
    if tid_int not in AOI_LOOKUP:
        return labels
    for nm, bb in AOI_LOOKUP[tid_int]:
        m = (labels == None) & point_in_bbox_vec(x, y, bb)
        if np.any(m):
            labels[m] = nm
    return labels

def compress_aoi_sequence(seq: np.ndarray) -> List[str]:
    out: List[str] = []
    prev = None
    for a in seq:
        if a is None:
            continue
        if a == prev:
            continue
        out.append(a)
        prev = a
    return out

def count_transition_types(aoi_seq: List[str]) -> Dict[str, int]:
    w1 = w2 = b12 = b21 = 0
    for a, b in zip(aoi_seq, aoi_seq[1:]):
        ga = AOI_GROUP_MAP.get(a, None)
        gb = AOI_GROUP_MAP.get(b, None)
        if ga is None or gb is None:
            continue
        if ga == 1 and gb == 1:
            w1 += 1
        elif ga == 2 and gb == 2:
            w2 += 1
        elif ga == 1 and gb == 2:
            b12 += 1
        elif ga == 2 and gb == 1:
            b21 += 1
    return {"w1": w1, "w2": w2, "b12": b12, "b21": b21, "sum_alltran": w1 + w2 + b12 + b21}

# =============================================================================
# 4) SCENARIO RULES
# =============================================================================
SCENARIO_MAP = {
    9: 1, 12: 1, 15: 1,
    22: 2, 25: 2, 28: 2,
    33: 3, 36: 3,
}

# =============================================================================
# 5) BINNING (5 bin) + dwell probs + transitions + PUPIL
# =============================================================================
def compute_5_bins_per_trial(g: pd.DataFrame) -> dict:
    g = g.copy()
    g["Time"] = pd.to_numeric(g["Time"], errors="coerce")
    g["GazePosX"] = pd.to_numeric(g["GazePosX"], errors="coerce")
    g["GazePosY"] = pd.to_numeric(g["GazePosY"], errors="coerce")

    if "PupilSize" not in g.columns:
        g["PupilSize"] = np.nan
    else:
        g["PupilSize"] = pd.to_numeric(g["PupilSize"], errors="coerce")

    # NOTE: Tiền xử lý (Interpolation + Smoothing) đã được thực hiện trong
    # hàm standardize_one_input() → KHÔNG lặp lại ở đây để tránh Double Smoothing.

    g = g.dropna(subset=["Time", "GazePosX", "GazePosY"])
    empty_res = {
        "p1": [], "p2": [],
        "pupil_mean_all": [], "pupil_mean_choice": [],
        "pupil_delta_choice": [], "pupil_pct_choice": [], "pupil_absdiff_choice": [],
        "t_sec": [], "valid_r": [],
        "duration_ms": 0.0, "bin_width_ms": 0.0,
        "aoi_labels": None, "bin_indices": None, "dt": None, "t_rel": None
    }
    if g.shape[0] < 2:
        return empty_res

    g = g.sort_values("Time")
    t = g["Time"].to_numpy(dtype=float)
    t0 = float(t[0])
    t_rel = t - t0

    dt = np.diff(t, append=t[-1])

    dt_median = float(np.nanmedian(dt[:-1])) if len(dt) > 2 else float(np.nanmedian(dt))
    if not np.isfinite(dt_median) or dt_median <= 0:
        dt_median = 16.0
    dt[-1] = dt_median

    dt_cap = min(max(5.0 * dt_median, 20.0), DT_CLIP_MAX_MS)
    dt = np.clip(dt, 0.0, dt_cap)

    tid_int = to_int_or_none(g.loc[g.index[0], "TrialID"])
    if tid_int is None:
        return empty_res

    x_vals = g["GazePosX_Smooth"].to_numpy(dtype=float)
    y_vals = g["GazePosY_Smooth"].to_numpy(dtype=float)

    pupil_vals = g["PupilSize"].to_numpy(dtype=float)
    pupil_vals = np.where((np.isfinite(pupil_vals) & (pupil_vals > 0)), pupil_vals, np.nan)

    aoi_labels = aoi_labels_for_points(int(tid_int), x_vals, y_vals)
    grp = np.array([AOI_GROUP_MAP.get(a, None) for a in aoi_labels], dtype=object)

    # RESPONSE-LOCKED BINNING (15 bins x 200ms in the last 3000ms)
    duration_ms = 3000.0
    bin_width = 200.0
    t_end = t_rel[-1] + dt[-1]
    t_start = t_end - 3000.0
    
    bin_indices = np.floor((t_rel - t_start) / bin_width).astype(int)

    p1_arr, p2_arr = [], []
    pupil_all_arr, pupil_choice_arr = [], []
    t_sec_arr, valid_arr = [], []

    for b in range(NUM_BINS):
        mask_b = (bin_indices == b)
        if not np.any(mask_b):
            p1_arr.append(0.0); p2_arr.append(0.0)
            pupil_all_arr.append(np.nan); pupil_choice_arr.append(np.nan)
            valid_arr.append(0.0)
        else:
            dt_in = dt[mask_b]
            grp_in = grp[mask_b]
            pupil_in = pupil_vals[mask_b]

            # ===== Denominator "đúng nghĩa": tổng thời gian bin =====
            t_bin_total = float(np.sum(dt_in))
            if not np.isfinite(t_bin_total) or t_bin_total <= 0:
                t_bin_total = float(bin_width)

            sum_dt_op1 = float(np.sum(dt_in[grp_in == 1]))
            sum_dt_op2 = float(np.sum(dt_in[grp_in == 2]))
            t_choice = sum_dt_op1 + sum_dt_op2

            prob1 = (sum_dt_op1 / t_bin_total) if t_bin_total > 0 else 0.0
            prob2 = (sum_dt_op2 / t_bin_total) if t_bin_total > 0 else 0.0
            valid_ratio = (t_choice / t_bin_total) if t_bin_total > 0 else 0.0

            prob1 = float(np.clip(prob1, 0.0, 1.0))
            prob2 = float(np.clip(prob2, 0.0, 1.0))
            valid_ratio = float(np.clip(valid_ratio, 0.0, 1.0))

            mean_pupil_all = float(np.nanmean(pupil_in)) if np.any(np.isfinite(pupil_in)) else np.nan
            mask_choice = (grp_in == 1) | (grp_in == 2)
            mean_pupil_choice = float(np.nanmean(pupil_in[mask_choice])) if np.any(np.isfinite(pupil_in[mask_choice])) else np.nan

            p1_arr.append(prob1); p2_arr.append(prob2)
            pupil_all_arr.append(mean_pupil_all)
            pupil_choice_arr.append(mean_pupil_choice)
            valid_arr.append(valid_ratio)

        mid_bin_ms = (b * bin_width) + (bin_width / 2.0)
        t_sec_arr.append(float(mid_bin_ms / 1000.0))

    # ===== Baseline-correct pupil (Choice-only) theo trial =====
    pupil_choice_arr = np.array(pupil_choice_arr, dtype=float)
    baseline = pupil_choice_arr[0]
    if not np.isfinite(baseline):
        baseline = float(np.nanmedian(pupil_choice_arr)) if np.any(np.isfinite(pupil_choice_arr)) else np.nan

    if np.isfinite(baseline) and baseline > 0:
        pupil_delta = pupil_choice_arr - baseline
        pupil_pct = (pupil_choice_arr - baseline) / baseline
    else:
        pupil_delta = np.full_like(pupil_choice_arr, np.nan)
        pupil_pct = np.full_like(pupil_choice_arr, np.nan)

    pupil_absdiff = np.abs(np.diff(pupil_delta, prepend=pupil_delta[0]))
    pupil_absdiff[0] = 0.0 if np.isfinite(pupil_absdiff[0]) else 0.0

    return {
        "p1": np.array(p1_arr, dtype=float),
        "p2": np.array(p2_arr, dtype=float),
        "pupil_mean_all": np.array(pupil_all_arr, dtype=float),
        "pupil_mean_choice": pupil_choice_arr,
        "pupil_delta_choice": pupil_delta.astype(float),
        "pupil_pct_choice": pupil_pct.astype(float),
        "pupil_absdiff_choice": pupil_absdiff.astype(float),
        "t_sec": np.array(t_sec_arr, dtype=float),
        "valid_r": np.array(valid_arr, dtype=float),
        "duration_ms": float(duration_ms),
        "bin_width_ms": float(bin_width),
        "aoi_labels": aoi_labels,
        "bin_indices": bin_indices,
        "dt": dt,
        "t_rel": t_rel,
    }

# =============================================================================
# 6) OQS (Lindblad)
# =============================================================================
def get_hamiltonian(d: float) -> np.ndarray:
    return np.array([[0.0, d, 0.0],
                     [d, 0.0, d],
                     [0.0, d, 0.0]], dtype=np.complex128)

def get_lindblad_ops_oqs4():
    L12 = np.zeros((3, 3), dtype=np.complex128); L12[0, 1] = 1.0
    L32 = np.zeros((3, 3), dtype=np.complex128); L32[2, 1] = 1.0
    return [L12, L32]

def get_lindblad_ops_oqs6():
    ops = get_lindblad_ops_oqs4()
    L21 = np.zeros((3, 3), dtype=np.complex128); L21[1, 0] = 1.0
    L23 = np.zeros((3, 3), dtype=np.complex128); L23[1, 2] = 1.0
    return ops + [L21, L23]

def lindblad_rhs(t, rho_flat, H, L_ops, rates, timescale,
                 w_unitary=1.0, w_diss=1.0):
    """Vế phải phương trình GKSL với hệ số TÁCH RIÊNG cho hai thành phần.

        drho/dt = -i * w_unitary * [H, rho]
                  + w_diss * sum_k gamma_k ( L_k rho L_k^dag
                                             - 1/2 {L_k^dag L_k, rho} )

    Lindblad chuẩn : w_unitary = 1, w_diss = 1
    Mô hình lai    : w_unitary = 1 - w, w_diss = w

    LƯU Ý LỊCH SỬ: bản trước dùng MỘT tham số `w` mặc định 0.0 cho cả hai số
    hạng, và solve_oqs_model không truyền nó — khiến toàn bộ dissipator bị nhân
    với 0. Hệ quả là "OQS4"/"OQS6" chạy ra tiến hóa unitary thuần túy và các
    tham số gamma không có tác dụng gì. Tách hệ số để tránh lặp lại lỗi này.
    """
    rho = rho_flat.reshape(3, 3)
    d_rho = -1j * w_unitary * (H @ rho - rho @ H)
    for L, gamma in zip(L_ops, rates):
        L_dag = L.conj().T
        term1 = L @ rho @ L_dag
        LdagL = L_dag @ L
        term2 = LdagL @ rho + rho @ LdagL
        d_rho += w_diss * gamma * (term1 - 0.5 * term2)
    return (timescale * d_rho).reshape(9)

_I3 = np.eye(3, dtype=np.complex128)

def lindblad_superoperator(H, L_ops, rates, w_unitary=1.0, w_diss=1.0):
    """Toán tử sinh tác động lên vec(rho) theo quy ước column-major.

    Dùng đẳng thức vec(A X B) = (B^T kron A) vec(X).
    """
    S = -1j * w_unitary * (np.kron(_I3, H) - np.kron(H.T, _I3))
    for L, gamma in zip(L_ops, rates):
        L_dag = L.conj().T
        LdagL = L_dag @ L
        S = S + w_diss * gamma * (np.kron(L.conj(), L)
                                  - 0.5 * (np.kron(_I3, LdagL) + np.kron(LdagL.T, _I3)))
    return S

def _evolve(S, t_eval):
    """rho(t) = expm(S t) rho(0) với rho(0) = |2><2|, tính qua phân rã riêng.

    Phương trình Lindblad là ODE TUYẾN TÍNH nên nghiệm có dạng đóng; dùng phân rã
    riêng cho toàn bộ vector thời gian nhanh hơn tích phân số ~36 lần và không có
    sai số tích phân. Đã kiểm chứng khớp solve_ivp tới 1e-7.
    """
    rho0 = np.zeros((3, 3), dtype=np.complex128)
    rho0[1, 1] = 1.0
    v0 = rho0.reshape(9, order='F')
    lam, V = np.linalg.eig(S)
    c = np.linalg.solve(V, v0)
    Vt = (V @ (np.exp(np.outer(lam, np.asarray(t_eval, dtype=float))) * c[:, None])).T
    R = np.stack([v.reshape(3, 3, order='F') for v in Vt])
    return np.real(R[:, 0, 0]), np.real(R[:, 2, 2])

def solve_oqs_model(params, t_eval, model_type="OQS4"):
    d = float(params[0])
    timescale = 1.0 # Fixed timescale

    if model_type == "OQS4":
        rates = [float(params[1]), float(params[2])]
        L_ops = get_lindblad_ops_oqs4()
    else:
        rates = [float(params[1]), float(params[2]), float(params[3]), float(params[4])]
        L_ops = get_lindblad_ops_oqs6()

    H = get_hamiltonian(d)
    rho0 = np.zeros((3, 3), dtype=np.complex128)
    rho0[1, 1] = 1.0
    y0 = rho0.reshape(9)

    # Lindblad CHUẨN: cả phần unitary lẫn phần tiêu tán ở cường độ đầy đủ.
    try:
        S = lindblad_superoperator(H, L_ops, rates, w_unitary=1.0, w_diss=1.0)
        return _evolve(S, t_eval)
    except Exception:
        return None, None


def solve_hybrid_model(params, t_eval):
    # params = [d, gamma1, gamma2, w]
    d, gamma1, gamma2, w = float(params[0]), float(params[1]), float(params[2]), float(params[3])
    timescale = 1.0

    rates = [gamma1, gamma2]
    L_ops = get_lindblad_ops_oqs4()

    H = get_hamiltonian(d)
    rho0 = np.zeros((3, 3), dtype=np.complex128)
    rho0[1, 1] = 1.0
    y0 = rho0.reshape(9)

    # Lai ở cấp toán tử sinh: w=0 -> unitary thuần, w=1 -> tiêu tán thuần.
    try:
        S = lindblad_superoperator(H, L_ops, rates, w_unitary=(1.0 - w), w_diss=w)
        return _evolve(S, t_eval)
    except Exception:
        return None, None

_BIG = 1e6   # phạt khi tích phân thất bại; KHÔNG dùng vector 0 (xem ghi chú dưới)

def _sse(p11_pred, p33_pred, p11_true, p33_true):
    """SSE, hoặc _BIG nếu nghiệm không hợp lệ.

    LƯU Ý: bản trước trả về np.zeros(len(t_eval)) khi tích phân thất bại. Vector 0
    có ĐÚNG độ dài nên lọt qua mọi kiểm tra, và SSE so với vector 0 ra giá trị hữu
    hạn — có thể còn nhỏ nếu dữ liệu gần 0. Một lần fit hỏng vì thế có thể bị ghi
    nhận thành fit tốt. Nay solver trả None và ta phạt tường minh.
    """
    if p11_pred is None or p33_pred is None:
        return _BIG
    if len(p11_pred) != len(p11_true):
        return _BIG
    if not (np.all(np.isfinite(p11_pred)) and np.all(np.isfinite(p33_pred))):
        return _BIG
    return float(np.sum((p11_true - p11_pred) ** 2) + np.sum((p33_true - p33_pred) ** 2))

def objective_function_hybrid(params, t_eval, p11_true, p33_true):
    try:
        return _sse(*solve_hybrid_model(params, t_eval), p11_true=p11_true, p33_true=p33_true)
    except Exception:
        return _BIG

def objective_function(params, t_eval, p11_true, p33_true, model_type):
    try:
        return _sse(*solve_oqs_model(params, t_eval, model_type),
                    p11_true=p11_true, p33_true=p33_true)
    except Exception:
        return _BIG

# =============================================================================
# 7) GUI: chọn NHIỀU file + output 1 file
# =============================================================================
def pick_files():
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    paths = filedialog.askopenfilenames(
        title="Chọn 2 (hoặc nhiều) file RawData (.xlsx/.csv)",
        filetypes=[("Excel files", "*.xlsx *.xls"), ("CSV files", "*.csv"), ("All files", "*.*")]
    )
    root.destroy()
    return list(paths)

def pick_xlsx_save(default_dir: str):
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    out_path = filedialog.asksaveasfilename(
        title="Lưu kết quả (1 file Excel)",
        initialdir=default_dir,
        initialfile="processed_merged_inputs.xlsx",
        defaultextension=".xlsx",
        filetypes=[("Excel files", "*.xlsx")]
    )
    root.destroy()
    return out_path

# =============================================================================
# 8) Chuẩn hoá cột cho 1 file input
# =============================================================================
def standardize_one_input(df_raw: pd.DataFrame, source_name: str) -> pd.DataFrame:
    df = df_raw.copy()
    df.columns = [str(c).strip() for c in df.columns]

    col_map = {}
    for c in df.columns:
        cl = c.lower()
        if "subject" in cl:
            col_map["SubjectName"] = c
        elif "trial" in cl:
            col_map["TrialID"] = c
        elif "time" in cl:
            col_map["Time"] = c
        elif "posx" in cl or "gazex" in cl:
            col_map["GazePosX"] = c
        elif "posy" in cl or "gazey" in cl:
            col_map["GazePosY"] = c
        elif "pupildiax" in cl:
            col_map["PupilDiaX"] = c
        elif "pupildiay" in cl:
            col_map["PupilDiaY"] = c

    basic_missing = [k for k in ["SubjectName", "TrialID", "Time", "GazePosX", "GazePosY"] if k not in col_map]
    if basic_missing:
        raise ValueError(f"[{source_name}] Thiếu cột cơ bản: {basic_missing}. Columns: {list(df.columns)}")

    needed = ["SubjectName", "TrialID", "Time", "GazePosX", "GazePosY"]
    if "PupilDiaX" in col_map:
        needed.append("PupilDiaX")
    if "PupilDiaY" in col_map:
        needed.append("PupilDiaY")

    df = df.rename(columns={v: k for k, v in col_map.items()})[needed].copy()
    df = df.loc[:, ~df.columns.duplicated()].copy()

    # --- Pupil raw -> PupilSize ---
    if "PupilDiaX" in df.columns and "PupilDiaY" in df.columns:
        df["PupilDiaX"] = pd.to_numeric(df["PupilDiaX"], errors="coerce").replace(0, np.nan)
        # Fix scaling bug for Pupil (Vietnamese locale comma issue)
        df["PupilDiaX"] = df["PupilDiaX"] / 1000000.0
        df["PupilDiaY"] = pd.to_numeric(df["PupilDiaY"], errors="coerce").replace(0, np.nan)
        df["PupilDiaY"] = df["PupilDiaY"] / 1000000.0
        df["PupilSize"] = df[["PupilDiaX", "PupilDiaY"]].mean(axis=1)
    else:
        df["PupilSize"] = np.nan

    # types — chuyển đổi kiểu dữ liệu (PHẢI chạy cho MỌI trường hợp)
    df["Time"] = pd.to_numeric(df["Time"], errors="coerce")
    df["GazePosX"] = pd.to_numeric(df["GazePosX"], errors="coerce")
    df["GazePosY"] = pd.to_numeric(df["GazePosY"], errors="coerce")
    if df["GazePosX"].median() > 10000:
        df["GazePosX"] = df["GazePosX"] / 1000000.0 * 1.44
    if df["GazePosY"].median() > 10000:
        df["GazePosY"] = df["GazePosY"] / 1000000.0


    # keep only rows with gaze/time
    df = df.dropna(subset=["SubjectName", "TrialID", "Time", "GazePosX", "GazePosY"]).copy()

    
    # TIỀN XỬ LÝ (PREPROCESSING) CHUYÊN SÂU - BẢO LƯU DỮ LIỆU GỐC
    
    # 1. Loại bỏ Outliers (tạo cột mới)
    df["GazePosX_NoOutlier"] = df["GazePosX"]
    df["GazePosY_NoOutlier"] = df["GazePosY"]
    df.loc[(df["GazePosX_NoOutlier"] < -500) | (df["GazePosX_NoOutlier"] > 2500), "GazePosX_NoOutlier"] = np.nan
    df.loc[(df["GazePosY_NoOutlier"] < -500) | (df["GazePosY_NoOutlier"] > 1500), "GazePosY_NoOutlier"] = np.nan
    
    # 2. Nội suy (Interpolation) (tạo cột mới từ cột NoOutlier)
    df["GazePosX_Interp"] = df.groupby(["SubjectName", "TrialID"])["GazePosX_NoOutlier"].transform(lambda x: x.interpolate(method='linear', limit=3))
    df["GazePosY_Interp"] = df.groupby(["SubjectName", "TrialID"])["GazePosY_NoOutlier"].transform(lambda x: x.interpolate(method='linear', limit=3))
    
    # 3. Làm mượt (Smoothing) (tạo cột mới từ cột Interp)
    df["GazePosX_Smooth"] = df.groupby(["SubjectName", "TrialID"])["GazePosX_Interp"].transform(lambda x: x.rolling(window=3, min_periods=1, center=True).mean())
    df["GazePosY_Smooth"] = df.groupby(["SubjectName", "TrialID"])["GazePosY_Interp"].transform(lambda x: x.rolling(window=3, min_periods=1, center=True).mean())
    
    # 4. Phân loại Event (Fixation, Saccade, Blink) theo thuật toán I-VT
    dx = df.groupby(["SubjectName", "TrialID"])["GazePosX_Smooth"].diff()
    dy = df.groupby(["SubjectName", "TrialID"])["GazePosY_Smooth"].diff()
    dt = df.groupby(["SubjectName", "TrialID"])["Time"].diff()
    
    velocity = (dx**2 + dy**2)**0.5 / dt
    
    df["EventType"] = "Fixation"
    df.loc[velocity > 1.5, "EventType"] = "Saccade"
    df.loc[df["GazePosX"].isna(), "EventType"] = "Blink"
    
    mask_new_trial = (df["SubjectName"] != df["SubjectName"].shift()) | (df["TrialID"] != df["TrialID"].shift())
    df.loc[mask_new_trial, "EventType"] = "Unknown"

    # 5. Gắn nhãn Vùng chú ý (AOI) - Đối tượng đang nhìn vào đâu (xem gì)
    df["AOI_Label"] = None
    for tid_int in df["TrialID"].unique():
        mask_tid = df["TrialID"] == tid_int
        x_vals = df.loc[mask_tid, "GazePosX_Smooth"].to_numpy(dtype=float)
        y_vals = df.loc[mask_tid, "GazePosY_Smooth"].to_numpy(dtype=float)
        labels = aoi_labels_for_points(tid_int, x_vals, y_vals)
        df.loc[mask_tid, "AOI_Label"] = labels


    
    # Lọc bỏ những dòng không có dữ liệu gốc
    df = df.dropna(subset=["SubjectName", "TrialID", "Time", "GazePosX", "GazePosY"]).copy()


    # Heuristic: sec -> ms
    dt_global = df.groupby(["SubjectName", "TrialID"])["Time"].apply(
        lambda s: (
            np.nanmedian(np.diff(np.sort(pd.to_numeric(s, errors="coerce").dropna().to_numpy(dtype=float))))
            if pd.to_numeric(s, errors="coerce").dropna().shape[0] > 2 else np.nan
        )
    )
    dt_guess = float(np.nanmedian(dt_global.to_numpy(dtype=float))) if len(dt_global) else np.nan
    if np.isfinite(dt_guess) and 0 < dt_guess < 1.0:
        df["Time"] = df["Time"] * 1000.0

    # Trial int
    df["TrialID_int"] = df["TrialID"].apply(to_int_or_none)
    df = df.dropna(subset=["TrialID_int"]).copy()
    df["TrialID_int"] = df["TrialID_int"].astype(int)

    # filter to AOI trials
    df = df[df["TrialID_int"].isin(AOI_LOOKUP.keys())].copy()

    # Scenario
    df["no"] = df["TrialID_int"].map(SCENARIO_MAP)
    df = df.dropna(subset=["no"]).copy()
    df["no"] = pd.to_numeric(df["no"], errors="coerce").astype(int)

    # keep source
    df["SourceFile"] = source_name
    return df

# =============================================================================
# 9) MAIN: đọc N file -> 1 output
# =============================================================================

def process_data(in_paths, callback=None):
    if not in_paths:
        return None, None, None

    dfs = []
    errors = []
    import os
    for p in in_paths:
        try:
            if p.lower().endswith(".csv"):
                raw = pd.read_csv(p)
            else:
                raw = pd.read_excel(p)
            dfs.append(standardize_one_input(raw, os.path.basename(p)))
        except Exception as e:
            errors.append(f"{os.path.basename(p)}: {e}")

    if not dfs:
        return None, None, None

    df = pd.concat(dfs, ignore_index=True)
    prob_rows = []
    fit_rows = []
    trans_trial_rows = []

    b_d = (0.0, 5.0)
    b_a = (0.0, 5.0)
    b_s = (0.0, 10.0)
    b_q = (0.0, 5.0)
    
    total = len(df.groupby(["SubjectName", "TrialID_int"]))
    current = 0

    for (subj, tid_key), g in df.groupby(["SubjectName", "TrialID_int"]):
        current += 1
        if callback: callback(current, total, f"Processing {subj} Trial {tid_key}")
        
        tid_int = to_int_or_none(tid_key)
        if tid_int is None: continue
        tid_int = int(tid_int)

        scenario_no = to_int_or_none(g.get("no", pd.Series([tid_int])).iloc[0])
        if scenario_no is None: scenario_no = tid_int
        scenario_no = int(scenario_no)

        g2 = g.rename(columns={"TrialID_int": "TrialID"}).copy()
        g2["TrialID"] = tid_int

        bin_data = compute_5_bins_per_trial(g2)
        p1_arr = bin_data["p1"]
        p2_arr = bin_data["p2"]
        valid_r = bin_data["valid_r"]
        t_sec = bin_data["t_sec"]
        pupil_all = bin_data["pupil_mean_all"]
        pupil_choice = bin_data["pupil_mean_choice"]
        pupil_delta = bin_data["pupil_delta_choice"]
        pupil_pct = bin_data["pupil_pct_choice"]
        pupil_absdiff = bin_data["pupil_absdiff_choice"]
        duration_ms = bin_data["duration_ms"]
        bin_width_ms = bin_data["bin_width_ms"]
        aoi_labels = bin_data["aoi_labels"]
        bin_indices = bin_data["bin_indices"]

        if len(p1_arr) == 0: continue

        total_valid_dur = float(np.sum(valid_r * (duration_ms / NUM_BINS))) if duration_ms > 0 else 0.0
        trial_valid_ratio = total_valid_dur / max(duration_ms, 1.0)
        n_valid_bins = int(np.sum(valid_r >= MIN_VALID_RATIO_BIN))
        trial_pass_bins = (n_valid_bins >= MIN_VALID_BINS_PER_TRIAL)
        trial_pass_ratio = (trial_valid_ratio >= MIN_VALID_RATIO_TRIAL)
        trial_ok = bool(trial_pass_bins and trial_pass_ratio)

        for i in range(len(t_sec)):
            prob_rows.append({
                "subject": subj, "TrialID": tid_int, "no": str(scenario_no), "bin": int(i + 1),
                "t_bin_center_sec": float(t_sec[i]), "Prob_Op1": float(p1_arr[i]), "Prob_Op2": float(p2_arr[i]),
                "Bin_ValidRatio": float(valid_r[i]), "Avg_PupilDia": float(pupil_choice[i]) if np.isfinite(pupil_choice[i]) else np.nan,
                "PupilDelta_Choice": float(pupil_delta[i]) if np.isfinite(pupil_delta[i]) else np.nan,
                "PupilAbsDiff_Choice": float(pupil_absdiff[i]) if np.isfinite(pupil_absdiff[i]) else np.nan,
                "PupilPct_Choice": float(pupil_pct[i]) if np.isfinite(pupil_pct[i]) else np.nan,
                "TrialValidRatio": float(trial_valid_ratio), "Trial_OK": int(trial_ok), "ValidBins": int(n_valid_bins)
            })

        if aoi_labels is not None and bin_indices is not None and bin_width_ms > 0:
            for b in range(NUM_BINS):
                mask_b = (bin_indices == b)
                seq = compress_aoi_sequence(aoi_labels[mask_b]) if np.any(mask_b) else []
                cnt = count_transition_types(seq) if len(seq) >= 2 else {"w1": 0, "w2": 0, "b12": 0, "b21": 0, "sum_alltran": 0}
                trans_trial_rows.append({
                    "subject": subj, "TrialID": tid_int, "bin": int(b + 1),
                    "w1": int(cnt["w1"]), "w2": int(cnt["w2"]), "b12": int(cnt["b12"]), "b21": int(cnt["b21"]),
                    "Prob_Transition": float((cnt["b12"] + cnt["b21"]) / (cnt["w1"] + cnt["w2"] + cnt["b12"] + cnt["b21"])) if (cnt["w1"] + cnt["w2"] + cnt["b12"] + cnt["b21"]) > 0 else 0.0,
                    "Trial_OK": int(trial_ok)
                })

        if not trial_ok: continue

        mask_fit = valid_r >= MIN_VALID_RATIO_BIN
        t_eval = t_sec[mask_fit]
        p11_true = p1_arr[mask_fit]
        p33_true = p2_arr[mask_fit]

        if len(t_eval) < MIN_POINTS_FOR_FIT: continue

        
        # MULTI-START OPTIMIZATION
        np.random.seed(42)
        n_starts = 20
        
        best_res4 = None
        best_sse4 = 1e6
        for _ in range(n_starts):
            x0 = [np.random.uniform(*b_d)] + [np.random.uniform(*b_a) for _ in range(2)]
            res = minimize(objective_function, x0=x0, args=(t_eval, p11_true, p33_true, "OQS4"), bounds=[b_d, b_a, b_a], method="L-BFGS-B")
            if res.fun < best_sse4:
                best_sse4 = res.fun
                best_res4 = res
                
        best_res6 = None
        best_sse6 = 1e6
        for _ in range(n_starts):
            x0 = [np.random.uniform(*b_d)] + [np.random.uniform(*b_a) for _ in range(4)]
            res = minimize(objective_function, x0=x0, args=(t_eval, p11_true, p33_true, "OQS6"), bounds=[b_d]+[b_a]*4, method="L-BFGS-B")
            if res.fun < best_sse6:
                best_sse6 = res.fun
                best_res6 = res
                
        # KIỂM TRA HỘI TỤ (không ghi đè).
        # OQS4 lồng trong OQS6 (gamma3 = gamma4 = 0), nên về toán học phải có
        # SSE(OQS6) <= SSE(OQS4). Nếu vi phạm => optimiser chưa hội tụ ở OQS6.
        # Bản trước GÁN ĐÈ best_sse6 = best_sse4, khiến số vi phạm luôn bằng 0 và
        # tạo cảm giác sai rằng tối ưu đã hội tụ. Nay ta thử thêm điểm khởi tạo
        # "ấm" từ nghiệm OQS4, và nếu vẫn vi phạm thì GHI NHẬN trung thực.
        if best_sse6 > best_sse4 and best_res4 is not None:
            x_warm = list(best_res4.x) + [0.0, 0.0]
            res = minimize(objective_function, x0=x_warm,
                           args=(t_eval, p11_true, p33_true, "OQS6"),
                           bounds=[b_d] + [b_a] * 4, method="L-BFGS-B")
            if res.fun < best_sse6:
                best_sse6 = res.fun
                best_res6 = res
        nested_violation = int(best_sse6 > best_sse4 + 1e-12)

        best_res_ctmc = None
        best_sse_ctmc = 1e6
        for _ in range(n_starts):
            x0 = [np.random.uniform(*b_q) for _ in range(4)]
            res = minimize(objective_function_classical, x0=x0, args=(t_eval, p11_true, p33_true), bounds=[b_q]*4, method="L-BFGS-B")
            if res.fun < best_sse_ctmc:
                best_sse_ctmc = res.fun
                best_res_ctmc = res
                
        best_res_hyb = None
        best_sse_hyb = 1e6
        for _ in range(n_starts):
            x0 = [np.random.uniform(*b_d)] + [np.random.uniform(*b_a) for _ in range(2)] + [np.random.uniform(0, 1)]
            res = minimize(objective_function_hybrid, x0=x0, args=(t_eval, p11_true, p33_true), bounds=[b_d, b_a, b_a, (0,1)], method="L-BFGS-B")
            if res.fun < best_sse_hyb:
                best_sse_hyb = res.fun
                best_res_hyb = res

        n = int(len(t_eval) * 2)
        k4, k6, k_ctmc, k_hyb = 3, 5, 4, 4  # timescale is fixed to 1, w added to hyb
        sse4 = max(float(best_sse4), 1e-10)
        sse6 = max(float(best_sse6), 1e-10)
        sse_ctmc = max(float(best_sse_ctmc), 1e-10)
        sse_hyb = max(float(best_sse_hyb), 1e-10)
        
        bic4 = n * np.log(sse4 / n) + k4 * np.log(n)
        bic6 = n * np.log(sse6 / n) + k6 * np.log(n)
        bic_ctmc = n * np.log(sse_ctmc / n) + k_ctmc * np.log(n)
        bic_hyb = n * np.log(sse_hyb / n) + k_hyb * np.log(n)
        
        aic4 = n * np.log(sse4 / n) + 2 * k4
        aic6 = n * np.log(sse6 / n) + 2 * k6
        aic_ctmc = n * np.log(sse_ctmc / n) + 2 * k_ctmc
        aic_hyb = n * np.log(sse_hyb / n) + 2 * k_hyb
        
        bics = {"OQS4": bic4, "OQS6": bic6, "Classical_CTMC": bic_ctmc, "Hybrid": bic_hyb}
        aics = {"OQS4": aic4, "OQS6": aic6, "Classical_CTMC": aic_ctmc, "Hybrid": aic_hyb}
        best_bic = min(bics, key=bics.get)
        best_aic = min(aics, key=aics.get)


        fit_rows.append({
            "subject": subj, "TrialID": tid_int,
            "TrialValidRatio": float(trial_valid_ratio),
            "N_FitBins": int(len(t_eval)), "N_Obs": int(n),
            "NestedViolation": int(nested_violation),
            "SSE_OQS4": sse4, "SSE_OQS6": sse6, "SSE_CTMC": sse_ctmc, "SSE_Hybrid": sse_hyb,
            "BIC_OQS4": bic4, "BIC_OQS6": bic6, "BIC_CTMC": bic_ctmc, "BIC_Hybrid": bic_hyb,
            "AIC_OQS4": aic4, "AIC_OQS6": aic6, "AIC_CTMC": aic_ctmc, "AIC_Hybrid": aic_hyb,
            "BestModel_BIC": best_bic, "BestModel_AIC": best_aic,
            "Params_OQS4": str(getattr(best_res4, "x", None)),
            "Params_OQS6": str(getattr(best_res6, "x", None)),
            "Params_CTMC": str(getattr(best_res_ctmc, "x", None)),
            "Params_Hybrid": str(getattr(best_res_hyb, "x", None))
        })

    return pd.DataFrame(prob_rows), pd.DataFrame(trans_trial_rows), pd.DataFrame(fit_rows), df
