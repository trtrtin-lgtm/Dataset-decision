import os, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import codecs

ROOT = r"E:\ET\ĐA Đánh giá sự lưỡng lự bằng mô hình lượng tử OQS4 và OQS6-20260828T140941Z-1-001"
RUN = os.path.join(ROOT, r"results\OQS_Run_20260923_213629_FIXED\Data_Excel\OQS_Results_Auto.xlsx")

fit = pd.read_excel(RUN, sheet_name='OQS_Fit_Results')
pr  = pd.read_excel(RUN, sheet_name='Prob_5bins')
pr  = pr[pr.Trial_OK == 1]

N_TRIAL = int(len(fit))
bic = fit.BestModel_BIC.value_counts().to_dict()

import re
def parse(s):
    return np.array([float(x) for x in re.findall(r'-?\d+\.?\d*(?:[eE][-+]?\d+)?', str(s))])
W = np.vstack([parse(s) for s in fit.Params_Hybrid])[:, 3]

# survey
s1 = pd.read_excel(os.path.join(ROOT, 'Khảo sát hành vi ra quyết định của sinh viên từ 18 đến 22 tuổi (1) (Câu trả lời).xlsx'))
s2 = pd.read_excel(os.path.join(ROOT, 'Khảo sát hành vi ra quyết định của sinh viên từ 18 đến 22 tuổi (2) (Câu trả lời).xlsx'))
QH = 'Bạn thấy Huy là người như thế nào?'
QK = 'Kiểm tra toàn bộ có ảnh hưởng đến bạn không?'
def qq(df):
    hu = df[QH].apply(lambda x: 'T' if 'thông' in str(x).lower() else 'X')
    kt = df[QK].apply(lambda x: 'C' if str(x).startswith('Có') else 'K')
    yy = int(((hu == 'T') & (kt == 'C')).sum())
    nn = int(((hu == 'X') & (kt == 'K')).sum())
    return yy, nn, (yy + nn) / len(df)
y1, n1_, qq1 = qq(s1)
y2, n2_, qq2 = qq(s2)

FIGDIR = os.path.join(ROOT, 'Figures', 'paper')
os.makedirs(FIGDIR, exist_ok=True)
plt.rcParams.update({'font.size': 11, 'font.family': 'DejaVu Sans',
                     'axes.spines.top': False, 'axes.spines.right': False})
C1, C2, CG = '#1f4e79', '#c0504d', '#7f7f7f'

def make_figures(lang='VI'):
    suffix = '_VI' if lang == 'VI' else '_EN'
    
    # --- Fig 1: Cascade ---
    gg = pr.groupby('bin')[['Prob_Op1', 'Prob_Op2']].agg(['mean', 'sem'])
    tt = (gg.index.values - 1) * 0.2 - 3.0 + 0.2
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    
    lab1 = 'Phương án 1' if lang == 'VI' else 'Option 1'
    lab2 = 'Phương án 2' if lang == 'VI' else 'Option 2'
    
    for col, c, lab in (('Prob_Op1', C1, lab1), ('Prob_Op2', C2, lab2)):
        m = gg[(col, 'mean')].values; s = gg[(col, 'sem')].values
        ax.plot(tt, m, '-o', color=c, ms=4, lw=2, label=lab)
        ax.fill_between(tt, m - s, m + s, color=c, alpha=.18, lw=0)
    ax.axvline(0, color='k', ls='--', lw=1)
    
    annot = 'thời điểm\nphản hồi' if lang == 'VI' else 'response\nonset'
    ax.annotate(annot, xy=(0, ax.get_ylim()[0]), xytext=(-0.12, 0.06),
                textcoords=('data', 'axes fraction'), ha='right', va='bottom',
                fontsize=9, color=CG)
                
    xl = 'Thời gian tương đối so với phản hồi (s)' if lang == 'VI' else 'Relative time to response (s)'
    yl = 'Xác suất chú ý (tỷ lệ thời gian lưu trú)' if lang == 'VI' else 'Attention probability (dwell time ratio)'
    ax.set_xlabel(xl); ax.set_ylabel(yl)
    ax.legend(frameon=False, loc='upper left')
    fig.tight_layout(); fig.savefig(os.path.join(FIGDIR, f'Fig1_Cascade{suffix}.png'), dpi=300); plt.close(fig)

    # --- Fig 2: Model Comparison ---
    order = ['Classical_CTMC', 'Hybrid', 'OQS4', 'OQS6']; lbl = ['CTMC', 'Hybrid', 'OQS4', 'OQS6']
    cnt = [int(bic.get(k, 0)) for k in order]
    cols = [CG, '#4f81bd', C1, C2]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
    bars = axes[0].bar(lbl, cnt, color=cols)
    for r, v in zip(bars, cnt):
        axes[0].text(r.get_x() + r.get_width() / 2, v + 1, '%d\n(%.1f%%)' % (v, 100 * v / N_TRIAL),
                     ha='center', fontsize=9)
                     
    yl_a = 'Số lượt thử thắng theo BIC' if lang == 'VI' else 'Winning trials (BIC)'
    axes[0].set_ylabel(yl_a); axes[0].set_ylim(0, max(cnt) * 1.28)
    
    bp = axes[1].boxplot([fit.SSE_CTMC, fit.SSE_Hybrid, fit.SSE_OQS4, fit.SSE_OQS6],
                         tick_labels=lbl, showfliers=False, patch_artist=True,
                         medianprops=dict(color='k'))
    for p_, c in zip(bp['boxes'], cols): p_.set_facecolor(c); p_.set_alpha(.65)
    
    yl_b = 'Tổng bình phương phần dư (SSE)' if lang == 'VI' else 'Sum of Squared Errors (SSE)'
    axes[1].set_ylabel(yl_b)
    fig.tight_layout(); fig.savefig(os.path.join(FIGDIR, f'Fig2_ModelComparison{suffix}.png'), dpi=300); plt.close(fig)

    # --- Fig 3: Hybrid W ---
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.hist(W, bins=20, range=(0, 1), color='#4f81bd', edgecolor='white', alpha=.85)
    
    med_lbl = f'trung vị w = {np.median(W):.3f}' if lang == 'VI' else f'median w = {np.median(W):.3f}'
    ax.axvline(np.median(W), color=C2, ls='--', lw=2, label=med_lbl)
    
    xl_w = 'Trọng số lai w (0 = unitary thuần túy, 1 = tiêu tán thuần túy)' if lang == 'VI' else 'Hybrid weight w (0 = pure unitary, 1 = pure dissipative)'
    yl_w = 'Số lượt thử' if lang == 'VI' else 'Frequency (trials)'
    ax.set_xlabel(xl_w); ax.set_ylabel(yl_w)
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(FIGDIR, f'Fig3_HybridW{suffix}.png'), dpi=300); plt.close(fig)

    # --- Fig 4: QQ Survey ---
    pa = [y1 / len(s1), n1_ / len(s1)]; pb = [y2 / len(s2), n2_ / len(s2)]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    x = np.arange(2); wd = .35
    
    l1 = f'Nhóm 1 (n={len(s1)})' if lang == 'VI' else f'Group 1 (n={len(s1)})'
    l2 = f'Nhóm 2 (n={len(s2)})' if lang == 'VI' else f'Group 2 (n={len(s2)})'
    
    ax.bar(x - wd / 2, pa, wd, label=l1, color=C1, alpha=.85)
    ax.bar(x + wd / 2, pb, wd, label=l2, color=C2, alpha=.85)
    for i, (a, b) in enumerate(zip(pa, pb)):
        ax.text(i - wd / 2, a + .008, '%.3f' % a, ha='center', fontsize=9)
        ax.text(i + wd / 2, b + .008, '%.3f' % b, ha='center', fontsize=9)
        
    l_sum = 'tổng QQ' if lang == 'VI' else 'QQ sum'
    ax.plot([-.3, .3], [sum(pa)] * 2, color='k', ls=':', lw=1.6)
    ax.plot([.7, 1.3], [sum(pb)] * 2, color='k', ls=':', lw=1.6, label=l_sum)
    ax.set_xticks(x)
    
    xt = ['p(Đồng ý, Đồng ý)', 'p(Từ chối, Từ chối)'] if lang == 'VI' else ['p(Agree, Agree)', 'p(Reject, Reject)']
    ax.set_xticklabels(xt)
    
    yl_qq = 'Xác suất' if lang == 'VI' else 'Probability'
    ax.set_ylabel(yl_qq); ax.set_ylim(0, max(sum(pa), sum(pb)) * 1.35)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(FIGDIR, f'Fig4_QQ{suffix}.png'), dpi=300); plt.close(fig)

make_figures('VI')
make_figures('EN')
print("Plots generated successfully!")
