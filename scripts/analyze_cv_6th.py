"""
6차_139장 CV 결과 정확도 분석.
5차 vs 6차 비교, 새 항목 변별력 분석.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from scipy import stats
from src.utils import PROJECT_ROOT, parse_level_str

LEVEL_INT = {"L0": 0, "L1": 1, "L2": 2, "L3": 3, "L4": 4, "L5": 5}

def lv2i(s):
    if pd.isna(s): return np.nan
    return LEVEL_INT.get(str(s).strip().upper(), np.nan)

def metrics(cv_lv, gt_lv, label):
    diff = cv_lv - gt_lv
    valid = diff.dropna()
    n = len(valid)
    exact   = (valid == 0).sum()
    w1      = (valid.abs() <= 1).sum()
    mae     = valid.abs().mean()
    bias    = valid.mean()
    print(f"  [{label}] n={n}  exact={exact}({exact/n*100:.1f}%)  "
          f"±1={w1}({w1/n*100:.1f}%)  MAE={mae:.3f}  bias={bias:+.3f}")
    return dict(label=label, n=n, exact=exact, exact_pct=round(exact/n*100,1),
                w1=w1, w1_pct=round(w1/n*100,1), mae=round(float(mae),3), bias=round(float(bias),3))

# ── 데이터 로드 ───────────────────────────────────────────────────────────────
gt_raw = pd.read_csv(PROJECT_ROOT / "data" / "image_eval_human_gt.csv")
gt = gt_raw[gt_raw["image_generated"].astype(str).str.upper() == "Y"].copy()
gt["level"]   = gt["Lv.평가"].apply(parse_level_str)
gt["gt_lv"]   = gt["level"].apply(lv2i)
gt["_id"]     = gt["session_id"].astype(str).str.strip()
gt["total_gt"]= pd.to_numeric(gt["총점(만점 18점)"], errors="coerce")
for col in ["피사체", "구도", "스타일", "사람", "프롬프트반영", "배경", "기타"]:
    gt[col] = pd.to_numeric(gt.get(col, np.nan), errors="coerce")

cv5 = pd.read_csv(PROJECT_ROOT / "results" / "cv" / "5차_139장" / "cv_results.csv")
cv6 = pd.read_csv(PROJECT_ROOT / "results" / "cv" / "6차_139장" / "cv_results.csv")

for df in (cv5, cv6):
    df["_id"]    = df["image_id"].astype(str).str.strip()
    df["cv_lv"]  = df["final_level"].apply(lv2i)

# 공통 ID
ids = set(gt["_id"]) & set(cv5["_id"]) & set(cv6["_id"])
gt5 = gt[gt["_id"].isin(ids)].set_index("_id")
c5  = cv5[cv5["_id"].isin(ids)].set_index("_id")
c6  = cv6[cv6["_id"].isin(ids)].set_index("_id")

idx = gt5.index.intersection(c5.index).intersection(c6.index)
gt5, c5, c6 = gt5.loc[idx], c5.loc[idx], c6.loc[idx]

# ── 1. 레벨 정확도 비교 ────────────────────────────────────────────────────────
print("=" * 70)
print("[1] GT 레벨 분포")
gd = gt5["gt_lv"].value_counts().sort_index()
print("  " + "  ".join(f"L{int(k)}:{v}" for k, v in gd.items()))

print()
print("[2] CV final_level 분포")
print("  5차:", dict(c5["final_level"].value_counts().sort_index()))
print("  6차:", dict(c6["final_level"].value_counts().sort_index()))

print()
print("[3] 정확도 메트릭 (GT 레벨 기준)")
r5 = metrics(c5["cv_lv"], gt5["gt_lv"], "5차_139장 (구 CV, 7항목)")
r6 = metrics(c6["cv_lv"], gt5["gt_lv"], "6차_139장 (신 CV, 11항목)")

print()
print("[비교 표]")
print(f"  {'지표':<12} {'5차':>8} {'6차':>8} {'변화':>8}")
print(f"  {'-'*40}")
print(f"  {'Exact':<12} {r5['exact_pct']:>7.1f}% {r6['exact_pct']:>7.1f}% {r6['exact_pct']-r5['exact_pct']:>+7.1f}%")
print(f"  {'±1 within':<12} {r5['w1_pct']:>7.1f}% {r6['w1_pct']:>7.1f}% {r6['w1_pct']-r5['w1_pct']:>+7.1f}%")
print(f"  {'MAE':<12} {r5['mae']:>8.3f} {r6['mae']:>8.3f} {r6['mae']-r5['mae']:>+8.3f}")
print(f"  {'bias':<12} {r5['bias']:>+8.3f} {r6['bias']:>+8.3f} {r6['bias']-r5['bias']:>+8.3f}")

# ── 2. total_score 비교 ────────────────────────────────────────────────────────
print()
print("=" * 70)
print("[4] total_score vs GT 총점 비교")
gt_ts = gt5["total_gt"]
c5_ts = pd.to_numeric(c5["total_score"], errors="coerce")
c6_ts = pd.to_numeric(c6["total_score"], errors="coerce")
print(f"  GT  총점 : mean={gt_ts.mean():.2f}  median={gt_ts.median():.1f}")
print(f"  5차 총점 : mean={c5_ts.mean():.2f}  median={c5_ts.median():.1f}  MAE={( c5_ts - gt_ts).abs().mean():.2f}")
print(f"  6차 총점 : mean={c6_ts.mean():.2f}  median={c6_ts.median():.1f}  MAE={(c6_ts - gt_ts).abs().mean():.2f}")

# ── 3. 새 항목 변별력 분석 ─────────────────────────────────────────────────────
print()
print("=" * 70)
print("[5] 새 CLIP 항목 변별력 분석")

def corr_analysis(cv_vals, gt_cat, item_name, cat_name, filter_none=True):
    """CV 항목 예측값과 GT 카테고리 점수의 상관계수."""
    cv_s = pd.to_numeric(cv_vals, errors="coerce")
    gt_s = pd.to_numeric(gt_cat, errors="coerce")
    if filter_none:
        mask = cv_s.notna() & gt_s.notna()
        cv_s, gt_s = cv_s[mask], gt_s[mask]
    n = len(cv_s)
    if n < 5:
        print(f"  {item_name} vs GT.{cat_name}: 데이터 부족 (n={n})")
        return
    uniq = cv_s.nunique()
    if uniq <= 1:
        print(f"  {item_name} vs GT.{cat_name}: n={n}  분산=0 (모든 예측이 {cv_s.iloc[0]}) → 상관계수 계산 불가")
        dist = dict(cv_s.value_counts())
        print(f"    예측 분포: {dist}  → 평균 기여 점수: +{float(cv_s.mean()):.1f}")
        return
    try:
        r_p, p_p = stats.pearsonr(cv_s, gt_s)
        r_s, p_s = stats.spearmanr(cv_s, gt_s)
    except Exception as e:
        print(f"  {item_name}: 계산 오류 {e}")
        return
    dist = dict(cv_s.value_counts().sort_index())
    print(f"  {item_name:<28} vs GT.{cat_name:<6} n={n:>3}  "
          f"Pearson r={r_p:+.3f}(p={p_p:.3f})  Spearman r={r_s:+.3f}(p={p_s:.3f})  "
          f"분포={dist}")

# 기존 항목
corr_analysis(c6["dog_visible"],    gt5["피사체"],  "dog_visible",            "피사체")
corr_analysis(c6["face_clear"],     gt5["피사체"],  "face_clear",             "피사체")
corr_analysis(c6["pastel_style"],   gt5["스타일"],  "pastel_style",           "스타일")
corr_analysis(c6["no_realistic_rendering"], gt5["스타일"], "no_realistic_rendering", "스타일")
corr_analysis(c6["place_context_ok"], gt5["배경"], "place_context_ok",       "배경")
corr_analysis(c6["pastel_color_ok"], gt5["스타일"], "pastel_color_ok",        "스타일")
corr_analysis(c6["sharpness_ok"],   gt5["피사체"],  "sharpness_ok",           "피사체")
corr_analysis(c6["no_multi_panel"], gt5["구도"],    "no_multi_panel",         "구도")

# ── 4. CLIP raw score 상관계수 ─────────────────────────────────────────────────
print()
print("[6] CLIP raw score vs GT 카테고리 상관계수 (0/1 예측값 대신 연속 score)")
for sc_col, gt_col in [
    ("clip_face_score",   "피사체"),
    ("clip_pastel_score", "스타일"),
    ("clip_place_score",  "배경"),
    ("clip_dog_score",    "피사체"),
]:
    if sc_col not in c6.columns:
        continue
    cv_s = pd.to_numeric(c6[sc_col], errors="coerce")
    gt_s = gt5[gt_col]
    mask = cv_s.notna() & gt_s.notna()
    cv_s, gt_s = cv_s[mask], gt_s[mask]
    n = len(cv_s)
    if n < 5:
        print(f"  {sc_col}: n={n} 부족")
        continue
    r_p, p_p = stats.pearsonr(cv_s, gt_s)
    r_s, p_s = stats.spearmanr(cv_s, gt_s)
    print(f"  {sc_col:<25} vs GT.{gt_col:<6} n={n:>3}  "
          f"Pearson={r_p:+.3f}(p={p_p:.3f})  Spearman={r_s:+.3f}(p={p_s:.3f})")

# ── 5. 분산 기여 분석 ─────────────────────────────────────────────────────────
print()
print("[7] 항목별 기여 분석: '모든 이미지 동일 점수' vs '실제 변별'")
print(f"  {'항목':<28} {'예측분포':<30} {'변별력'}")
print(f"  {'-'*75}")
for col in ["dog_visible","face_clear","one_dog","pastel_style","no_realistic_rendering",
            "brightness_ok","sharpness_ok","pastel_color_ok","no_multi_panel","no_text","place_context_ok"]:
    if col not in c6.columns:
        continue
    s = pd.to_numeric(c6[col], errors="coerce")
    n1   = (s == 1).sum()
    n0   = (s == 0).sum()
    nnan = s.isna().sum()
    total = len(s)
    if n1 == total or (n1 + nnan == total and n0 == 0):
        disc = "분산=0 (상수)"
    elif n0 == total or (n0 + nnan == total and n1 == 0):
        disc = "분산=0 (상수)"
    else:
        disc = f"변별 O ({n0}장=0, {n1}장=1)"
    dist_str = f"1:{n1} 0:{n0} N:{nnan}"
    print(f"  {col:<28} {dist_str:<30} {disc}")
