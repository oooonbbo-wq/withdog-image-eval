"""
v1 (1차_10장) vs v2 (3차_10장) LLM 결과 비교.
GT와 비교하여 exact_match, within_1, bias 개선 여부 확인.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from src.utils import PROJECT_ROOT, parse_level_str

LEVEL_ORDER = {"L1": 1, "L2": 2, "L3": 3, "L4": 4, "L5": 5}

def level_to_int(s):
    if pd.isna(s):
        return np.nan
    return LEVEL_ORDER.get(str(s).strip().upper(), np.nan)

# --- GT 로드 ---
gt_path = PROJECT_ROOT / "data" / "image_eval_human_gt.csv"
gt = pd.read_csv(gt_path)
gt = gt[gt["image_generated"].astype(str).str.upper() == "Y"].copy()
gt["level"] = gt["Lv.평가"].apply(parse_level_str)
gt["total"] = pd.to_numeric(gt["총점(만점 18점)"], errors="coerce")

# --- LLM v1 (1차_10장) ---
v1_dir = PROJECT_ROOT / "results" / "llm"
# 1차_10장 폴더 찾기
v1_candidates = sorted(v1_dir.glob("1차_10장*/llm_results.csv"))
if not v1_candidates:
    v1_candidates = sorted(v1_dir.glob("1차_10*/llm_results.csv"))
if not v1_candidates:
    print("[!] v1 결과 파일을 찾을 수 없습니다. results/llm/ 폴더 목록:")
    for p in v1_dir.iterdir():
        print(f"    {p.name}")
    sys.exit(1)
v1_path = v1_candidates[0]

# --- LLM v2 (3차_10장) ---
v2_candidates = sorted(v1_dir.glob("3차_10장*/llm_results.csv"))
if not v2_candidates:
    v2_candidates = sorted(v1_dir.glob("3차_10*/llm_results.csv"))
if not v2_candidates:
    print("[!] v2 결과 파일을 찾을 수 없습니다.")
    sys.exit(1)
v2_path = v2_candidates[0]

print(f"v1: {v1_path}")
print(f"v2: {v2_path}")
print()

v1 = pd.read_csv(v1_path)
v2 = pd.read_csv(v2_path)

def get_image_ids(df):
    for col in ("image_id", "session_id"):
        if col in df.columns:
            return df[col].astype(str).str.strip()
    return pd.Series([""] * len(df))

v1["_id"] = get_image_ids(v1)
v2["_id"] = get_image_ids(v2)
gt["_id"] = gt["session_id"].astype(str).str.strip() if "session_id" in gt.columns else (
    gt["image_id"].astype(str).str.strip() if "image_id" in gt.columns else gt.index.astype(str)
)

# v1, v2에 공통으로 있는 image_id 로 맞추기
common_ids = list(set(v1["_id"]) & set(v2["_id"]))
print(f"공통 image_id: {len(common_ids)}장")

v1 = v1[v1["_id"].isin(common_ids)].set_index("_id")
v2 = v2[v2["_id"].isin(common_ids)].set_index("_id")
gt_sub = gt[gt["_id"].isin(common_ids)].set_index("_id")

# GT 레벨이 없으면 total_score에서 추정
if "level" not in gt_sub.columns:
    gt_sub["level"] = gt_sub["total"].apply(lambda x: parse_level_str(str(int(x)) if not pd.isna(x) else ""))

# 레벨 숫자 변환
gt_lv  = gt_sub["level"].apply(level_to_int)
v1_lv  = v1["final_level"].apply(level_to_int) if "final_level" in v1.columns else v1["raw_level"].apply(level_to_int)
v2_lv  = v2["final_level"].apply(level_to_int) if "final_level" in v2.columns else v2["raw_level"].apply(level_to_int)

# 공통 index 맞추기
idx = gt_lv.index.intersection(v1_lv.index).intersection(v2_lv.index)
gt_lv = gt_lv.loc[idx]
v1_lv = v1_lv.loc[idx]
v2_lv = v2_lv.loc[idx]

n = len(idx)
print(f"비교 가능 이미지: {n}장\n")

def metrics(pred, gt_vals, label):
    diff = pred - gt_vals
    exact = (diff == 0).sum()
    within1 = (diff.abs() <= 1).sum()
    bias = diff.mean()
    print(f"[{label}]")
    print(f"  exact_match : {exact}/{n} ({exact/n*100:.1f}%)")
    print(f"  within_1    : {within1}/{n} ({within1/n*100:.1f}%)")
    print(f"  bias(pred-GT): {bias:+.3f}  (양수=과대평가, 음수=과소평가)")
    print(f"  score 분포  : {dict(pred.value_counts().sort_index())}")
    return {"label": label, "exact": exact, "within1": within1, "bias": round(float(bias), 3), "n": n}

print("=" * 55)
gt_dist = dict(gt_lv.value_counts().sort_index())
print(f"GT 레벨 분포     : {gt_dist}")
print("=" * 55)
r1 = metrics(v1_lv, gt_lv, "v1 (구 프롬프트, 1차_10장)")
print()
r2 = metrics(v2_lv, gt_lv, "v2 (엄격 프롬프트, 3차_10장)")
print()

# 개선량
delta_exact = r2["exact"] - r1["exact"]
delta_bias  = r2["bias"]  - r1["bias"]
print("=" * 55)
print(f"개선 요약:")
print(f"  exact_match 변화: {r1['exact']} → {r2['exact']}  ({delta_exact:+d}장)")
print(f"  bias 변화       : {r1['bias']:+.3f} → {r2['bias']:+.3f}  ({delta_bias:+.3f})")
if abs(r2["bias"]) <= 0.3:
    print("  [OK] bias <= +-0.3 -> 139장 본실험 진행 가능")
else:
    print("  [!!] bias > +-0.3 -> 프롬프트 추가 조정 필요")
print("=" * 55)

# 개별 예측 상세
print("\n[개별 예측]")
print(f"{'image_id':<40}  {'GT':>4}  {'v1':>4}  {'v2':>4}  {'d1':>4}  {'d2':>4}")
print("-" * 65)
for img_id in sorted(idx):
    gt_v  = gt_lv.get(img_id, np.nan)
    v1_v  = v1_lv.get(img_id, np.nan)
    v2_v  = v2_lv.get(img_id, np.nan)
    d1 = v1_v - gt_v if not pd.isna(v1_v) and not pd.isna(gt_v) else np.nan
    d2 = v2_v - gt_v if not pd.isna(v2_v) and not pd.isna(gt_v) else np.nan
    flag = " <-- improved" if (not pd.isna(d1) and not pd.isna(d2) and abs(d2) < abs(d1)) else ""
    print(f"  {str(img_id):<38}  L{int(gt_v) if not pd.isna(gt_v) else '?':>1}  L{int(v1_v) if not pd.isna(v1_v) else '?':>1}  L{int(v2_v) if not pd.isna(v2_v) else '?':>1}  {d1:+.0f}  {d2:+.0f}{flag}")

# CSV 저장
out_path = PROJECT_ROOT / "results" / "llm" / "comparison_v1_vs_v2.csv"
rows = []
for img_id in sorted(idx):
    rows.append({
        "image_id": img_id,
        "gt_level": gt_lv.get(img_id),
        "v1_level": v1_lv.get(img_id),
        "v2_level": v2_lv.get(img_id),
        "v1_diff": v1_lv.get(img_id) - gt_lv.get(img_id) if not pd.isna(v1_lv.get(img_id, np.nan)) else None,
        "v2_diff": v2_lv.get(img_id) - gt_lv.get(img_id) if not pd.isna(v2_lv.get(img_id, np.nan)) else None,
    })
pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")
print(f"\n저장 완료 -> {out_path}")
