"""
2차_139장 (구 프롬프트) vs 4차_139장 (엄격 프롬프트) LLM 결과 비교.
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

# --- GT ---
gt = pd.read_csv(PROJECT_ROOT / "data" / "image_eval_human_gt.csv")
gt = gt[gt["image_generated"].astype(str).str.upper() == "Y"].copy()
gt["level"] = gt["Lv.평가"].apply(parse_level_str)
gt["_id"] = gt["session_id"].astype(str).str.strip()

# --- v2 = 2차_139장 (구 프롬프트) ---
v2_path = PROJECT_ROOT / "results" / "llm" / "2차_139장" / "llm_results.csv"
# --- v3 = 4차_139장 (엄격 프롬프트) ---
v3_path = PROJECT_ROOT / "results" / "llm" / "4차_139장" / "llm_results.csv"

print(f"구 프롬프트 : {v2_path}")
print(f"엄격 프롬프트: {v3_path}")
print()

v2 = pd.read_csv(v2_path)
v3 = pd.read_csv(v3_path)

v2["_id"] = v2["image_id"].astype(str).str.strip()
v3["_id"] = v3["image_id"].astype(str).str.strip()

common = set(v2["_id"]) & set(v3["_id"]) & set(gt["_id"])
print(f"공통 image_id: {len(common)}장\n")

v2 = v2[v2["_id"].isin(common)].set_index("_id")
v3 = v3[v3["_id"].isin(common)].set_index("_id")
gt_sub = gt[gt["_id"].isin(common)].set_index("_id")

gt_lv = gt_sub["level"].apply(level_to_int)
lv_col = "final_level"
v2_lv = v2[lv_col].apply(level_to_int)
v3_lv = v3[lv_col].apply(level_to_int)

idx = gt_lv.index.intersection(v2_lv.index).intersection(v3_lv.index)
gt_lv = gt_lv.loc[idx]
v2_lv = v2_lv.loc[idx]
v3_lv = v3_lv.loc[idx]

n = len(idx)
print(f"비교 이미지: {n}장\n")

def dist_str(s):
    vc = s.value_counts().sort_index()
    return "  ".join(f"L{int(k)}:{v}({v/n*100:.0f}%)" for k, v in vc.items())

def metrics(pred, gt_vals, label):
    diff = pred - gt_vals
    valid = diff.dropna()
    nv = len(valid)
    exact   = (valid == 0).sum()
    within1 = (valid.abs() <= 1).sum()
    bias    = valid.mean()
    print(f"[{label}]")
    print(f"  exact_match  : {exact}/{nv} ({exact/nv*100:.1f}%)")
    print(f"  within_1     : {within1}/{nv} ({within1/nv*100:.1f}%)")
    print(f"  bias (pred-GT): {bias:+.3f}")
    print(f"  레벨 분포    : {dist_str(pred)}")
    return {"label": label, "exact": int(exact), "within1": int(within1), "bias": round(float(bias), 3), "n": nv}

print("=" * 60)
print(f"GT 레벨 분포: {dist_str(gt_lv)}")
print("=" * 60)
r2 = metrics(v2_lv, gt_lv, "구 프롬프트  (2차_139장)")
print()
r3 = metrics(v3_lv, gt_lv, "엄격 프롬프트 (4차_139장)")
print()

d_exact = r3["exact"] - r2["exact"]
d_bias  = r3["bias"]  - r2["bias"]
print("=" * 60)
print("개선 요약:")
print(f"  exact_match : {r2['exact']} -> {r3['exact']}  ({d_exact:+d}장)")
print(f"  within_1    : {r2['within1']} -> {r3['within1']}  ({r3['within1']-r2['within1']:+d}장)")
print(f"  bias        : {r2['bias']:+.3f} -> {r3['bias']:+.3f}  ({d_bias:+.3f})")
if abs(r3["bias"]) <= 0.3:
    print("  [OK] bias <= +-0.3 -- 엄격 프롬프트 채택 가능")
elif r3["bias"] > 0.3:
    print("  [!!] 여전히 과대평가 -- 프롬프트 추가 조정 필요")
else:
    print("  [!!] 과소평가 전환 -- 프롬프트가 너무 엄격함")
print("=" * 60)

# 과대/과소 분포
print("\n[오류 유형 분석]")
for label, pred in [("구 프롬프트", v2_lv), ("엄격 프롬프트", v3_lv)]:
    diff = pred - gt_lv
    over  = (diff > 0).sum()
    exact = (diff == 0).sum()
    under = (diff < 0).sum()
    severe_over  = (diff >= 2).sum()
    severe_under = (diff <= -2).sum()
    print(f"  {label}: 과대평가={over}({over/n*100:.0f}%)  정확={exact}({exact/n*100:.0f}%)  과소평가={under}({under/n*100:.0f}%)  "
          f"(심각과대(>=2)={severe_over}  심각과소(<=-2)={severe_under})")

# 총점 분포 변화
print("\n[total_score 분포]")
for label, df in [("구 프롬프트  (2차)", v2), ("엄격 프롬프트 (4차)", v3)]:
    if "total_score" in df.columns:
        ts = pd.to_numeric(df["total_score"], errors="coerce")
        mean = ts.mean()
        median = ts.median()
        p18 = (ts == 18).sum()
        print(f"  {label}: mean={mean:.2f}  median={median:.1f}  만점(18)={p18}({p18/n*100:.0f}%)")

# GT 총점 분포
gt_total = pd.to_numeric(gt_sub.loc[idx, "총점(만점 18점)"], errors="coerce")
print(f"  GT           : mean={gt_total.mean():.2f}  median={gt_total.median():.1f}")

# CSV 저장
out_path = PROJECT_ROOT / "results" / "llm" / "comparison_v2_v3_139.csv"
rows = []
for img_id in sorted(idx):
    rows.append({
        "image_id": img_id,
        "gt_level": gt_lv.get(img_id),
        "v2_level": v2_lv.get(img_id),
        "v3_level": v3_lv.get(img_id),
        "v2_diff": (v2_lv.get(img_id) - gt_lv.get(img_id))
                   if not pd.isna(v2_lv.get(img_id, np.nan)) else None,
        "v3_diff": (v3_lv.get(img_id) - gt_lv.get(img_id))
                   if not pd.isna(v3_lv.get(img_id, np.nan)) else None,
    })
pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")
print(f"\n저장 완료 -> {out_path}")
