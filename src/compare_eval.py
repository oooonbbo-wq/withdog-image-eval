"""
src/compare_eval.py

image_eval_human_gt.csv vs llm_results.csv vs cv_results.csv 비교 → comparison_result.csv

출력 CSV 구조:
  row_type  : item | group | overall
  name      : 항목명 또는 그룹명
  group     : objective | semantic | subtle  (row_type=item일 때만)
  n         : 비교 가능한 샘플 수 (양쪽 모두 1/0)
  llm_acc   : 정확도
  llm_prec  : 정밀도
  llm_rec   : 재현율
  llm_f1    : F1
  cv_acc, cv_prec, cv_rec, cv_f1 : 동일
  delta_f1  : cv_f1 - llm_f1 (양수면 CV 우위)
  winner    : LLM / CV / TIE / NA

실행:
  python -m src.compare_eval \
      --gt   data/image_eval_human_gt.csv \
      --llm  results/llm_results.csv \
      --cv   results/cv_results.csv \
      --out  results/comparison_result.csv
"""
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*[Mm]ean of empty slice.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*All-NaN slice encountered.*")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.scoring import ITEM_KEYS, LEVEL_ORDER
from src.utils import resolve_path, ensure_dir, to_binary_or_none, parse_level_str, get_latest_run_path, PROJECT_ROOT


# 항목 그룹 (계획서 §9.1)
ITEM_GROUPS: Dict[str, List[str]] = {
    "objective": [
        "dog_visible", "one_dog", "brightness_ok", "sharpness_ok",
        "pastel_color_ok", "no_text", "no_multi_panel",
    ],
    "semantic": [
        "human_prompt_consistency", "prompt_reflected",
        "place_context_ok", "time_palette_ok", "pastel_style",
    ],
    "subtle": [
        "face_clear", "no_duplicate_face_parts", "no_extra_legs",
        "no_mirror", "human_face_clear", "no_realistic_rendering",
    ],
}
ITEM_TO_GROUP = {it: g for g, items in ITEM_GROUPS.items() for it in items}

CRITICAL_ITEMS = [
    "dog_visible", "one_dog", "no_multi_panel", "no_text",
    "face_clear", "no_extra_legs", "prompt_reflected",
]


def _normalize_gt(df: pd.DataFrame) -> pd.DataFrame:
    """
    GT CSV의 다양한 컬럼명을 compare_eval이 기대하는 형태로 정규화.

    - image_id 없으면 session_id 에서 생성
    - final_level 없으면 'Lv.평가' 컬럼에서 파싱 ('Lv5: 우수' → 'L5')
    - total_score 없으면 '총점(만점 18점)' 컬럼에서 rename
    """
    df = df.copy()

    # image_id 정규화
    if "image_id" not in df.columns:
        if "session_id" in df.columns:
            df["image_id"] = df["session_id"]
        else:
            df["image_id"] = df.index.astype(str)

    # final_level 정규화
    if "final_level" not in df.columns:
        level_col = next(
            (c for c in df.columns if "lv" in c.lower() or "level" in c.lower() or "평가" in c),
            None
        )
        if level_col:
            df["final_level"] = df[level_col].apply(parse_level_str)

    # total_score 정규화
    if "total_score" not in df.columns:
        score_col = next(
            (c for c in df.columns if "총점" in c or "total" in c.lower()),
            None
        )
        if score_col:
            df["total_score"] = pd.to_numeric(df[score_col], errors="coerce")

    return df


def _metrics_for_item(gt_col: pd.Series, pred_col: pd.Series) -> Dict:
    """null(둘 중 하나라도) 제외 후 accuracy/precision/recall/f1."""
    gt = gt_col.apply(to_binary_or_none)
    pr = pred_col.apply(to_binary_or_none)
    mask = gt.notna() & pr.notna()
    n = int(mask.sum())
    if n == 0:
        return {"acc": np.nan, "prec": np.nan, "rec": np.nan, "f1": np.nan, "n": 0}
    y_true = gt[mask].astype(int).values
    y_pred = pr[mask].astype(int).values
    acc = accuracy_score(y_true, y_pred)
    if len(set(y_true)) < 2:
        return {"acc": acc, "prec": acc, "rec": acc, "f1": acc, "n": n,
                "note": "single_class_gt"}
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )
    return {"acc": acc, "prec": float(p), "rec": float(r), "f1": float(f), "n": n}


def _pick_winner(llm_f1, cv_f1, tol: float = 0.01) -> str:
    if np.isnan(llm_f1) and np.isnan(cv_f1):
        return "NA"
    if np.isnan(llm_f1): return "CV"
    if np.isnan(cv_f1):  return "LLM"
    if abs(cv_f1 - llm_f1) < tol:
        return "TIE"
    return "CV" if cv_f1 > llm_f1 else "LLM"


def make_comparison(gt_csv: str, llm_csv: str = "auto",
                    cv_csv: str = "auto", out_csv: str = "auto") -> None:
    gt_path = resolve_path(gt_csv)

    # LLM 경로 결정
    if llm_csv == "auto":
        llm_path = get_latest_run_path("llm", "llm_results.csv")
        if llm_path is None:
            raise RuntimeError("results/llm/ 에 저장된 LLM 결과가 없습니다. 먼저 llm_eval 을 실행하세요.")
    else:
        llm_path = resolve_path(llm_csv)

    # CV 경로 결정
    if cv_csv == "auto":
        cv_path = get_latest_run_path("cv", "cv_results.csv")
        if cv_path is None:
            raise RuntimeError("results/cv/ 에 저장된 CV 결과가 없습니다. 먼저 cv_eval 을 실행하세요.")
    else:
        cv_path = resolve_path(cv_csv)

    # 출력 경로 결정 (LLM 폴더 이름 기준으로 compare 폴더 동기화)
    if out_csv == "auto":
        llm_folder_name = llm_path.parent.name  # 예: "2차_30장"
        out_path = ensure_dir(PROJECT_ROOT / "results" / "compare" / llm_folder_name / "comparison_result.csv")
    else:
        out_path = ensure_dir(out_csv)

    # 차수 불일치 경고
    llm_run = llm_path.parent.name
    cv_run  = cv_path.parent.name
    if llm_run != cv_run:
        print(f"[compare] 주의: LLM({llm_run})과 CV({cv_run}) 차수가 다릅니다.")
    else:
        print(f"[compare] 사용 중인 실행: {llm_run}")

    gt  = _normalize_gt(pd.read_csv(gt_path))
    llm = pd.read_csv(llm_path)
    cv  = pd.read_csv(cv_path)

    # image_id 기준 inner join
    common = sorted(set(gt["image_id"].astype(str)) &
                    set(llm["image_id"].astype(str)) &
                    set(cv["image_id"].astype(str)))
    if not common:
        raise RuntimeError(
            "3개 CSV에 공통 image_id가 없습니다.\n"
            f"  GT  image_id 샘플: {gt['image_id'].head(3).tolist()}\n"
            f"  LLM image_id 샘플: {llm['image_id'].head(3).tolist()}\n"
            f"  CV  image_id 샘플: {cv['image_id'].head(3).tolist()}"
        )
    gt  = gt[gt["image_id"].astype(str).isin(common)].set_index("image_id").loc[common]
    llm = llm[llm["image_id"].astype(str).isin(common)].set_index("image_id").loc[common]
    cv  = cv[cv["image_id"].astype(str).isin(common)].set_index("image_id").loc[common]

    print(f"[compare] common N = {len(common)}")

    rows = []
    item_metrics_cache = {}

    # --- 1) 항목별 ---
    for item in ITEM_KEYS:
        if item not in gt.columns:
            print(f"  ! GT에 컬럼 없음: {item} (스킵)")
            continue
        llm_m = _metrics_for_item(gt[item], llm[item] if item in llm.columns else pd.Series(dtype=float))
        cv_m  = _metrics_for_item(gt[item], cv[item]  if item in cv.columns  else pd.Series(dtype=float))
        item_metrics_cache[item] = (llm_m, cv_m)

        delta_f1 = (cv_m["f1"] - llm_m["f1"]) if not (np.isnan(cv_m["f1"]) or np.isnan(llm_m["f1"])) else np.nan
        rows.append({
            "row_type": "item",
            "name": item,
            "group": ITEM_TO_GROUP.get(item, ""),
            "n": llm_m["n"],
            "llm_acc":  llm_m["acc"],  "llm_prec": llm_m["prec"],
            "llm_rec":  llm_m["rec"],  "llm_f1":   llm_m["f1"],
            "cv_acc":   cv_m["acc"],   "cv_prec":  cv_m["prec"],
            "cv_rec":   cv_m["rec"],   "cv_f1":    cv_m["f1"],
            "delta_f1": delta_f1,
            "winner":   _pick_winner(llm_m["f1"], cv_m["f1"]),
        })

    # --- 2) 그룹별 macro F1 ---
    for group_name, items in ITEM_GROUPS.items():
        valid = [item_metrics_cache.get(i) for i in items if i in item_metrics_cache]
        valid = [v for v in valid if v is not None]
        if not valid:
            continue
        llm_f1  = float(np.nanmean([v[0]["f1"]  for v in valid]))
        cv_f1   = float(np.nanmean([v[1]["f1"]  for v in valid]))
        llm_acc = float(np.nanmean([v[0]["acc"] for v in valid]))
        cv_acc  = float(np.nanmean([v[1]["acc"] for v in valid]))
        rows.append({
            "row_type": "group",
            "name": group_name,
            "group": group_name,
            "n": int(np.nansum([v[0]["n"] for v in valid])),
            "llm_acc": llm_acc, "llm_prec": np.nan, "llm_rec": np.nan, "llm_f1": llm_f1,
            "cv_acc":  cv_acc,  "cv_prec":  np.nan, "cv_rec":  np.nan, "cv_f1":  cv_f1,
            "delta_f1": cv_f1 - llm_f1,
            "winner": _pick_winner(llm_f1, cv_f1),
        })

    # --- 3) 전체 macro ---
    all_llm_f1 = [m[0]["f1"] for m in item_metrics_cache.values()]
    all_cv_f1  = [m[1]["f1"] for m in item_metrics_cache.values()]
    llm_macro = float(np.nanmean(all_llm_f1)) if all_llm_f1 else np.nan
    cv_macro  = float(np.nanmean(all_cv_f1))  if all_cv_f1  else np.nan
    rows.append({
        "row_type": "overall",
        "name": "ALL_ITEMS_MACRO",
        "group": "",
        "n": len(common),
        "llm_acc": np.nan, "llm_prec": np.nan, "llm_rec": np.nan, "llm_f1": llm_macro,
        "cv_acc":  np.nan, "cv_prec":  np.nan, "cv_rec":  np.nan, "cv_f1":  cv_macro,
        "delta_f1": cv_macro - llm_macro if not np.isnan(llm_macro) and not np.isnan(cv_macro) else np.nan,
        "winner": _pick_winner(llm_macro, cv_macro),
    })

    # --- 4) 치명 오류 Recall ---
    for item in CRITICAL_ITEMS:
        if item not in gt.columns:
            continue
        g = gt[item].apply(to_binary_or_none)
        l = llm[item].apply(to_binary_or_none) if item in llm.columns else pd.Series(dtype=float)
        c = cv[item].apply(to_binary_or_none)  if item in cv.columns  else pd.Series(dtype=float)
        neg_mask = (g == 0)
        n_neg = int(neg_mask.sum())
        if n_neg == 0:
            continue
        llm_caught = int(((l[neg_mask] == 0)).sum())
        cv_caught  = int(((c[neg_mask] == 0)).sum())
        llm_rec = llm_caught / n_neg
        cv_rec  = cv_caught  / n_neg
        rows.append({
            "row_type": "critical_recall",
            "name": item,
            "group": "critical",
            "n": n_neg,
            "llm_acc": np.nan, "llm_prec": np.nan, "llm_rec": llm_rec, "llm_f1": np.nan,
            "cv_acc":  np.nan, "cv_prec":  np.nan, "cv_rec":  cv_rec,  "cv_f1":  np.nan,
            "delta_f1": np.nan,
            "winner": _pick_winner(llm_rec, cv_rec),
        })

    # --- 5) 레벨 예측 정확도 ---
    # CV 구조적 상한 감지: 현재 CV는 최대 7개 항목만 평가 → 최대 총점 7 → 최고 L1
    cv_max_scored = int(cv["cv_items_scored"].max()) if "cv_items_scored" in cv.columns else 18
    from src.scoring import LEVEL_RANGES
    cv_max_level = "L0"
    for lvl, lo, hi in LEVEL_RANGES:
        if lo <= cv_max_scored <= hi:
            cv_max_level = lvl
            break
        if cv_max_scored > hi:
            cv_max_level = lvl
    cv_level_capped = cv_max_scored <= 6  # 최대 6점 = L1 이하
    if cv_level_capped:
        print(
            f"[compare] ! CV 레벨 비교 제한: 최대 채점 {cv_max_scored}항목 → 이론 상한 {cv_max_level}\n"
            f"          CV final_level 수치는 참고용입니다. 등급 비교는 LLM 기준으로 보세요."
        )

    if "final_level" in gt.columns and "final_level" in llm.columns and "final_level" in cv.columns:
        g_str = gt["final_level"].apply(
            lambda x: parse_level_str(x) if parse_level_str(x) else x
        )
        g_idx = g_str.map(LEVEL_ORDER)
        l_idx = llm["final_level"].map(LEVEL_ORDER)
        c_idx = cv["final_level"].map(LEVEL_ORDER)
        ok = g_idx.notna() & l_idx.notna() & c_idx.notna()
        if ok.sum() > 0:
            gl, ll, cl = g_idx[ok], l_idx[ok], c_idx[ok]
            cv_note = f"max_{cv_max_level}_cap" if cv_level_capped else ""
            rows.append({
                "row_type": "level_metric",
                "name": "exact_match",
                "group": "",
                "n": int(ok.sum()),
                "llm_acc": float((gl == ll).mean()),
                "llm_prec": np.nan, "llm_rec": np.nan, "llm_f1": np.nan,
                "cv_acc": float((gl == cl).mean()),
                "cv_prec": np.nan, "cv_rec": np.nan, "cv_f1": np.nan,
                "delta_f1": np.nan, "winner": "LLM" if cv_level_capped else "",
                "note": cv_note,
            })
            rows.append({
                "row_type": "level_metric",
                "name": "within_1",
                "group": "",
                "n": int(ok.sum()),
                "llm_acc": float(((gl - ll).abs() <= 1).mean()),
                "llm_prec": np.nan, "llm_rec": np.nan, "llm_f1": np.nan,
                "cv_acc": float(((gl - cl).abs() <= 1).mean()),
                "cv_prec": np.nan, "cv_rec": np.nan, "cv_f1": np.nan,
                "delta_f1": np.nan, "winner": "LLM" if cv_level_capped else "",
                "note": cv_note,
            })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig", float_format="%.4f")
    print(f"[compare] saved -> {out_path}")

    print("\n[요약]")
    print(out_df[out_df.row_type.isin(["group", "overall"])]
          [["name", "llm_f1", "cv_f1", "delta_f1", "winner"]]
          .to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt",  default="data/image_eval_human_gt.csv")
    parser.add_argument("--llm", default="auto",
                        help="LLM 결과 CSV. 기본값 'auto' 는 results/llm/ 의 최신 차수 사용")
    parser.add_argument("--cv",  default="auto",
                        help="CV 결과 CSV. 기본값 'auto' 는 results/cv/ 의 최신 차수 사용")
    parser.add_argument("--out", default="auto",
                        help="저장 경로. 기본값 'auto' 는 results/compare/n차_m장/ 에 자동 저장")
    args = parser.parse_args()
    make_comparison(args.gt, args.llm, args.cv, args.out)
