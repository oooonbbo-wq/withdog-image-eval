"""
새 CLIP 항목(face_clear, pastel_style, place_context_ok) threshold 캘리브레이션.
10장에 대해 raw CLIP score + 현재 예측 + GT 레벨을 출력한다.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from src.utils import PROJECT_ROOT, parse_level_str, safe_imread, resolve_path
from src.cv_eval import CVEvaluator, _get_image_path, _get_prompt, _get_image_id, DEFAULT_THRESHOLDS

GT_PATH  = PROJECT_ROOT / "data" / "image_eval_human_gt.csv"
LIMIT    = 10

gt = pd.read_csv(GT_PATH)
gt = gt[gt["image_generated"].astype(str).str.upper() == "Y"].copy()
gt["level"] = gt["Lv.평가"].apply(parse_level_str)
gt["_id"]   = gt["session_id"].astype(str).str.strip()
gt_sub = gt.head(LIMIT)

ev = CVEvaluator()

print(f"{'image_id':<25} {'GT':>3}  {'dog':>7} {'face':>7} {'past':>7} {'plac':>7}  "
      f"{'dv':>3} {'fc':>3} {'ps':>3} {'nrr':>4} {'pco':>4}  {'tot':>4} {'lv':>4}")
print("-" * 95)

rows = []
for _, row in gt_sub.iterrows():
    img_id   = _get_image_id(row)
    img_path = _get_image_path(row)
    prompt   = _get_prompt(row)
    gt_lv    = row["level"]

    img_bgr = safe_imread(resolve_path(img_path))
    if img_bgr is None:
        print(f"  {img_id:<23} {gt_lv:>3}  [이미지 로드 실패]")
        continue

    img_features = ev._clip_encode_image(img_bgr)
    dog_sc   = ev._clip_score(img_features, "dog")
    face_sc  = ev._clip_score(img_features, "face")
    past_sc  = ev._clip_score(img_features, "pastel")
    plac_sc  = ev._clip_place_score(img_features, prompt)

    yolo_cnt = ev._yolo_dog_count(img_bgr)

    dv  = ev.eval_dog_visible(dog_sc)
    fc  = ev.eval_face_clear(face_sc, dv)
    ps  = ev.eval_pastel_style(past_sc)
    nrr = ev.eval_no_realistic_rendering(past_sc)
    pco = ev.eval_place_context_ok(plac_sc)

    res = ev.evaluate(img_path, prompt)
    tot = res["total_score"]
    lv  = res["final_level"]

    plac_str = f"{plac_sc:+.4f}" if plac_sc is not None else "  None"
    fc_str   = str(fc) if fc is not None else " N"
    pco_str  = str(pco) if pco is not None else "  N"

    print(f"  {img_id:<23} {gt_lv:>3}  {dog_sc:+.4f} {face_sc:+.4f} {past_sc:+.4f} {plac_str}  "
          f"{dv:>3} {fc_str:>3} {ps:>3} {nrr:>4} {pco_str:>4}  {tot:>4} {lv:>4}")

    rows.append({
        "image_id": img_id, "gt_level": gt_lv,
        "dog_score": round(dog_sc, 4), "face_score": round(face_sc, 4),
        "pastel_score": round(past_sc, 4),
        "place_score": round(plac_sc, 4) if plac_sc is not None else None,
        "prompt_excerpt": prompt[:40] if prompt else "",
        "dog_visible": dv, "face_clear": fc,
        "pastel_style": ps, "no_realistic_rendering": nrr, "place_context_ok": pco,
        "cv_total": tot, "cv_level": lv,
    })

print()
print("현재 threshold 설정:")
for k in ("clip_dog_threshold", "clip_face_threshold", "clip_pastel_threshold", "clip_place_threshold"):
    print(f"  {k}: {DEFAULT_THRESHOLDS[k]}")

print()
print("[컬럼 설명] dog/face/past/plac = CLIP score, dv=dog_visible, fc=face_clear,")
print("           ps=pastel_style, nrr=no_realistic_rendering, pco=place_context_ok")

pd.DataFrame(rows).to_csv(
    PROJECT_ROOT / "results" / "cv_calibration_10.csv",
    index=False, encoding="utf-8-sig"
)
print("\n저장 완료 -> results/cv_calibration_10.csv")
