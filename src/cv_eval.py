"""
src/cv_eval.py

MVP 범위: 학습이 필요 없는 항목만 평가.
나머지 항목은 None(null)로 반환 -> LLM 또는 향후 단계에 위임.

[구현 항목 - 11개]
  - dog_visible          : CLIP zero-shot
  - one_dog              : YOLO count (CLIP이 dog 감지한 경우에만)
  - face_clear           : CLIP zero-shot (강아지 얼굴 명확성)
  - brightness_ok        : OpenCV mean gray
  - sharpness_ok         : OpenCV Laplacian variance
  - pastel_color_ok      : OpenCV HSV (mean saturation)
  - pastel_style         : CLIP zero-shot (파스텔 일러스트 스타일)
  - no_realistic_rendering: CLIP zero-shot (pastel_style와 동일 신호, 반전 없음)
  - no_multi_panel       : OpenCV Hough Line
  - no_text              : EasyOCR
  - place_context_ok     : CLIP zero-shot (프롬프트 장소 키워드 매칭)

[CLIP 구조]
  - 이미지 특징을 한 번만 계산 후 모든 항목에 재사용
  - cosine similarity difference (pos_mean - neg_mean) 방식

실행:
  python -m src.cv_eval --gt data/image_eval_human_gt.csv --out results/cv_results.csv
"""
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.scoring import ITEM_KEYS, score_items
from src.utils import safe_imread, resolve_path, ensure_dir, get_run_out_path


DEFAULT_THRESHOLDS = {
    # ── dog_visible ──────────────────────────────────────────────────────────
    # CLIP ViT-B-32 실측 (139장):
    #   전형적 강아지: +0.023 ~ +0.065  비전형(그레이하운드 등): -0.002 ~ +0.020
    #   비강아지: -0.015 ~ +0.020
    # threshold=0.020: TP=96.4%(134/139), FP=9.1%(3/33)
    "clip_dog_threshold": 0.020,

    # ── face_clear ───────────────────────────────────────────────────────────
    # 강아지 얼굴 선명도. dog_visible=1 인 경우에만 평가.
    # 초기값: 0.0 (pos>neg 이면 통과). 10장 캘리브레이션 후 조정.
    "clip_face_threshold": 0.0,

    # ── pastel_style / no_realistic_rendering ────────────────────────────────
    # 파스텔 일러스트 vs 사실적 렌더링. 동일 score 사용, 부호 동일.
    # 초기값: 0.0. 10장 캘리브레이션 후 조정.
    "clip_pastel_threshold": 0.0,

    # ── place_context_ok ─────────────────────────────────────────────────────
    # 프롬프트 장소 키워드 매칭. 키워드 없으면 None 반환.
    # 초기값: 0.0.
    "clip_place_threshold": 0.0,

    # ── one_dog (YOLO) ───────────────────────────────────────────────────────
    "yolo_conf": 0.10,

    # ── OpenCV 기반 ──────────────────────────────────────────────────────────
    "brightness_min": 55.0,
    "brightness_max": 215.0,
    "sharpness_var_min": 80.0,
    "hsv_sat_min": 20.0,
    "hsv_sat_max": 165.0,
    "ocr_min_conf": 0.40,
    "ocr_min_text_len": 2,
    "hough_min_panel_lines": 2,
}

# ── CLIP 텍스트 프롬프트 ────────────────────────────────────────────────────

_CLIP_DOG_POS = [
    "a picture of a dog",
    "a cute cartoon dog",
    "an illustration of a dog",
    "a pastel drawing of a dog",
    "a dog in a children's book style",
]
_CLIP_DOG_NEG = [
    "a picture without a dog",
    "a scene with only people and no animals",
    "a landscape with no dog",
    "an illustration with no animals",
]

_CLIP_FACE_POS = [
    "a dog with a clear well-drawn cute face",
    "a dog with expressive eyes and clearly illustrated facial features",
    "a dog face with neat clean distinct features in illustration style",
]
_CLIP_FACE_NEG = [
    "a dog with a blurry distorted face",
    "a dog with an unclear smeared malformed deformed face",
    "a dog with no discernible facial features",
]

_CLIP_PASTEL_POS = [
    "soft pastel illustration artwork in picture book style",
    "cute gentle watercolor pastel children's book illustration",
    "Korean Japanese pastel soft color gentle illustration",
]
_CLIP_PASTEL_NEG = [
    "realistic photographic image",
    "photorealistic 3D CGI rendering",
    "detailed realistic computer graphics photography",
]

# 한국어 장소 키워드 → (pos_prompts, neg_prompts)
_LOCATION_CLIP_MAP: Dict[str, tuple] = {
    "공원": (
        ["a dog in a park with trees and grass", "outdoor park scenery with a dog", "a sunny park with benches and a dog"],
        ["a plain featureless white background", "an indoor room with no outside view", "a minimal background with no setting"],
    ),
    "집": (
        ["a dog inside a cozy home living room", "indoor home scene with furniture and a dog", "a dog relaxing in a house interior"],
        ["an outdoor park or nature scene", "a plain featureless background", "a beach or outdoor scenery"],
    ),
    "카페": (
        ["a cute cafe interior with tables and a dog", "a cozy coffee shop scene", "a dog in a cafe setting with warm lighting"],
        ["an outdoor park or nature scene", "a plain featureless background", "a beach or outdoor scenery"],
    ),
    "해변": (
        ["a beach scene with sand and ocean waves", "a dog on a sunny beach", "seaside beach scenery with a dog"],
        ["an indoor home scene", "a plain white background", "a city street scene"],
    ),
    "바다": (
        ["a seaside ocean scene with water and waves", "a dog near the ocean", "beach and sea scenery with a dog"],
        ["an indoor home scene", "a plain white background", "a city street scene"],
    ),
    "숲": (
        ["a forest with tall trees and nature", "a dog in a lush green forest", "woodland nature scenery with a dog"],
        ["an indoor home scene", "a plain white background", "a beach or ocean scene"],
    ),
    "산": (
        ["a mountain trail scenery with trees", "a dog on a mountain path", "mountain landscape nature scene"],
        ["an indoor home scene", "a plain white background", "a beach or ocean scene"],
    ),
    "산책로": (
        ["a walking trail path with trees and greenery", "a dog on a scenic walking path", "a promenade with plants and a dog"],
        ["an indoor home scene", "a plain white background", "a beach or ocean scene"],
    ),
    "마당": (
        ["a backyard garden with grass and plants", "a dog in a home garden yard", "outdoor backyard scenery with a dog"],
        ["an indoor home scene", "a plain white background", "a beach or ocean scene"],
    ),
    "놀이터": (
        ["a playground with swings and slides", "a dog in a children's playground", "outdoor playground park scene"],
        ["an indoor home scene", "a plain white background", "a beach or ocean scene"],
    ),
    "실내": (
        ["an indoor room interior with furniture", "a cozy indoor scene with a dog", "interior home or building scene"],
        ["an outdoor park or nature scene", "a plain white background", "a beach scene"],
    ),
    "야외": (
        ["an outdoor nature scene with sky and trees", "a dog in an outdoor setting", "outdoor scenery with nature"],
        ["an indoor home scene", "a plain white background", "a featureless background"],
    ),
    "실외": (
        ["an outdoor nature scene with sky and trees", "a dog in an outdoor setting", "outdoor scenery with nature"],
        ["an indoor home scene", "a plain white background", "a featureless background"],
    ),
}


class CVEvaluator:
    """CLIP + YOLO + OpenCV + EasyOCR 통합 평가기. 모델은 lazy 로딩."""

    def __init__(self, thresholds: Optional[dict] = None, use_gpu: bool = False):
        self.th = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
        self.use_gpu = use_gpu
        self._clip_model = None
        self._clip_preprocess = None
        self._clip_tokenizer = None
        self._clip_text_cache: Dict[str, tuple] = {}  # key -> (features_tensor, n_pos)
        self._yolo = None
        self._ocr = None

    # ------------------------------------------------------------------ loaders

    def _load_clip(self):
        if self._clip_model is not None:
            return
        import torch
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        model.eval()
        tokenizer = open_clip.get_tokenizer("ViT-B-32")
        self._clip_model = model
        self._clip_preprocess = preprocess
        self._clip_tokenizer = tokenizer

        # 고정 프롬프트 그룹 사전 계산
        for key, (pos, neg) in [
            ("dog",    (_CLIP_DOG_POS,    _CLIP_DOG_NEG)),
            ("face",   (_CLIP_FACE_POS,   _CLIP_FACE_NEG)),
            ("pastel", (_CLIP_PASTEL_POS, _CLIP_PASTEL_NEG)),
        ]:
            self._clip_text_cache[key] = self._precompute_text(pos + neg, len(pos))

    def _precompute_text(self, all_texts: List[str], n_pos: int) -> tuple:
        import torch
        with torch.no_grad():
            tokens = self._clip_tokenizer(all_texts)
            feats = self._clip_model.encode_text(tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats, n_pos

    def _load_yolo(self):
        if self._yolo is None:
            from ultralytics import YOLO
            _model_path = Path(__file__).resolve().parent.parent / "models" / "yolov8n.pt"
            self._yolo = YOLO(str(_model_path))

    def _load_ocr(self):
        if self._ocr is None:
            import easyocr
            self._ocr = easyocr.Reader(["ko", "en"], gpu=self.use_gpu, verbose=False)

    # ------------------------------------------------------------------ CLIP core

    def _clip_encode_image(self, img_bgr: np.ndarray):
        """이미지를 정규화된 CLIP 특징 벡터로 인코딩. 한 이미지당 한 번만 호출."""
        import torch
        self._load_clip()
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        img_tensor = self._clip_preprocess(pil_img).unsqueeze(0)
        with torch.no_grad():
            feats = self._clip_model.encode_image(img_tensor)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats

    def _clip_score(self, img_features, key: str) -> float:
        """사전 계산된 텍스트 특징으로 cosine diff 계산."""
        feats, n_pos = self._clip_text_cache[key]
        sims = (img_features @ feats.T)[0]
        pos_mean = float(sims[:n_pos].mean().item())
        neg_mean = float(sims[n_pos:].mean().item())
        return pos_mean - neg_mean

    def _clip_score_dynamic(self, img_features, pos_texts: List[str], neg_texts: List[str]) -> float:
        """동적 프롬프트로 cosine diff 계산 (place_context_ok 용)."""
        import torch
        self._load_clip()
        all_texts = pos_texts + neg_texts
        n_pos = len(pos_texts)
        with torch.no_grad():
            tokens = self._clip_tokenizer(all_texts)
            text_feats = self._clip_model.encode_text(tokens)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
        sims = (img_features @ text_feats.T)[0]
        pos_mean = float(sims[:n_pos].mean().item())
        neg_mean = float(sims[n_pos:].mean().item())
        return pos_mean - neg_mean

    def _clip_dog_score(self, img_bgr: np.ndarray) -> float:
        """fp_test.py 등 외부에서 직접 호출하는 backward-compat 래퍼."""
        img_features = self._clip_encode_image(img_bgr)
        return self._clip_score(img_features, "dog")

    def _clip_place_score(self, img_features, prompt: str) -> Optional[float]:
        """프롬프트에서 장소 키워드를 찾아 CLIP score 반환. 없으면 None."""
        for kw, (pos, neg) in _LOCATION_CLIP_MAP.items():
            if kw in prompt:
                return self._clip_score_dynamic(img_features, pos, neg)
        return None

    # ------------------------------------------------------------------ YOLO

    def _yolo_dog_count(self, img_bgr: np.ndarray) -> int:
        self._load_yolo()
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        results = self._yolo.predict(img_rgb, verbose=False, conf=self.th["yolo_conf"])
        count = 0
        for r in results:
            if r.boxes is None:
                continue
            for c in r.boxes.cls:
                if int(c.item()) == 16:
                    count += 1
        return count

    # ------------------------------------------------------------------ evaluators

    def eval_dog_visible(self, dog_score: float) -> int:
        return 1 if dog_score >= self.th["clip_dog_threshold"] else 0

    def eval_one_dog(self, dog_score: float, yolo_count: int) -> Optional[int]:
        if dog_score < self.th["clip_dog_threshold"]:
            return None
        if yolo_count >= 1:
            return 1 if yolo_count == 1 else 0
        return None

    def eval_face_clear(self, face_score: float, dog_visible: int) -> Optional[int]:
        if dog_visible == 0:
            return None
        return 1 if face_score >= self.th["clip_face_threshold"] else 0

    def eval_pastel_style(self, pastel_score: float) -> int:
        return 1 if pastel_score >= self.th["clip_pastel_threshold"] else 0

    def eval_no_realistic_rendering(self, pastel_score: float) -> int:
        return 1 if pastel_score >= self.th["clip_pastel_threshold"] else 0

    def eval_place_context_ok(self, place_score: Optional[float]) -> Optional[int]:
        if place_score is None:
            return None
        return 1 if place_score >= self.th["clip_place_threshold"] else 0

    def eval_brightness(self, img_bgr: np.ndarray) -> int:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        m = float(gray.mean())
        return 1 if self.th["brightness_min"] <= m <= self.th["brightness_max"] else 0

    def eval_sharpness(self, img_bgr: np.ndarray) -> int:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        return 1 if var >= self.th["sharpness_var_min"] else 0

    def eval_pastel_color(self, img_bgr: np.ndarray) -> int:
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        sat_mean = float(hsv[..., 1].mean())
        return 1 if self.th["hsv_sat_min"] <= sat_mean <= self.th["hsv_sat_max"] else 0

    def eval_no_multi_panel(self, img_bgr: np.ndarray) -> int:
        h, w = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 200)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, 200,
            minLineLength=int(0.8 * min(h, w)),
            maxLineGap=10,
        )
        n_panel = 0
        if lines is not None:
            for l in lines[:50]:
                x1, y1, x2, y2 = l[0]
                if abs(x1 - x2) < 5 or abs(y1 - y2) < 5:
                    n_panel += 1
        return 0 if n_panel >= self.th["hough_min_panel_lines"] else 1

    def eval_no_text(self, img_bgr: np.ndarray) -> int:
        self._load_ocr()
        try:
            results = self._ocr.readtext(img_bgr)
        except Exception as e:
            print(f"[OCR] failed: {e}", file=sys.stderr)
            return 1
        for (_bbox, txt, conf) in results:
            if conf < self.th["ocr_min_conf"]:
                continue
            if len(str(txt).strip()) < self.th["ocr_min_text_len"]:
                continue
            return 0
        return 1

    # ------------------------------------------------------------------ main

    def evaluate(self, image_path: str, prompt: str = "") -> Dict:
        img_bgr = safe_imread(image_path)
        if img_bgr is None:
            items = {k: None for k in ITEM_KEYS}
            items["dog_visible"] = 0
            scored = score_items(items)
            return {
                "items": items, **scored, "error": "image_load_failed",
                "clip_dog_score": None, "clip_face_score": None,
                "clip_pastel_score": None, "clip_place_score": None,
                "yolo_dog_count": 0, "cv_items_scored": 0,
            }

        # 이미지 특징 한 번 계산
        img_features = self._clip_encode_image(img_bgr)

        dog_score    = self._clip_score(img_features, "dog")
        face_score   = self._clip_score(img_features, "face")
        pastel_score = self._clip_score(img_features, "pastel")
        place_score  = self._clip_place_score(img_features, prompt)

        yolo_count = self._yolo_dog_count(img_bgr)

        items: Dict[str, Optional[int]] = {k: None for k in ITEM_KEYS}
        items["dog_visible"]            = self.eval_dog_visible(dog_score)
        items["one_dog"]                = self.eval_one_dog(dog_score, yolo_count)
        items["face_clear"]             = self.eval_face_clear(face_score, items["dog_visible"])
        items["pastel_style"]           = self.eval_pastel_style(pastel_score)
        items["no_realistic_rendering"] = self.eval_no_realistic_rendering(pastel_score)
        items["brightness_ok"]          = self.eval_brightness(img_bgr)
        items["sharpness_ok"]           = self.eval_sharpness(img_bgr)
        items["pastel_color_ok"]        = self.eval_pastel_color(img_bgr)
        items["no_multi_panel"]         = self.eval_no_multi_panel(img_bgr)
        items["no_text"]                = self.eval_no_text(img_bgr)
        items["place_context_ok"]       = self.eval_place_context_ok(place_score)

        scored = score_items(items)
        n_scored = sum(1 for v in items.values() if v is not None)
        return {
            "items": items, **scored,
            "clip_dog_score":    round(dog_score, 4),
            "clip_face_score":   round(face_score, 4),
            "clip_pastel_score": round(pastel_score, 4),
            "clip_place_score":  round(place_score, 4) if place_score is not None else None,
            "yolo_dog_count":    yolo_count,
            "cv_items_scored":   n_scored,
        }


def _get_image_id(row: pd.Series) -> str:
    for col in ("image_id", "session_id"):
        v = row.get(col)
        if v is not None and str(v).strip() not in ("", "nan"):
            return str(v).strip()
    return ""


def _get_image_path(row: pd.Series) -> str:
    v = row.get("image_path")
    if v is not None and str(v).strip() not in ("", "nan"):
        return str(v).strip()
    v = row.get("image")
    if v is not None and str(v).strip() not in ("", "nan"):
        return f"data/eval_images/{str(v).strip()}"
    sid = _get_image_id(row)
    return f"data/eval_images/{sid}.png"


def _get_prompt(row: pd.Series) -> str:
    for col in ("image_prompt_base", "diary_summary", "diary_content"):
        v = row.get(col)
        if v is not None and str(v).strip() not in ("", "nan"):
            return str(v).strip()
    return ""


def evaluate_dataset(gt_csv: str, out_csv: str = "auto", limit: Optional[int] = None) -> None:
    gt_path = resolve_path(gt_csv)
    df = pd.read_csv(gt_path)

    if "image_generated" in df.columns:
        before = len(df)
        df = df[df["image_generated"].astype(str).str.upper() == "Y"].copy()
        print(f"[CV] image_generated=Y 필터: {before} -> {len(df)}행")

    if limit:
        df = df.head(limit)

    if out_csv == "auto":
        out_path = get_run_out_path("cv", len(df))
    else:
        out_path = ensure_dir(out_csv)

    ev = CVEvaluator()
    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="CV eval"):
        image_id = _get_image_id(row)
        rec = {"image_id": image_id}
        try:
            img_path = _get_image_path(row)
            prompt   = _get_prompt(row)
            res = ev.evaluate(img_path, prompt)
            rec.update(res["items"])
            rec["total_score"]      = res.get("total_score")
            rec["raw_level"]        = res.get("raw_level")
            rec["caps_applied"]     = res.get("caps_applied")
            rec["final_level"]      = res.get("final_level")
            rec["clip_dog_score"]   = res.get("clip_dog_score")
            rec["clip_face_score"]  = res.get("clip_face_score")
            rec["clip_pastel_score"]= res.get("clip_pastel_score")
            rec["clip_place_score"] = res.get("clip_place_score")
            rec["yolo_dog_count"]   = res.get("yolo_dog_count")
            rec["cv_items_scored"]  = res.get("cv_items_scored")
            rec["error"]            = res.get("error", "")
        except Exception as e:
            rec["error"] = str(e)
        rows.append(rec)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[CV] saved -> {out_path}  (n={len(out_df)})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt",    default="data/image_eval_human_gt.csv")
    parser.add_argument("--out",   default="auto",
                        help="저장 경로. 기본값 'auto'는 results/cv/n차_m장/ 에 자동 저장")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    evaluate_dataset(args.gt, args.out, args.limit)
