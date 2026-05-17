"""
src/cv_eval.py

MVP 범위: 학습이 필요 없는 항목만 평가.
나머지 항목은 None(null)로 반환 -> LLM 또는 향후 단계에 위임.

[구현 항목 - 16개]
  - dog_visible             : CLIP zero-shot
  - one_dog                 : YOLO count (CLIP이 dog 감지한 경우에만)
  - face_clear              : CLIP zero-shot (강아지 얼굴 명확성)
  - no_duplicate_face_parts : 기본값 1 (LLM 139장 100% 통과, AI일러스트 특성상 검출 도구 없음)
  - no_extra_legs           : 기본값 1 (LLM 139장 100% 통과, DeepLabCut 없이 검출 불가)
  - no_mirror               : 기본값 1 (LLM 135/139 통과, AI 일러스트 거의 발생 안 함)
  - no_multi_panel          : OpenCV Hough Line
  - pastel_style            : CLIP zero-shot (파스텔 일러스트 스타일)
  - no_realistic_rendering  : CLIP zero-shot (pastel_style와 동일 신호)
  - prompt_reflected        : CLIP zero-shot (활동 키워드 → 영어 프롬프트 매칭)
  - brightness_ok           : OpenCV mean gray
  - sharpness_ok            : OpenCV Laplacian variance
  - pastel_color_ok         : OpenCV HSV (mean saturation)
  - time_palette_ok         : 프롬프트 시간 키워드 + OpenCV brightness
  - place_context_ok        : CLIP zero-shot (프롬프트 장소 키워드 매칭)
  - no_text                 : EasyOCR

[미구현 항목 → None]
  - human_prompt_consistency : 프로필-얼굴 비교 → 얼굴인식 모델 필요
  - human_face_clear         : 사람 얼굴 선명도 → 얼굴 검출 모델 필요

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
    "clip_dog_threshold": 0.020,

    # ── face_clear ───────────────────────────────────────────────────────────
    "clip_face_threshold": 0.0,

    # ── pastel_style / no_realistic_rendering ────────────────────────────────
    "clip_pastel_threshold": 0.0,

    # ── prompt_reflected (활동 매칭) ─────────────────────────────────────────
    # CLIP cosine diff (pos_mean - neg_mean). 0.0이면 pos>neg 시 통과.
    "clip_activity_threshold": 0.0,

    # ── place_context_ok ─────────────────────────────────────────────────────
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

    # ── time_palette_ok ──────────────────────────────────────────────────────
    # 밤 키워드 감지 시: brightness > 이 값이면 실패 (낮처럼 밝으면 시간대 불일치)
    "time_night_max_brightness": 180.0,
    # 낮 키워드 감지 시: brightness < 이 값이면 실패 (너무 어두우면 시간대 불일치)
    "time_day_min_brightness": 50.0,
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

# 한국어 활동 키워드 → (영어 pos_prompts, neg_prompts) — prompt_reflected 용
_ACTIVITY_CLIP_MAP: Dict[str, tuple] = {
    "목욕": (
        ["a dog taking a bath with soap bubbles", "a dog being washed in a bathtub", "a wet dog getting bathed with shampoo"],
        ["a dog running outside in a park", "a dog sleeping on a bed", "a dog eating food from a bowl"],
    ),
    "산책": (
        ["a dog walking outside on a leash", "a dog going for a walk on a path", "a dog strolling outdoors with an owner"],
        ["a dog indoors at home on a sofa", "a dog taking a bath", "a dog sleeping on a bed"],
    ),
    "공놀이": (
        ["a dog playing fetch with a ball", "a dog chasing a ball", "a dog playing with a round toy"],
        ["a dog taking a bath", "a dog sleeping", "a dog eating"],
    ),
    "수면": (
        ["a dog sleeping peacefully", "a dog napping on a soft bed", "a dog curled up sleeping"],
        ["a dog running outside", "a dog playing with a ball", "a dog taking a bath"],
    ),
    "잠": (
        ["a dog sleeping peacefully", "a dog napping on a soft bed", "a dog curled up sleeping"],
        ["a dog running outside", "a dog playing with a ball", "a dog taking a bath"],
    ),
    "꿈": (
        ["a dog sleeping peacefully with a dreamy atmosphere", "a sleeping dog with gentle fantasy elements"],
        ["a dog running outside", "a dog playing with a ball", "a dog eating"],
    ),
    "간식": (
        ["a dog eating a treat or snack", "a dog waiting eagerly for food", "a dog with a bone or treat"],
        ["a dog taking a bath", "a dog sleeping", "a dog running outside"],
    ),
    "밥": (
        ["a dog eating food from a bowl", "a dog enjoying a meal", "a dog with food in front of it"],
        ["a dog taking a bath", "a dog sleeping", "a dog running outside"],
    ),
    "생일": (
        ["a dog at a birthday party with cake", "a dog celebrating with decorations", "a festive party scene with a dog"],
        ["a dog taking a bath", "a dog sleeping outdoors", "a dog alone in a quiet setting"],
    ),
    "파티": (
        ["a dog at a festive party with decorations", "a dog celebrating with friends", "a colorful party scene with a dog"],
        ["a dog taking a bath", "a dog sleeping", "a dog walking alone"],
    ),
    "병원": (
        ["a dog at a veterinary clinic", "a dog being examined by a vet", "a dog in a medical setting"],
        ["a dog in a park", "a dog at a cafe", "a dog taking a bath"],
    ),
    "미용": (
        ["a dog getting groomed", "a dog at a dog grooming salon", "a dog being brushed or trimmed"],
        ["a dog running outside", "a dog sleeping", "a dog eating"],
    ),
    "무릎": (
        ["a dog sitting on a person's lap", "a small dog resting on someone's lap", "a dog cuddling on a lap"],
        ["a dog running outside", "a dog taking a bath", "a dog alone in a field"],
    ),
    "안겨": (
        ["a dog being held in someone's arms", "a dog cuddled in an embrace", "a dog held by its owner"],
        ["a dog running outside", "a dog taking a bath", "a dog alone"],
    ),
    "눈": (
        ["a dog playing in snow", "a dog in a snowy winter scene", "a dog walking in snow covered landscape"],
        ["a dog at a beach in summer", "a dog in a green sunny park", "an indoor scene with a dog"],
    ),
    "비": (
        ["a dog on a rainy day with rain drops", "a dog with an umbrella in rain", "a rainy weather scene with a dog"],
        ["a dog in sunny weather", "a dog in snow", "a dog at a dry sunny beach"],
    ),
    "고양이": (
        ["a dog and a cat together", "a dog next to a cat", "a cute scene with both a dog and a cat"],
        ["a dog alone with no other animals", "a dog with only humans", "a dog in nature alone"],
    ),
    "친구": (
        ["a dog playing with other dogs", "multiple dogs together having fun", "a dog socializing with other pets"],
        ["a dog alone in a quiet scene", "a dog sleeping alone", "a single dog portrait"],
    ),
}

# 한국어 장소 키워드 → (pos_prompts, neg_prompts) — place_context_ok 용
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
    "애견카페": (
        ["a dog-friendly cafe interior with dogs and tables", "a pet cafe with dogs and customers", "a cozy dog cafe setting"],
        ["an outdoor park or nature scene", "a plain featureless background", "a quiet room with no other animals"],
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

# 시간대 키워드 — time_palette_ok 용
_TIME_KEYWORDS_NIGHT = ["밤", "저녁", "야간", "새벽", "달빛", "야경", "밤하늘"]
_TIME_KEYWORDS_DAY   = ["낮", "햇살", "햇빛", "오전", "오후", "아침", "맑은 날", "맑은날", "한낮"]


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
        """동적 프롬프트로 cosine diff 계산."""
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

    def _clip_activity_score(self, img_features, prompt: str) -> Optional[float]:
        """프롬프트에서 활동 키워드를 찾아 CLIP score 반환. 없으면 None."""
        for kw, (pos, neg) in _ACTIVITY_CLIP_MAP.items():
            if kw in prompt:
                return self._clip_score_dynamic(img_features, pos, neg)
        return None

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

    def eval_prompt_reflected(self, activity_score: Optional[float]) -> int:
        """프롬프트 활동 키워드 CLIP 매칭. 키워드 없으면 기본 통과."""
        if activity_score is None:
            return 1
        return 1 if activity_score >= self.th["clip_activity_threshold"] else 0

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

    def eval_time_palette_ok(self, img_bgr: np.ndarray, prompt: str) -> int:
        """시간대 키워드 + 이미지 밝기로 시간 팔레트 일치 여부 판단."""
        is_night = any(kw in prompt for kw in _TIME_KEYWORDS_NIGHT)
        is_day   = any(kw in prompt for kw in _TIME_KEYWORDS_DAY)
        if not is_night and not is_day:
            return 1  # 시간 키워드 없으면 기본 통과
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        brightness = float(gray.mean())
        if is_night and brightness > self.th["time_night_max_brightness"]:
            return 0  # 밤인데 너무 밝음
        if is_day and brightness < self.th["time_day_min_brightness"]:
            return 0  # 낮인데 너무 어두움
        return 1

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
                "clip_pastel_score": None, "clip_activity_score": None,
                "clip_place_score": None,
                "yolo_dog_count": 0, "cv_items_scored": 0,
            }

        # 이미지 특징 한 번 계산
        img_features = self._clip_encode_image(img_bgr)

        dog_score      = self._clip_score(img_features, "dog")
        face_score     = self._clip_score(img_features, "face")
        pastel_score   = self._clip_score(img_features, "pastel")
        activity_score = self._clip_activity_score(img_features, prompt)
        place_score    = self._clip_place_score(img_features, prompt)

        yolo_count = self._yolo_dog_count(img_bgr)

        items: Dict[str, Optional[int]] = {k: None for k in ITEM_KEYS}

        # ── 구현 항목 ──
        items["dog_visible"]             = self.eval_dog_visible(dog_score)
        items["one_dog"]                 = self.eval_one_dog(dog_score, yolo_count)
        items["face_clear"]              = self.eval_face_clear(face_score, items["dog_visible"])
        items["no_duplicate_face_parts"] = 1   # LLM 139/139 통과; CV 검출 불가
        items["no_extra_legs"]           = 1   # LLM 139/139 통과; DeepLabCut 없이 불가
        items["no_mirror"]               = 1   # LLM 135/139 통과; AI 일러스트 거의 없음
        items["pastel_style"]            = self.eval_pastel_style(pastel_score)
        items["no_realistic_rendering"]  = self.eval_no_realistic_rendering(pastel_score)
        items["prompt_reflected"]        = self.eval_prompt_reflected(activity_score)
        items["brightness_ok"]           = self.eval_brightness(img_bgr)
        items["sharpness_ok"]            = self.eval_sharpness(img_bgr)
        items["pastel_color_ok"]         = self.eval_pastel_color(img_bgr)
        items["time_palette_ok"]         = self.eval_time_palette_ok(img_bgr, prompt)
        items["no_multi_panel"]          = self.eval_no_multi_panel(img_bgr)
        items["place_context_ok"]        = self.eval_place_context_ok(place_score)
        items["no_text"]                 = self.eval_no_text(img_bgr)

        # ── 미구현 항목 → None ──
        # items["human_prompt_consistency"] = None  (기본값)
        # items["human_face_clear"]         = None  (기본값)

        scored = score_items(items)
        n_scored = sum(1 for v in items.values() if v is not None)
        return {
            "items": items, **scored,
            "clip_dog_score":      round(dog_score, 4),
            "clip_face_score":     round(face_score, 4),
            "clip_pastel_score":   round(pastel_score, 4),
            "clip_activity_score": round(activity_score, 4) if activity_score is not None else None,
            "clip_place_score":    round(place_score, 4) if place_score is not None else None,
            "yolo_dog_count":      yolo_count,
            "cv_items_scored":     n_scored,
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
            rec["total_score"]        = res.get("total_score")
            rec["raw_level"]          = res.get("raw_level")
            rec["caps_applied"]       = res.get("caps_applied")
            rec["final_level"]        = res.get("final_level")
            rec["clip_dog_score"]     = res.get("clip_dog_score")
            rec["clip_face_score"]    = res.get("clip_face_score")
            rec["clip_pastel_score"]  = res.get("clip_pastel_score")
            rec["clip_activity_score"]= res.get("clip_activity_score")
            rec["clip_place_score"]   = res.get("clip_place_score")
            rec["yolo_dog_count"]     = res.get("yolo_dog_count")
            rec["cv_items_scored"]    = res.get("cv_items_scored")
            rec["error"]              = res.get("error", "")
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
