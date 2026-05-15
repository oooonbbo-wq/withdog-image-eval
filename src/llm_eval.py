"""
src/llm_eval.py

OpenAI GPT-4o vision API로 이미지를 18개 항목에 대해 평가한다.
점수/캡/레벨은 LLM이 아니라 scoring.py 에서 계산.

[모드]
  REAL  : OPENAI_API_KEY 가 있고 LLM_MOCK_MODE 가 false (기본)
  MOCK  : LLM_MOCK_MODE=true 또는 API 키가 없을 때
          - image_id 시드 기반 의사 난수로 18항목 생성
          - API 비용 0, 인터넷 불필요 - 파이프라인 점검 용도

실행:
  python -m src.llm_eval --gt data/image_eval_human_gt.csv --out results/llm_results.csv
"""
import argparse
import base64
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.scoring import ITEM_KEYS, score_items
from src.utils import resolve_path, ensure_dir, load_env, get_run_out_path


MODEL_ID = "gpt-4o"


# ---------------------------------------------------------------
# 시스템 프롬프트 (한국어, 18 항목 정의 포함)
# ---------------------------------------------------------------
SYSTEM_PROMPT = """당신은 한국식 파스텔 그림일기 이미지의 품질 평가 전문가입니다.
이미지와 생성 프롬프트(image_prompt_base)를 받아, 아래 18개 항목을 각각
1 (통과), 0 (실패), null (해당 없음 / 판단 불가) 중 하나로 평가합니다.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[핵심 평가 철학 — 반드시 숙지]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
당신이 평가하는 기준은 "일러스트로서의 완성도"입니다. 단순히 "있다/없다"가 아니라,
"전문 일러스트 수준으로 잘 그려졌는가"를 판단합니다.

■ 1점: 해당 항목에서 결함이 전혀 없음. 전문 일러스트 수준.
■ 0점: 눈에 띄는 결함이 하나라도 있음. 어색함, 왜곡, 부자연스러움 포함.
■ null: 해당 없음 또는 판단 불가.

⚠️ 교정 기준 (calibration): 사람 평가자 데이터 기반
- 전체 이미지의 약 32%만 L5(우수), 약 19%는 L2(많이 미흡)
- 18/18 만점은 매우 드물어야 함. 대부분 이미지는 어딘가 미흡함이 있음.
- 거의 모든 항목에 1점을 주고 있다면 → 너무 관대한 평가. 다시 검토할 것.
- 아래 항목들은 특히 엄격하게 평가: face_clear, no_extra_legs,
  no_duplicate_face_parts, sharpness_ok, prompt_reflected, place_context_ok

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[등급 캘리브레이션 예시 — GT 평가자 실제 기준]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

▶ L2 예시 (총점 8~10점 / 많이 미흡):
  · 강아지 얼굴·꼬리와 인물 표현이 어색하고 완성도가 낮음
    → face_clear=0, human_face_clear=0
  · 사람·배경 강아지 얼굴이 흐리고, 텍스트가 직접 삽입됨
    → face_clear=0, no_text=0, sharpness_ok=0
  · 공원 배경은 표현되었으나 입력에 없던 개체(비둘기)가 전면에 추가되어 장면 해석을 바꿈
    → prompt_reflected=0, place_context_ok=0

▶ L3 예시 (총점 11~13점 / 미흡):
  · 핵심 자세는 반영되었으나, 사람 얼굴과 손 표현이 어색하고 배경이 지나치게 단순함
    → human_face_clear=0, sharpness_ok=0
  · 교감 장면은 맞지만 사람 손과 강아지 얼굴·발 표현이 어색함
    → face_clear=0, human_face_clear=0

▶ L4 예시 (총점 14~16점 / 보통):
  · 전반적으로 안정적이나, 화면이 2분할처럼 구성됨
    → no_multi_panel=0
  · 핵심 장면은 잘 반영되었으나, 배경 소품 형태가 모호하고 서사 디테일 부족
    → place_context_ok=0 또는 prompt_reflected=0

▶ L5 예시 (총점 17~18점 / 우수):
  · 장면 반영 우수, 스타일·색감 안정적, 감정 표현 자연스러움
    → 대부분 항목 1점, 치명 오류 없음

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[엄수 규칙]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. 출력은 오직 유효한 JSON. 마크다운 코드블록(```json) 금지.
2. null 우선 원칙
   - 강아지가 없으면(dog_visible=0) 강아지 종속 항목(face_clear,
     no_duplicate_face_parts, no_extra_legs, one_dog)은 모두 null.
   - 사람이 등장하지 않으면 human_prompt_consistency, human_face_clear 는 null.
   - 확신이 50% 이하면 0 이 아니라 null.
3. reason 은 한국어 1~2문장. 구체적 근거. "문제가 없다"면 어떤 디테일이 좋은지 명시.
4. 인종/성별/외모 추정은 절대 평가하지 않는다.
   - human_prompt_consistency 는 "프롬프트와의 충돌 여부"만 본다.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[항목 정의 — 엄격 기준 명시]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- dog_visible: 강아지가 이미지에 명확히 검출되는가
- face_clear: 강아지 얼굴이 왜곡/뭉개짐 없이 선명한가
    → 눈코입 비율이 조금이라도 어색하거나 뭉개지면 0
- no_duplicate_face_parts: 눈/코/귀가 중복 생성되지 않았는가
    → 눈이 3개이거나 귀가 비대칭으로 추가되면 0
- no_extra_legs: 다리 개수가 추가 생성되지 않았는가
    → 5개 이상이거나 위치가 부자연스러우면 0
- one_dog: 이미지 내 강아지가 정확히 1마리인가
- no_mirror: 반사/거울 이미지가 없는가
- no_multi_panel: 멀티패널/분할 화면이 아닌 단일 장면인가
    → 화면이 2개 이상 장면으로 나뉘어 보이면 0
- pastel_style: 한국-일본식 파스텔 그림책 스타일인가
- no_realistic_rendering: 사실적 렌더링(사진/3D)이 혼입되지 않았는가
- human_prompt_consistency: 사람 표현이 프롬프트와 충돌하지 않는가
- human_face_clear: 사람 얼굴이 식별 가능하게 표현되었는가
    → 얼굴이 흐리거나 이목구비가 불분명하면 0. 사람 없으면 null.
- prompt_reflected: image_prompt_base 의 핵심 장면·요소가 모두 반영되었는가
    → 프롬프트의 핵심 요소 중 하나라도 빠지거나 다른 개체로 대체됐으면 0
- brightness_ok: 밝기가 적절한가 (너무 어둡거나 너무 밝으면 0)
- sharpness_ok: 이미지 전반이 충분히 선명하고 디테일이 살아있는가
    → 전체적으로 흐릿하거나 디테일이 뭉개진 부분이 있으면 0
- pastel_color_ok: 색감이 파스텔 범위인가 (채도 과다 또는 탁한 색이면 0)
- time_palette_ok: 프롬프트의 낮/밤/저녁과 이미지 팔레트가 일치하는가
    → 낮 장면인데 어두운 팔레트, 또는 반대이면 0
- place_context_ok: 프롬프트의 장소 키워드와 이미지 배경이 정확히 일치하는가
    → "공원"인데 실내 배경, "목욕탕"인데 공원 배경이면 0
    → 배경이 지나치게 단순하거나 주요 소품이 없으면 0
- no_text: 이미지에 텍스트(글자)가 삽입되지 않았는가
    → 어떤 글자라도 보이면 0 (로고, 사인, 간판 포함)

[출력 형식 - 정확히 이 구조]
{
  "items": {
    "dog_visible":              {"value": 1, "reason": "..."},
    "face_clear":               {"value": 1, "reason": "..."},
    "no_duplicate_face_parts":  {"value": 1, "reason": "..."},
    "no_extra_legs":            {"value": 1, "reason": "..."},
    "one_dog":                  {"value": 1, "reason": "..."},
    "no_mirror":                {"value": 1, "reason": "..."},
    "no_multi_panel":           {"value": 1, "reason": "..."},
    "pastel_style":             {"value": 1, "reason": "..."},
    "no_realistic_rendering":   {"value": 1, "reason": "..."},
    "human_prompt_consistency": {"value": null, "reason": "..."},
    "human_face_clear":         {"value": null, "reason": "..."},
    "prompt_reflected":         {"value": 1, "reason": "..."},
    "brightness_ok":            {"value": 1, "reason": "..."},
    "sharpness_ok":             {"value": 1, "reason": "..."},
    "pastel_color_ok":          {"value": 1, "reason": "..."},
    "time_palette_ok":          {"value": 1, "reason": "..."},
    "place_context_ok":         {"value": 1, "reason": "..."},
    "no_text":                  {"value": 1, "reason": "..."}
  }
}
"""


# ---------------------------------------------------------------
# Mock 모드
# ---------------------------------------------------------------
def _seed_from_path(image_path: str) -> int:
    return int(hashlib.md5(str(image_path).encode("utf-8")).hexdigest()[:8], 16)


def mock_evaluate(image_path: str, prompt: str) -> Tuple[Dict, Dict]:
    """API 안 쓰고 의사 난수로 18 항목 채움. 통과 70% / 실패 20% / null 10%."""
    rng = random.Random(_seed_from_path(image_path))
    items: Dict[str, Optional[int]] = {}
    reasons: Dict[str, str] = {}
    for k in ITEM_KEYS:
        r = rng.random()
        if r < 0.70:
            items[k] = 1
        elif r < 0.90:
            items[k] = 0
        else:
            items[k] = None
        reasons[k] = "[MOCK]"
    items["human_prompt_consistency"] = None
    items["human_face_clear"] = None
    return items, reasons


# ---------------------------------------------------------------
# Real LLM 모드
# ---------------------------------------------------------------
def _encode_image(image_path: str) -> Tuple[str, str]:
    p = resolve_path(image_path)
    ext = p.suffix.lower()
    media_type = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png",  ".gif": "image/gif", ".webp": "image/webp",
    }.get(ext, "image/png")
    with open(p, "rb") as f:
        data = base64.standard_b64encode(f.read()).decode("ascii")
    return data, media_type


def _parse_response(raw: str) -> Tuple[Dict, Dict]:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip("` \n")
    obj = json.loads(text)
    items_raw = obj.get("items", {})

    items: Dict[str, Optional[int]] = {}
    reasons: Dict[str, str] = {}
    for k in ITEM_KEYS:
        entry = items_raw.get(k, {}) or {}
        v = entry.get("value", None)
        if v in (1, 1.0, "1", True):
            items[k] = 1
        elif v in (0, 0.0, "0", False):
            items[k] = 0
        else:
            items[k] = None
        reasons[k] = str(entry.get("reason", ""))[:300]
    return items, reasons


def real_evaluate(image_path: str, prompt: str, client, model: str = MODEL_ID,
                  max_retries: int = 2) -> Tuple[Dict, Dict, int, int]:
    data, media_type = _encode_image(image_path)
    user_content = [
        {"type": "image_url",
         "image_url": {"url": f"data:{media_type};base64,{data}"}},
        {"type": "text",
         "text": f'image_prompt_base: "{prompt}"\n\n위 항목 18개를 JSON으로 평가하세요.'},
    ]
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                max_tokens=1500,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
            )
            raw = resp.choices[0].message.content
            items, reasons = _parse_response(raw)
            return items, reasons, resp.usage.prompt_tokens, resp.usage.completion_tokens
        except Exception as e:
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"LLM 평가 실패: {last_err}")


# ---------------------------------------------------------------
# CSV 컬럼 헬퍼
# ---------------------------------------------------------------
def _get_image_id(row: pd.Series) -> str:
    """image_id 컬럼 우선, 없으면 session_id 사용."""
    for col in ("image_id", "session_id"):
        v = row.get(col)
        if v is not None and str(v).strip() not in ("", "nan"):
            return str(v).strip()
    return ""


def _get_image_path(row: pd.Series) -> str:
    """
    image_path 컬럼 우선 → image 컬럼 → data/eval_images/{session_id}.png 순으로 결정.
    """
    v = row.get("image_path")
    if v is not None and str(v).strip() not in ("", "nan"):
        return str(v).strip()
    v = row.get("image")
    if v is not None and str(v).strip() not in ("", "nan"):
        return f"data/eval_images/{str(v).strip()}"
    sid = _get_image_id(row)
    return f"data/eval_images/{sid}.png"


def _get_prompt(row: pd.Series) -> str:
    """image_prompt_base 컬럼 우선, 없으면 diary_summary → diary_content 순으로 사용."""
    for col in ("image_prompt_base", "diary_summary", "diary_content"):
        v = row.get(col)
        if v is not None and str(v).strip() not in ("", "nan"):
            return str(v).strip()
    return ""


# ---------------------------------------------------------------
# 데이터셋 평가
# ---------------------------------------------------------------
def evaluate_dataset(gt_csv: str, out_csv: str = "auto", model: str = MODEL_ID,
                     limit: Optional[int] = None, force_mock: bool = False) -> None:
    load_env()
    gt_path = resolve_path(gt_csv)
    df = pd.read_csv(gt_path)

    # image_generated 컬럼이 있으면 실제 이미지가 있는 행만 처리
    if "image_generated" in df.columns:
        before = len(df)
        df = df[df["image_generated"].astype(str).str.upper() == "Y"].copy()
        print(f"[LLM] image_generated=Y 필터: {before} → {len(df)}행")

    if limit:
        df = df.head(limit)

    # 출력 경로 결정 (auto → 차수 자동 부여)
    if out_csv == "auto":
        out_path = get_run_out_path("llm", len(df))
    else:
        out_path = ensure_dir(out_csv)

    use_mock = force_mock or (os.environ.get("LLM_MOCK_MODE", "").lower() == "true") \
               or not os.environ.get("OPENAI_API_KEY")

    if use_mock:
        print("[LLM] *** MOCK MODE *** (no API call, no cost)")
        client = None
    else:
        print(f"[LLM] REAL MODE - model={model}")
        from openai import OpenAI
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    rows = []
    total_in, total_out = 0, 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="LLM eval"):
        image_id = _get_image_id(row)
        rec = {"image_id": image_id}
        try:
            img_path = _get_image_path(row)
            prompt   = _get_prompt(row)
            if use_mock:
                items, reasons = mock_evaluate(img_path, prompt)
                in_tok, out_tok = 0, 0
            else:
                items, reasons, in_tok, out_tok = real_evaluate(
                    img_path, prompt, client, model
                )
            scored = score_items(items)
            rec.update(items)
            for k, r in reasons.items():
                rec[f"reason_{k}"] = r
            rec["total_score"]  = scored["total_score"]
            rec["raw_level"]    = scored["raw_level"]
            rec["caps_applied"] = scored["caps_applied"]
            rec["final_level"]  = scored["final_level"]
            rec["input_tokens"]  = in_tok
            rec["output_tokens"] = out_tok
            total_in  += in_tok
            total_out += out_tok
        except Exception as e:
            rec["error"] = str(e)
        rows.append(rec)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[LLM] saved -> {out_path}  (n={len(out_df)})")
    if not use_mock and total_in + total_out > 0:
        # gpt-4o: $2.50/$10 per MTok
        cost = total_in / 1_000_000 * 2.5 + total_out / 1_000_000 * 10.0
        print(f"[LLM] tokens: in={total_in:,}  out={total_out:,}  est ${cost:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt",    default="data/image_eval_human_gt.csv")
    parser.add_argument("--out",   default="auto",
                        help="저장 경로. 기본값 'auto' 는 results/llm/n차_m장/ 에 자동 저장")
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--mock",  action="store_true", help="강제로 mock 모드 사용")
    args = parser.parse_args()
    evaluate_dataset(args.gt, args.out, args.model, args.limit, args.mock)
