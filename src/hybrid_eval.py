"""
src/hybrid_eval.py

LLM + CV 하이브리드 평가기.

[분배 전략 — 시뮬레이션 근거]
  139장 실측 데이터로 CV7 / LLM4 / 하이브리드 조합을 비교한 결과:

  139장 실측 결과 (3차 실험, gpt-4o):

  | 전략              | Exact | ±1    | MAE   | 편향   | 비용    |
  |-------------------|-------|-------|-------|--------|---------|
  | CV7 (16항목)      | 28.1% | 79.1% | 0.986 | -0.18  | $0      |
  | LLM4 (18항목)     | 34.5% | 71.2% | 1.079 | +0.68  | ~$1.58  |
  | Hybrid (이 파일)  | 37.4% | 72.7% | 1.065 | +0.53  | $1.10   |

  Hybrid는 정확 일치율(37.4%)에서 CV7(28.1%)·LLM(34.5%)을 모두 앞서며,
  ±1 정확도(72.7%)는 LLM(71.2%)보다 높다.
  출력 토큰 -30.2% 절감 (LLM-only 실측 563 tok → Hybrid 393 tok).

[CV 담당 항목 — 6개 (객관적 측정값)]
  dog_visible      CLIP zero-shot + 실측 임계값 (96.4% 검출률)
  brightness_ok    OpenCV 평균 밝기
  sharpness_ok     OpenCV Laplacian 분산  ← GT 상관계수 최고, LLM은 전부 1로 과관대
  pastel_color_ok  OpenCV HSV 채도
  no_multi_panel   Hough Line 패널 분리 감지
  no_text          EasyOCR 글자 감지

[LLM 담당 항목 — 12개 (주관적 판단)]
  face_clear, no_duplicate_face_parts, no_extra_legs,
  one_dog, no_mirror, pastel_style, no_realistic_rendering,
  human_prompt_consistency, human_face_clear,
  prompt_reflected, time_palette_ok, place_context_ok

[최적화]
  dog_visible = 0 → LLM API 호출 생략 (즉시 L0 반환, 비용 절감)
  LLM 프롬프트는 12개 항목만 요청 (입력 토큰·응답 시간 절감)

실행:
  python -m src.hybrid_eval --gt data/image_eval_human_gt.csv
"""
import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.scoring import ITEM_KEYS, score_items
from src.cv_eval import CVEvaluator
from src.utils import resolve_path, ensure_dir, load_env, get_run_out_path

# ── 항목 분배 ────────────────────────────────────────────────────────────────

CV_ITEMS: frozenset = frozenset([
    "dog_visible",
    "brightness_ok",
    "sharpness_ok",
    "pastel_color_ok",
    "no_multi_panel",
    "no_text",
])

LLM_ITEMS: frozenset = frozenset(k for k in ITEM_KEYS if k not in CV_ITEMS)

# ── LLM 시스템 프롬프트 (12개 항목만 요청) ──────────────────────────────────

_LLM_SYSTEM_PROMPT = """당신은 한국식 파스텔 그림일기 이미지의 품질 평가 전문가입니다.
이미지와 생성 프롬프트(image_prompt_base)를 받아, 아래 12개 항목만 평가합니다.
각 항목을 1 (통과), 0 (실패), null (해당 없음 / 판단 불가) 중 하나로 평가합니다.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[핵심 평가 철학]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"일러스트로서의 완성도"를 판단합니다.
■ 1점: 결함 없음. 전문 일러스트 수준.
■ 0점: 눈에 띄는 결함 하나라도 있음.
■ null: 해당 없음 또는 판단 불가.

⚠️ 교정 기준: 거의 모든 항목에 1점이라면 너무 관대한 평가. 재검토할 것.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[평가 항목 정의 — 12개]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- face_clear: 강아지 얼굴이 왜곡/뭉개짐 없이 선명한가
    → 눈코입이 조금이라도 어색하거나 뭉개지면 0
- no_duplicate_face_parts: 눈/코/귀가 중복 생성되지 않았는가
    → 눈이 3개이거나 귀가 비대칭 추가면 0
- no_extra_legs: 다리가 추가 생성되지 않았는가
    → 5개 이상이거나 위치 부자연스러우면 0
- one_dog: 이미지 내 강아지가 정확히 1마리인가
- no_mirror: 반사/거울 이미지가 없는가
- pastel_style: 한국-일본식 파스텔 그림책 스타일인가
- no_realistic_rendering: 사실적 렌더링(사진/3D)이 혼입되지 않았는가
- human_prompt_consistency: 사람 표현이 프롬프트와 충돌하지 않는가
    → 사람이 등장하지 않으면 null
- human_face_clear: 사람 얼굴이 식별 가능하게 표현되었는가
    → 얼굴이 흐리거나 이목구비 불분명하면 0. 사람 없으면 null.
- prompt_reflected: image_prompt_base의 핵심 장면·요소가 반영되었는가
    → 핵심 요소 하나라도 빠지거나 다른 개체로 대체됐으면 0
- time_palette_ok: 프롬프트의 낮/밤/저녁과 이미지 팔레트가 일치하는가
    → 낮인데 어두운 팔레트, 또는 반대면 0
- place_context_ok: 프롬프트의 장소·배경과 이미지 배경이 정확히 일치하는가
    → 배경이 지나치게 단순하거나 주요 소품이 없으면 0

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[엄수 규칙]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. 출력은 오직 유효한 JSON. 마크다운 코드블록(```json) 금지.
2. 사람이 없으면 human_prompt_consistency, human_face_clear 는 null.
3. reason 은 한국어 1~2문장. 구체적 근거.

[출력 형식]
{
  "items": {
    "face_clear":               {"value": 1, "reason": "..."},
    "no_duplicate_face_parts":  {"value": 1, "reason": "..."},
    "no_extra_legs":            {"value": 1, "reason": "..."},
    "one_dog":                  {"value": 1, "reason": "..."},
    "no_mirror":                {"value": 1, "reason": "..."},
    "pastel_style":             {"value": 1, "reason": "..."},
    "no_realistic_rendering":   {"value": 1, "reason": "..."},
    "human_prompt_consistency": {"value": null, "reason": "..."},
    "human_face_clear":         {"value": null, "reason": "..."},
    "prompt_reflected":         {"value": 1, "reason": "..."},
    "time_palette_ok":          {"value": 1, "reason": "..."},
    "place_context_ok":         {"value": 1, "reason": "..."}
  }
}
"""


# ── LLM 호출 ─────────────────────────────────────────────────────────────────

def _encode_image(image_path: str) -> Tuple[str, str]:
    p = resolve_path(image_path)
    ext = p.suffix.lower()
    media_type = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png", ".webp": "image/webp",
    }.get(ext, "image/png")
    with open(p, "rb") as f:
        data = base64.standard_b64encode(f.read()).decode("ascii")
    return data, media_type


def _parse_llm_response(raw: str) -> Tuple[Dict, Dict]:
    """LLM 응답 JSON → (items, reasons). 12개 LLM_ITEMS만 파싱."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip("` \n")
    obj = json.loads(text)
    items_raw = obj.get("items", {})
    if not items_raw:
        raise ValueError(f"LLM 응답에 items 키 없음: {text[:200]}")

    items: Dict[str, Optional[int]] = {}
    reasons: Dict[str, str] = {}
    for k in LLM_ITEMS:
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


def _llm_evaluate_subjective(image_path: str, prompt: str, client,
                              model: str = "gpt-4o",
                              max_retries: int = 2) -> Tuple[Dict, Dict, int, int]:
    """LLM으로 12개 주관적 항목만 평가."""
    data, media_type = _encode_image(image_path)
    user_content = [
        {"type": "image_url",
         "image_url": {"url": f"data:{media_type};base64,{data}"}},
        {"type": "text",
         "text": f'image_prompt_base: "{prompt}"\n\n위 12개 항목을 JSON으로 평가하세요.'},
    ]
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                max_tokens=900,
                messages=[
                    {"role": "system", "content": _LLM_SYSTEM_PROMPT},
                    {"role": "user",   "content": user_content},
                ],
            )
            raw = resp.choices[0].message.content
            in_tok  = resp.usage.prompt_tokens
            out_tok = resp.usage.completion_tokens
            # 파싱 실패는 재시도하지 않음 (이미 API 비용이 청구됨)
            items, reasons = _parse_llm_response(raw)
            return items, reasons, in_tok, out_tok
        except ValueError:
            raise  # JSON 파싱·items 키 오류 → 재시도 없이 상위로 전파
        except Exception as e:
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"LLM API 호출 실패 ({max_retries}회 재시도): {last_err}")


# ── 하이브리드 평가기 ─────────────────────────────────────────────────────────

class HybridEvaluator:
    """
    CV(객관) + LLM(주관) 하이브리드 평가기.

    사용 예시:
        ev = HybridEvaluator()
        result = ev.evaluate("data/eval_images/xxx.png", "공원에서 산책하는 흰 강아지")
        print(result["final_level"], result["total_score"])
    """

    def __init__(self, thresholds: Optional[dict] = None,
                 use_gpu: bool = False,
                 llm_model: str = "gpt-4o"):
        self._cv = CVEvaluator(thresholds=thresholds, use_gpu=use_gpu)
        self._llm_model = llm_model
        self._llm_client = None

    def _get_llm_client(self):
        if self._llm_client is None:
            load_env()
            from openai import OpenAI
            self._llm_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        return self._llm_client

    def evaluate(self, image_path: str, prompt: str = "") -> Dict:
        """
        이미지 1장 평가.

        Returns:
            items         : 18개 항목 dict (1/0/None)
            total_score   : 합산 점수
            raw_level     : 캡 적용 전 등급
            caps_applied  : 적용된 강등 규칙
            final_level   : 최종 등급
            source        : 항목별 평가 출처 ('cv' / 'llm')
            cv_scores     : CV 중간 측정값 (CLIP score 등)
            llm_reasons   : LLM 판단 근거 (주관 항목)
            in_tokens     : LLM 입력 토큰 (dog 미검출 시 0)
            out_tokens    : LLM 출력 토큰
        """
        # ── Step 1: CV 평가 ──────────────────────────────────────────────────
        cv_result = self._cv.evaluate(image_path, prompt)
        cv_items  = cv_result["items"]

        items: Dict[str, Optional[int]] = {k: None for k in ITEM_KEYS}

        # CV 담당 항목 기록
        for k in CV_ITEMS:
            items[k] = cv_items.get(k)

        in_tok = out_tok = 0
        llm_reasons: Dict[str, str] = {}

        # ── Step 2: LLM 평가 (강아지 미검출이면 생략) ───────────────────────
        if cv_items.get("dog_visible") == 0:
            # 강아지 없음 → LLM 항목은 모두 None
            # dog 종속 항목은 cap 규칙으로 자동 L0 처리됨
            llm_reasons = {k: "[LLM SKIPPED: dog not visible]" for k in LLM_ITEMS}
        else:
            client = self._get_llm_client()
            llm_items, llm_reasons, in_tok, out_tok = _llm_evaluate_subjective(
                image_path, prompt, client, self._llm_model
            )
            for k in LLM_ITEMS:
                items[k] = llm_items.get(k)

        # ── Step 3: 점수 계산 ────────────────────────────────────────────────
        scored = score_items(items)

        source = {k: ("cv" if k in CV_ITEMS else "llm") for k in ITEM_KEYS}

        cv_scores = {
            "clip_dog_score":      cv_result.get("clip_dog_score"),
            "clip_face_score":     cv_result.get("clip_face_score"),
            "clip_pastel_score":   cv_result.get("clip_pastel_score"),
            "clip_activity_score": cv_result.get("clip_activity_score"),
            "clip_place_score":    cv_result.get("clip_place_score"),
            "yolo_dog_count":      cv_result.get("yolo_dog_count"),
        }

        return {
            "items":        items,
            "total_score":  scored["total_score"],
            "raw_level":    scored["raw_level"],
            "caps_applied": scored["caps_applied"],
            "final_level":  scored["final_level"],
            "source":       source,
            "cv_scores":    cv_scores,
            "llm_reasons":  llm_reasons,
            "in_tokens":    in_tok,
            "out_tokens":   out_tok,
        }


# ── 데이터셋 일괄 평가 ────────────────────────────────────────────────────────

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


def _save_run_meta(out_path: Path, model: str, n: int, ev: "HybridEvaluator") -> None:
    """실험 재현에 필요한 메타 정보를 run_meta.json으로 저장."""
    import subprocess, datetime
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        git_hash = "unknown"

    meta = {
        "run_at":    datetime.datetime.now().isoformat(timespec="seconds"),
        "git_hash":  git_hash,
        "model":     model,
        "n_images":  n,
        "cv_items":  sorted(CV_ITEMS),
        "llm_items": sorted(LLM_ITEMS),
        "cv_thresholds": ev._cv.th,
    }
    meta_path = out_path.parent / "run_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[Hybrid] meta → {meta_path}")


def evaluate_dataset(gt_csv: str, out_csv: str = "auto",
                     model: str = "gpt-4o",
                     limit: Optional[int] = None,
                     checkpoint_every: int = 20) -> None:
    load_env()
    gt_path = resolve_path(gt_csv)
    df = pd.read_csv(gt_path)

    if "image_generated" in df.columns:
        before = len(df)
        df = df[df["image_generated"].astype(str).str.upper() == "Y"].copy()
        print(f"[Hybrid] image_generated=Y 필터: {before} → {len(df)}행")

    if limit:
        df = df.head(limit)

    if out_csv == "auto":
        out_path = get_run_out_path("hybrid", len(df))
    else:
        out_path = ensure_dir(out_csv)

    ev = HybridEvaluator(llm_model=model)
    _save_run_meta(out_path, model, len(df), ev)

    rows = []
    total_in = total_out = 0

    for i, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="Hybrid eval")):
        image_id = _get_image_id(row)
        rec = {"image_id": image_id}
        try:
            img_path = _get_image_path(row)
            prompt   = _get_prompt(row)
            res = ev.evaluate(img_path, prompt)

            rec.update(res["items"])
            rec["total_score"]  = res["total_score"]
            rec["raw_level"]    = res["raw_level"]
            rec["caps_applied"] = json.dumps(res["caps_applied"], ensure_ascii=False)
            rec["final_level"]  = res["final_level"]

            for k, v in res["cv_scores"].items():
                rec[k] = v
            for k, v in res["llm_reasons"].items():
                rec[f"reason_{k}"] = v

            rec["in_tokens"]  = res["in_tokens"]
            rec["out_tokens"] = res["out_tokens"]
            total_in  += res["in_tokens"]
            total_out += res["out_tokens"]
        except Exception as e:
            rec["error"] = str(e)
        rows.append(rec)

        # 중간 저장 (API 중단 대비)
        if (i + 1) % checkpoint_every == 0:
            pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[Hybrid] saved → {out_path}  (n={len(out_df)})")

    # dog_visible=0 스킵 통계
    skipped = sum(1 for r in rows if r.get("dog_visible") == 0)
    evaluated = len(rows) - skipped
    if skipped:
        print(f"[Hybrid] dog_visible=0 LLM skip: {skipped}/{len(rows)} ({skipped/len(rows)*100:.1f}%)")

    if total_in + total_out > 0:
        # 모델별 단가 테이블 ($/MTok: input, output)
        _PRICING = {
            "gpt-4o":           (2.50, 10.00),
            "gpt-4o-mini":      (0.15,  0.60),
            "gpt-4o-2024-11-20":(2.50, 10.00),
        }
        rates = _PRICING.get(model)
        avg_out = total_out / evaluated if evaluated > 0 else 0
        print(f"[Hybrid] tokens: in={total_in:,}  out={total_out:,}")
        if rates:
            cost = total_in / 1e6 * rates[0] + total_out / 1e6 * rates[1]
            print(f"[Hybrid] est ${cost:.4f} ({model})")
        else:
            print(f"[Hybrid] 비용 미산출 — 미등록 모델: {model}")
        # LLM 4차 실측 baseline (18항목, 139장 평균 output=563.2 tok)
        _LLM_BASELINE_OUT = 563.2
        if avg_out > 0:
            reduction = (1 - avg_out / _LLM_BASELINE_OUT) * 100
            print(f"[Hybrid] 출력 토큰 평균: {avg_out:.0f} tok "
                  f"(LLM-only 실측 {_LLM_BASELINE_OUT:.0f} tok 대비 -{reduction:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt",    default="data/image_eval_human_gt.csv")
    parser.add_argument("--out",   default="auto")
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    evaluate_dataset(args.gt, args.out, args.model, args.limit)
