"""
src/scoring.py

withDOG 그림일기 평가: 18개 item dict → (total_score, caps_applied, final_level)
LLM/CV/사람 모두 이 함수를 통해 동일하게 점수화한다 (일관성 보장).
"""

from typing import Dict, List, Optional, Tuple


ITEM_KEYS: List[str] = [
    "dog_visible",
    "face_clear",
    "no_duplicate_face_parts",
    "no_extra_legs",
    "one_dog",
    "no_mirror",
    "no_multi_panel",
    "pastel_style",
    "no_realistic_rendering",
    "human_prompt_consistency",
    "human_face_clear",
    "prompt_reflected",
    "brightness_ok",
    "sharpness_ok",
    "pastel_color_ok",
    "time_palette_ok",
    "place_context_ok",
    "no_text",
]

# 점수 → 레벨 (사양서 그대로)
LEVEL_RANGES: List[Tuple[str, int, int]] = [
    ("L0", 0, 2),
    ("L1", 3, 6),
    ("L2", 7, 10),
    ("L3", 11, 13),
    ("L4", 14, 16),
    ("L5", 17, 18),
]

# 치명 오류 → 최대 허용 레벨 (해당 item이 0일 때 cap)
CAP_RULES: List[Tuple[str, str]] = [
    ("dog_visible",            "L0"),
    ("one_dog",                "L1"),
    ("no_multi_panel",         "L1"),
    ("face_clear",             "L2"),
    ("no_extra_legs",          "L2"),
    ("no_duplicate_face_parts","L2"),
    ("no_text",                "L3"),
    ("prompt_reflected",       "L3"),
]

LEVEL_ORDER = {lvl: i for i, (lvl, _, _) in enumerate(LEVEL_RANGES)}


def compute_total_score(items: Dict[str, Optional[int]]) -> int:
    """1로 표시된 항목 개수. null 은 빼고 센다."""
    return sum(1 for k in ITEM_KEYS if items.get(k) == 1)


def score_to_level(score: int) -> str:
    for lvl, lo, hi in LEVEL_RANGES:
        if lo <= score <= hi:
            return lvl
    return "L0" if score < 0 else "L5"


def get_caps(items: Dict[str, Optional[int]]) -> List[Tuple[str, str]]:
    """item value == 0 일 때만 cap 트리거 (null은 트리거 X)."""
    return [(k, lvl) for k, lvl in CAP_RULES if items.get(k) == 0]


def apply_caps(level: str, caps: List[Tuple[str, str]]) -> str:
    if not caps:
        return level
    strictest = min((c[1] for c in caps), key=lambda l: LEVEL_ORDER[l])
    return strictest if LEVEL_ORDER[strictest] < LEVEL_ORDER[level] else level


def score_items(items: Dict[str, Optional[int]]) -> Dict:
    """입력: 18개 item dict → 출력: 점수/캡/레벨 dict."""
    missing = [k for k in ITEM_KEYS if k not in items]
    if missing:
        raise ValueError(f"Missing item keys: {missing}")
    total = compute_total_score(items)
    raw_lvl = score_to_level(total)
    caps = get_caps(items)
    final_lvl = apply_caps(raw_lvl, caps)
    caps_summary = ",".join(f"{k}->{lvl}" for k, lvl in caps) if caps else ""
    return {
        "total_score": total,
        "raw_level": raw_lvl,
        "caps_applied": caps_summary,
        "final_level": final_lvl,
    }


# ---- self-test ----
if __name__ == "__main__":
    ok = {k: 1 for k in ITEM_KEYS}
    ok["human_prompt_consistency"] = None
    ok["human_face_clear"] = None
    print("OK case  :", score_items(ok))
    no_dog = {k: 0 for k in ITEM_KEYS}
    no_dog["dog_visible"] = 0
    no_dog["brightness_ok"] = 1
    no_dog["sharpness_ok"] = 1
    print("No dog   :", score_items(no_dog))
    text_in = {k: 1 for k in ITEM_KEYS}
    text_in["no_text"] = 0
    text_in["human_prompt_consistency"] = None
    text_in["human_face_clear"] = None
    print("Text in  :", score_items(text_in))
