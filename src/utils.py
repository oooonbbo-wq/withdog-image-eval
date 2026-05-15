"""
src/utils.py

Windows 로컬 실행을 위한 공통 헬퍼.

핵심:
- safe_imread()   : Windows에서 한글/유니코드 경로 cv2.imread 실패 회피
- get_project_root(): 어디서 실행해도 프로젝트 루트 자동 인식
- load_env()      : .env 자동 로딩
"""
import os
import re
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


# src/utils.py 위치 기준: parent = src/, parent.parent = 프로젝트 루트
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_project_root() -> Path:
    return PROJECT_ROOT


def resolve_path(p: str | Path) -> Path:
    """상대경로면 프로젝트 루트 기준으로 절대경로화."""
    p = Path(p)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def safe_imread(path: str | Path) -> Optional[np.ndarray]:
    """
    Windows에서 한글/유니코드 경로의 cv2.imread 실패를 회피.
    np.fromfile + cv2.imdecode 사용.
    실패 시 None 반환.
    """
    try:
        p = resolve_path(path)
        if not p.exists():
            return None
        arr = np.fromfile(str(p), dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"[safe_imread] failed for {path}: {e}", file=sys.stderr)
        return None


def load_env() -> None:
    """프로젝트 루트의 .env 를 로드. dotenv 없어도 무난하게 패스."""
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv
        load_dotenv(env_path)
    except ImportError:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def ensure_dir(p: str | Path) -> Path:
    """경로의 부모 디렉토리를 생성하고 Path 반환."""
    p = resolve_path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def to_binary_or_none(v) -> Optional[int]:
    """1 / 0 / null 로 정규화. 'true'/'false', 1.0, '1', '' 등 흡수."""
    if v is None:
        return None
    if isinstance(v, float) and np.isnan(v):
        return None
    s = str(v).strip().lower()
    if s in ("", "nan", "none", "null", "na"):
        return None
    if s in ("1", "1.0", "true", "t", "yes", "y", "통과"):
        return 1
    if s in ("0", "0.0", "false", "f", "no", "n", "실패"):
        return 0
    return None


def get_run_out_path(result_type: str, n_images: int) -> Path:
    """
    results/{type}/n차_{count}장/{type}_results.csv 경로를 자동 생성.
    기존 폴더를 스캔해 다음 차수를 결정한다.
    예) results/cv/1차_10장/cv_results.csv → 다음은 results/cv/2차_30장/...
    """
    base = PROJECT_ROOT / "results" / result_type
    base.mkdir(parents=True, exist_ok=True)
    existing = [
        d for d in base.iterdir()
        if d.is_dir() and re.match(r'^\d+차_\d+장$', d.name)
    ]
    next_n = max(
        (int(re.match(r'^(\d+)차_', d.name).group(1)) for d in existing),
        default=0
    ) + 1
    folder = base / f"{next_n}차_{n_images}장"
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{result_type}_results.csv"


def get_latest_run_path(result_type: str, filename: str) -> Optional[Path]:
    """
    results/{type}/ 에서 차수가 가장 높은 폴더의 파일 경로를 반환.
    파일이 없거나 폴더가 없으면 None 반환.
    """
    base = PROJECT_ROOT / "results" / result_type
    if not base.exists():
        return None
    runs = [
        d for d in base.iterdir()
        if d.is_dir() and re.match(r'^\d+차_\d+장$', d.name)
    ]
    if not runs:
        return None
    latest = max(runs, key=lambda d: int(re.match(r'^(\d+)차_', d.name).group(1)))
    p = latest / filename
    return p if p.exists() else None


def parse_level_str(s) -> Optional[str]:
    """
    'Lv5: 우수', 'Lv0: 생성 실패', 'L4', 'L3' 등을 'L0'~'L5' 로 정규화.
    인식 불가면 None 반환.
    """
    if s is None:
        return None
    import pandas as pd
    if isinstance(s, float) and np.isnan(s):
        return None
    text = str(s).strip()
    # 'L4' 형태
    m = re.match(r'^L(\d)$', text, re.IGNORECASE)
    if m:
        return f"L{m.group(1)}"
    # 'Lv5: 우수' 형태
    m = re.match(r'^Lv(\d)', text, re.IGNORECASE)
    if m:
        return f"L{m.group(1)}"
    return None
