"""
FP 테스트: fp_test_image/ 폴더의 비강아지 이미지에서 CLIP 오탐(FP) 확인.
- score >= threshold 이면 FP (강아지로 잘못 판정)
- score <  threshold 이면 TN (올바르게 비강아지 판정)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.cv_eval import CVEvaluator
from src.utils import safe_imread, PROJECT_ROOT

THRESHOLD = 0.01
IMAGE_FOLDER = PROJECT_ROOT / "data" / "FP_test_image"
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

ev = CVEvaluator()

images = [p for p in IMAGE_FOLDER.iterdir() if p.suffix.lower() in EXTENSIONS]
if not images:
    print(f"[!] {IMAGE_FOLDER} 에 이미지가 없습니다. jpg/png 파일을 넣어주세요.")
    sys.exit(0)

images.sort()
print(f"총 {len(images)}장 테스트 (threshold={THRESHOLD})\n")
print(f"{'파일명':<45} {'score':>8}  {'판정'}")
print("-" * 70)

fp_count = 0
for p in images:
    img = safe_imread(str(p))
    if img is None:
        print(f"  {p.name:<45} 로드실패")
        continue
    score = ev._clip_dog_score(img)
    is_fp = score >= THRESHOLD
    if is_fp:
        fp_count += 1
    label = "FP !! (강아지로 오탐)" if is_fp else "TN  OK"
    print(f"  {p.name:<45} {score:+.4f}  {label}")

print(f"\n결과: FP {fp_count}/{len(images)}  ({fp_count/len(images)*100:.1f}%)")
