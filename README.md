# withDOG 이미지 품질 자동 평가 시스템

> **강아지 그림일기 생성 이미지의 품질을 자동으로 평가하는 시스템.**  
> LLM 기반 평가기와 CV 기반 평가기를 병행 운영하여 인간 평가자(GT) 대비 정렬 성능을 측정·개선한다.

---

## 목차
1. [프로젝트 소개](#1-프로젝트-소개)
2. [목표](#2-목표)
3. [데이터셋](#3-데이터셋)
4. [CV 평가기 개발 과정](#4-cv-평가기-개발-과정)
5. [LLM 평가기 개발 과정](#5-llm-평가기-개발-과정)
6. [LLM vs CV 비교 분석](#6-llm-vs-cv-비교-분석)
7. [운영 결정](#7-운영-결정)
8. [향후 개선 방향](#8-향후-개선-방향)
9. [결과 파일 구조](#9-결과-파일-구조)
10. [실행 방법](#10-실행-방법)

---

## 1. 프로젝트 소개

### 배경

withDOG 서비스는 반려견 그림일기를 AI 이미지 생성 모델로 제작한다. 생성 이미지의 품질 평가는 현재 인간 평가자가 수동으로 수행하며, 이미지당 평가 비용과 시간이 고정적으로 발생한다. 평가 항목은 강아지 피사체 품질, 스타일, 구도, 배경, 프롬프트 반영도 등 18개 항목으로 구성되며, 항목별 통과(1) / 실패(0) / 해당 없음(null) 판정을 종합하여 최종 등급을 산출한다.

### 솔루션

본 프로젝트는 두 가지 자동 평가기를 개발·비교한다.

- **LLM 평가기** (`src/llm_eval.py`): GPT-4o Vision API로 이미지를 분석하여 18개 항목을 판정
- **CV 평가기** (`src/cv_eval.py`): CLIP zero-shot + YOLO + OpenCV + EasyOCR 조합으로 API 비용 없이 로컬 평가

---

## 2. 목표

| 목표 | 기준 |
|---|---|
| 정확도 | GT 평가자 대비 ±1 within 70% 이상 |
| 비용 | CV 평가기 단독 운영 시 API 비용 0 |
| 호환성 | 18개 item, 6단계 등급(L0~L5), 18점 만점 체계 지원 |

### 평가 체계

| 총점 범위 | 등급 | 의미 |
|---|---|---|
| 0~2 | L0 | 생성 실패 (강아지 미검출) |
| 3~6 | L1 | 매우 미흡 |
| 7~10 | L2 | 미흡 |
| 11~13 | L3 | 보통 |
| 14~16 | L4 | 양호 |
| 17~18 | L5 | 우수 |

**치명 오류 캡(Cap) 규칙**: 특정 항목이 실패(0)이면 최대 등급이 강제 하향된다.

| 항목 | 캡 등급 |
|---|---|
| `dog_visible = 0` | L0 |
| `one_dog = 0`, `no_multi_panel = 0` | L1 |
| `face_clear = 0`, `no_extra_legs = 0`, `no_duplicate_face_parts = 0` | L2 |
| `no_text = 0`, `prompt_reflected = 0` | L3 |

---

## 3. 데이터셋

- **전체 GT**: 171장 (`data/image_eval_human_gt.csv`)
- **평가 대상**: 139장 (`image_generated = Y` 필터; 32장은 생성 실패로 제외)

### GT 등급 분포

| 등급 | 장수 | 비율 |
|---|---|---|
| L1 | 6 | 4.3% |
| L2 | 26 | 18.7% |
| L3 | 21 | 15.1% |
| L4 | 42 | 30.2% |
| L5 | 44 | 31.7% |
| **합계** | **139** | **100%** |

GT 총점 평균 **13.81점** (중앙값 15.0점, 18점 만점).

*(출처: `data/image_eval_human_gt.csv`)*

### 시간순 품질 편향

GT는 수집 시기에 따라 뚜렷한 품질 차이를 보인다.

| 구간 | 장수 | GT 평균 레벨 | 날짜 범위 |
|---|---|---|---|
| 초기 | 80장 | **2.92** (L2~L3 경계) | 2026-04-15 ~ 2026-04-20 |
| 후반 | 59장 | **4.66** (L4~L5 경계) | 2026-04-20 ~ 2026-04-23 |

초기 이미지는 생성 모델 초기 버전 결과물이며, 후반 이미지는 개선된 버전에서 생성됐다. 이 편향은 평가기 성능 해석 시 고려해야 한다.

---

## 4. CV 평가기 개발 과정

### 4-1. 진화 단계

| 차수 | 주요 방식 | 결과 | 결정 |
|---|---|---|---|
| 1차 (`1차_10장`) | YOLO conf=0.35 | 10장 전량 `dog_visible=0` — 카툰·파스텔 스타일 미인식 | 폐기 |
| 2차 (`2차_10장`) | YOLO conf=0.10 | 동일 (카툰 이미지에 YOLO 구조적 한계) | 폐기 |
| 3차 (`3차_10장`) | CLIP softmax (9개 텍스트) | clip_dog_score 범위 0.562~0.568, 무작위 기준(5/9=0.556)과 동일 수준 — 변별력 없음 | 폐기 |
| 4차 (`4차_10장`) | CLIP cosine diff, 7항목 | dog_visible 작동, 10장 평가 성공 | 채택 |
| 5차 (`5차_139장`) | 4차와 동일, 139장 baseline | exact=6.5%, ±1=23.0%, MAE=2.568 | baseline |
| **6차 (`6차_139장`)** | CLIP 4개 항목 추가, 총 11항목 | **exact=15.1%, ±1=37.4%, MAE=1.820** | **Final** |

*(출처: `results/cv/5차_139장/cv_results.csv`, `results/cv/6차_139장/cv_results.csv`, `results/compare/2차_139장/comparison_result.csv`)*

### 4-2. CLIP 접근법 전환 배경

YOLO (YOLOv8n)는 사실적 사진 기반으로 학습된 모델로, 한국식 파스텔 일러스트 스타일의 강아지를 인식하지 못한다. CLIP (ViT-B-32, OpenAI pretrained)의 zero-shot 방식으로 전환 후 성능이 회복됐다.

**Cosine diff 방식 채택 근거**: softmax 대신 `pos_mean − neg_mean` (코사인 유사도 차이)를 직접 비교하면 무작위 기준값이 0.0으로 명확해지고 변별력이 높아진다. softmax 방식에서는 9개 텍스트 중 무작위 기준이 1/9=0.111이어서 실질적 변별 신호가 희석된다.

### 4-3. dog_visible threshold 결정 (FP 테스트)

비강아지 이미지 33장을 대상으로 threshold별 FP율을 측정했다.

| threshold | TP율 (139장) | FP율 (33장) | 결정 |
|---|---|---|---|
| 0.010 | 99.3% (138/139) | 18.2% (6/33) | FP 과다 |
| **0.020** | **96.4% (134/139)** | **9.1% (3/33)** | **채택** |

*(출처: `src/cv_eval.py` threshold 결정 근거 주석, FP 테스트 대상: `data/FP_test_image/` 33장)*

### 4-4. CLIP 검출 미달 5건 분석

| 견종 | clip_dog_score | 비고 |
|---|---|---|
| 그레이하운드 | −0.002 | 극도로 가는 체형 — CLIP ViT-B-32 구조적 한계 |
| 달마시안 | +0.015 | threshold 미달 |
| 골든레트리버 | +0.018 | threshold 미달 |
| 삽살개 | +0.019 | threshold 미달 |
| 말라뮤트 | +0.020 | threshold 경계 |

### 4-5. 6차 평가기 구현 항목 (11개)

| 항목 | 방식 | GT 상관 (Pearson r) | 비고 |
|---|---|---|---|
| `dog_visible` | CLIP cosine diff | r=+0.228, p=0.007 | 유의미 |
| `one_dog` | YOLO count (CLIP 감지 시에만) | — | 110장 None (YOLO 미검출) |
| `face_clear` | CLIP cosine diff | r=+0.106, p=0.221 | 비유의 |
| `brightness_ok` | OpenCV mean gray | — | — |
| `sharpness_ok` | OpenCV Laplacian variance | r=+0.233, p=0.006 | **유의미** |
| `pastel_color_ok` | OpenCV HSV saturation | 분산=0 | 139장 전부 통과 |
| `pastel_style` | CLIP cosine diff | 분산=0 | 139장 전부 통과 |
| `no_realistic_rendering` | CLIP cosine diff (pastel_style 동일 신호) | 분산=0 | 139장 전부 통과 |
| `no_multi_panel` | OpenCV Hough Lines | 분산=0 | 139장 전부 통과 |
| `no_text` | EasyOCR | — | 137장 통과, 2장 실패 |
| `place_context_ok` | CLIP + 장소 키워드 매칭 | r=+0.062, p=0.755 | n=28 (80% 키워드 없음) |

*(출처: `results/cv/6차_139장/cv_results.csv`, `scripts/analyze_cv_6th.py`)*

**결과 요약**: 6차에서 측정 항목이 7→11개로 확장되어 정확도 지표가 개선됐다. 그러나 `pastel_style`, `no_realistic_rendering`, `pastel_color_ok`, `no_multi_panel` 4개 항목은 139장 전체에서 분산=0으로, 실질적 변별 없이 점수를 균등하게 상승시키는 효과만 제공한다. GT와 통계적으로 유의미한 상관을 갖는 항목은 `sharpness_ok`와 `dog_visible` 연속값 2개에 한정된다.

---

## 5. LLM 평가기 개발 과정

GPT-4o Vision API를 사용하며, 이미지와 생성 프롬프트(`image_prompt_base`)를 입력받아 18개 항목을 판정한다. 판정 결과를 `scoring.py`에 전달하여 점수·등급을 산출한다.

### 5-1. 실험 이력

| 차수 | 대상 | 프롬프트 | Exact | ±1 within | 편향 |
|---|---|---|---|---|---|
| 1차 (`1차_10장`) | 10장 | 기본 | 50.0% | 90.0% | −0.700 |
| **2차 (`2차_139장`)** | **139장** | **기본** | **36.0%** | **73.4%** | **+0.705** |
| 3차 (`3차_10장`) | 10장 | 엄격 강화 | 40.0% | 90.0% | −0.800 |
| 4차 (`4차_139장`) | 139장 | 엄격 강화 | 34.5% | 71.2% | +0.676 |

*(출처: `results/llm/2차_139장/llm_results.csv`, `results/llm/4차_139장/llm_results.csv`, `results/llm/comparison_v2_v3_139.csv`)*

### 5-2. 프롬프트 엄격화 효과 (2차 → 4차)

| 지표 | 2차 (기본) | 4차 (엄격) | 변화 |
|---|---|---|---|
| Exact match | 36.0% | 34.5% | −1.5%p |
| ±1 within | 73.4% | 71.2% | −2.2%p |
| MAE | 1.050 | 1.079 | +0.029 |
| 편향 | +0.705 | +0.676 | −0.029 |
| 만점(18점) 비율 | 55% (76장) | 41% (57장) | −14%p |
| L5 판정 비율 | 68% (94장) | 67% (93장) | −1%p |

만점 비율은 14%p 감소했으나, 등급 임계값(L5: 총점 17점) 근처에서의 변화가 크지 않아 L5 판정 비율은 거의 변동 없다. 시스템 프롬프트 텍스트 강화만으로는 GPT-4o의 과대평가 편향을 교정하는 데 한계가 있다.

### 5-3. 시간순 분리 분석 (2차_139장 기준)

| 구간 | 장수 | Exact | ±1 within | 편향 |
|---|---|---|---|---|
| 초기 80장 (GT 평균 L2.92) | 80 | 23.8% | 57.5% | **+1.137** |
| 후반 59장 (GT 평균 L4.66) | 59 | **52.5%** | **94.9%** | **+0.119** |

LLM은 고품질 이미지(L4~L5)에서 GT와 거의 일치하나, 저품질 이미지(L1~L3) 영역에서 변별에 실패하고 과대평가하는 경향이 있다.

---

## 6. LLM vs CV 비교 분석

LLM 2차 (기본 프롬프트, 최고 성능) vs CV 6차 (최신 버전).

| 지표 | LLM (2차_139장) | CV (6차_139장) | 차이 |
|---|---|---|---|
| Exact match | **36.0%** | 15.1% | −20.9%p |
| ±1 within | **73.4%** | 37.4% | −36.0%p |
| MAE | **1.050** | 1.820 | +0.770 |
| 편향 | +0.705 (과대평가) | −1.705 (과소평가) | 반대 방향 |
| API 비용 | ~\$1.4 / 139장 | **0 (로컬)** | — |
| 처리 속도 | ~10~15분 / 139장 | **~5분 / 139장** | — |
| 평가 항목 수 | 18개 (전체) | 최대 11개 | −7개 |

*(출처: `results/compare/2차_139장/comparison_result.csv`, `results/cv/6차_139장/cv_results.csv`, `results/llm/2차_139장/llm_results.csv`)*

**CV 구조적 상한**: 18개 항목 중 최대 11개 평가, 그중 분산=0 항목 4개를 제외하면 실질 변별 항목 7개 이하. 최대 총점 11점 → 이론적 등급 상한 L3.

---

## 7. 운영 결정

| 역할 | 평가기 | 근거 |
|---|---|---|
| **본 평가기** | **LLM (GPT-4o)** | ±1 within 73.4%, 18개 항목 전체 커버 |
| **보조** | CV | Cap 메커니즘으로 명확한 실패 케이스 자동 감지 |

**CV 보조 활용 근거**: CV는 `dog_visible=0`(L0 강제), `one_dog=0` / `no_multi_panel=0`(L1 강제) 등 치명 오류를 낮은 비용으로 감지한다. LLM과 CV 판정이 크게 다를 경우 재검토 신호로 활용한다.

---

## 8. 향후 개선 방향

이하 3가지 작업은 현재 **미실행** 상태이며, 향후 phase 계획으로 분류한다.

### 8-1. 단기: 상수 항목 정리 (결정 보류)

`pastel_style`, `no_realistic_rendering`, `pastel_color_ok`, `no_multi_panel` 4개 항목이 139장 전체에서 분산=0이다. 데이터셋이 확장되거나 생성 모델이 다변화될 경우 재평가가 필요하다.

검토 중인 옵션:
- 코드는 유지하되 `non_discriminative` 플래그를 결과 CSV에 추가
- 유효 만점을 18점→14점으로 변경 (GT 호환성 손상 위험 존재)

현재 시점 기준으로 **결정 보류**. 데이터셋 확장 후 재평가 예정.

### 8-2. 중기: 기존 항목 변별력 향상

| 작업 | 대상 항목 | 방식 |
|---|---|---|
| sharpness 정밀화 | `sharpness_ok` | 단일 threshold → 연속값 점수, dog crop과 전체 이미지 별도 계산 예정 |
| face_clear 방식 전환 | `face_clear` | 현재 CLIP score 분포(−0.007~+0.036)가 노이즈 수준 → YOLO dog crop 후 OpenCV Laplacian variance로 전환 예정 |
| place_context_ok 활동 매칭 | `place_context_ok` | 현재 80% None(장소 키워드 없는 활동 중심 프롬프트) → 활동 키워드(목욕, 산책) 기반 장소 추론 매핑 추가 예정 |

예상 효과: ±1 within 37.4% → 45~50% (미검증 추정치).

### 8-3. 장기: Vision Model 추가

| 작업 | 추가 평가 가능 항목 | 비고 |
|---|---|---|
| 강아지 키포인트 검출 (AnimalPose 등) | `no_duplicate_face_parts`, `no_extra_legs` | 현재 CV 미평가 2개 항목 |
| Face detector (MediaPipe 등) | `human_face_clear` | 현재 CV 미평가 1개 항목 |
| CLIP ViT-L-14 업그레이드 | `dog_visible` 비전형 견종 5건 개선 | 모델 교체 필요 |
| Aesthetic Predictor (LAION 등) | 일러스트 완성도 종합 점수 | 추가 연구 필요 |

예상 효과: ±1 within 50% → 65~70% (LLM 73.4%에 근접, 미검증 추정치).  
**현재 프로젝트 범위 외.**

---

## 9. 결과 파일 구조

```
results/
├── cv/
│   ├── 1차_10장/      YOLO conf=0.35 (폐기)
│   ├── 2차_10장/      YOLO conf=0.10 (폐기)
│   ├── 3차_10장/      CLIP softmax (폐기)
│   ├── 4차_10장/      CLIP cosine diff, 7항목 (채택)
│   ├── 5차_139장/     7항목 baseline  exact=6.5%, ±1=23.0%
│   └── 6차_139장/ ★  11항목 Final    exact=15.1%, ±1=37.4%
├── llm/
│   ├── 1차_10장/      기본 프롬프트, 10장 테스트
│   ├── 2차_139장/ ★  기본 프롬프트   exact=36.0%, ±1=73.4%
│   ├── 3차_10장/      엄격 프롬프트, 10장 테스트
│   ├── 4차_139장/     엄격 프롬프트   exact=34.5%, ±1=71.2%
│   ├── comparison_v1_vs_v2.csv   10장 프롬프트 비교 결과
│   └── comparison_v2_v3_139.csv  139장 프롬프트 비교 결과
└── compare/
    ├── 1차_10장/      LLM vs CV 비교 (10장)
    └── 2차_139장/     LLM vs CV 공식 비교 리포트 (139장)
```

---

## 10. 실행 방법

### 환경 설정

```bash
python -m venv withdog.venv
withdog.venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

`.env` 파일에 OpenAI API 키를 설정한다:
```
OPENAI_API_KEY=sk-...
```

### CV 평가기 실행

```bash
# 139장 전체 평가 (자동 저장: results/cv/N차_139장/)
python -m src.cv_eval --gt data/image_eval_human_gt.csv

# 10장 테스트
python -m src.cv_eval --gt data/image_eval_human_gt.csv --limit 10

# 강아지 미인식 FP 테스트
python fp_test.py
```

### LLM 평가기 실행

```bash
# 139장 전체 평가 (자동 저장: results/llm/N차_139장/, 비용 ~$1.4)
python -m src.llm_eval --gt data/image_eval_human_gt.csv

# 10장 테스트 (비용 ~$0.12)
python -m src.llm_eval --gt data/image_eval_human_gt.csv --limit 10
```

### LLM vs CV 비교 분석

```bash
python -m src.compare_eval \
    --llm results/llm/2차_139장/llm_results.csv \
    --cv  results/cv/6차_139장/cv_results.csv \
    --gt  data/image_eval_human_gt.csv
```

---

## 주요 소스 파일

| 파일 | 역할 |
|---|---|
| `src/cv_eval.py` | CV 평가기 (CLIP + YOLO + OpenCV + EasyOCR) |
| `src/llm_eval.py` | LLM 평가기 (GPT-4o Vision API) |
| `src/scoring.py` | 18개 항목 → 총점·캡·등급 산출 공통 모듈 |
| `src/compare_eval.py` | LLM vs CV 비교 리포트 생성 |
| `src/utils.py` | 경로 해석, 이미지 로드, 유틸 함수 |
| `data/image_eval_human_gt.csv` | 인간 평가자 GT (171장, 7개 카테고리 점수 포함) |
| `models/yolov8n.pt` | YOLOv8n 가중치 (one_dog 카운트용) |
| `data/FP_test_image/` | 비강아지 이미지 33장 (FP 테스트용) |
| `scripts/` | 분석·캘리브레이션 스크립트 |
