# Lotto Analysis / 로또 분석 프로젝트

## 한국어

### 프로젝트 개요

이 프로젝트는 한국 로또 6/45 데이터를 대상으로, notebook-first 방식으로 분석과 모델링을 수행합니다. 현재 범위는 다음을 포함합니다.

- 로또 원천 데이터 동기화 및 전처리
- 탐색적 데이터 분석 및 통계 검정
- 달력 정보 및 날씨 컨텍스트 분석
- 기본 모델과 확장 모델 비교
- 최근 구간 학습(recent-window) 검증 및 최종 후보 선정
- 확률 랭킹 분석과 티켓 생성 전략 비교
- 보고서용 figure / table 산출물 생성

### 현재 notebook 구성

현재 프로젝트는 `01`부터 `15`까지의 notebook 흐름을 포함합니다.

- `01_data_collection.ipynb`: 데이터 수집 개요
- `02_eda.ipynb`: 탐색적 시각화
- `03_statistical_tests.ipynb`: 무작위성 중심 통계 검정
- `04_feature_engineering.ipynb`: 시계열 기반 feature 생성
- `05_model_baseline.ipynb`: 기본 모델 학습 및 artifact 저장
- `06_model_evaluation.ipynb`: 평가 시각화 및 요약
- `07_contextual_analysis.ipynb`: 달력 컨텍스트 분석
- `08_weather_context_analysis.ipynb`: 날씨 컨텍스트 분석
- `09_weather_feature_modeling.ipynb`: 날씨 feature 확장 비교
- `10_model_family_comparison.ipynb`: feature set / model family 비교
- `11_recent_window_modeling.ipynb`: 최근 회차 구간 학습 비교
- `12_final_candidate_selection.ipynb`: 최종 후보 shortlist 및 대표 추천 선정
- `13_probability_ranking_analysis.ipynb`: 확률 랭킹 및 calibration 진단
- `14_ticket_generation_strategy.ipynb`: 티켓 생성 전략 비교
- `15_next_draw_recommendation.ipynb`: 다음 회차 추천용 운영 notebook

### 현재 핵심 결과

#### 기본 모델링

- 최고 holdout `avg_hit`: `classifier_chain = 0.9000`
- 다음 holdout `avg_hit`: `logistic_regression = 0.8708`
- 최고 rolling backtest `mean_avg_hit`: `random_baseline = 0.8719`
- `subset_accuracy`는 저장된 주요 결과에서 계속 `0.0`

해석:
- 학습 모델이 무작위 성격의 baseline을 안정적으로 크게 앞선다고 보기는 아직 어렵습니다.

#### 날씨 컨텍스트

- `rain_flag`: `94`회차, 약 `7.72%`
- `snow_flag`: `6`회차, 약 `0.49%`

현재 검정 결과로는 비, 눈, 온도 구간, 습도 구간이 로또 패턴을 강하게 바꾼다는 근거는 약합니다.

#### 날씨 확장 feature 모델링

`09_weather_feature_modeling.ipynb` 결과:

- holdout `avg_hit`
  - `base = 0.8708`
  - `base_plus_calendar = 0.8583`
  - `base_plus_calendar_weather = 0.8458`
  - `calendar_weather_only = 0.8292`
- backtest `mean_avg_hit`
  - `base = 0.8389`
  - `base_plus_calendar_weather = 0.8139`
  - `base_plus_calendar = 0.7944`
  - `calendar_weather_only = 0.7722`

해석:
- 날짜/날씨 변수는 예측 성능 향상보다 설명용 컨텍스트로서의 가치가 더 큽니다.

#### 모델 패밀리 비교

`10_model_family_comparison.ipynb` 결과:

- 최고 holdout 조합: `base_plus_pattern + classifier_chain = 0.9042`
- 다음 holdout 조합: `base + classifier_chain = 0.9000`
- 최고 backtest 조합도 heuristic 또는 약한 구조적 baseline과 큰 차이를 보이지 않음
- `base_plus_context + mlp`는 평균 점수 기준 상위권이었지만, fold별 변동과 작은 표본 문제 때문에 보수적으로 해석해야 함

해석:
- 달력/날씨보다 로또 내부 패턴 feature가 더 유망합니다.
- 다만 robust한 우위는 여전히 제한적입니다.

#### 최근 구간 학습

`11_recent_window_modeling.ipynb` 결과:

- 최근 구간 중 가장 유망한 설정은 `last_500`
- strongest holdout candidate: `base_plus_context + soft_voting_ensemble + last_500 = 0.9046`
- shortlist 상위권은 `soft_voting_ensemble`, `mlp`, `classifier_chain` 위주
- backtest 개선 폭은 존재하지만 결정적이라고 보긴 어려움

해석:
- 전체 이력을 완전히 버리기보다는, 최근 `500`회 중심 학습이 가장 실험 가치가 높았습니다.

#### 최종 후보 및 확률 랭킹

`12_final_candidate_selection.ipynb`, `13_probability_ranking_analysis.ipynb` 결과:

- 최종 추천 후보
  - `feature_set = full_feature_set`
  - `model = soft_voting_ensemble`
  - `training_regime = last_500`
- 확률 랭킹 요약
  - `avg_hit = 0.8631`
  - `precision_at_6 = 0.1438`
  - `brier_score = 0.1952`
- 평균 예측 확률
  - 양성 번호 `= 0.2774`
  - 음성 번호 `= 0.2680`

해석:
- 최종 후보는 “정답 번호를 딱 맞히는 모델”보다는 “번호별 확률 랭킹을 만드는 모델”로 보는 편이 더 정직합니다.

#### 티켓 전략 비교

`14_ticket_generation_strategy.ipynb` 결과:

- 단일 티켓 기준 최고 전략:
  - `greedy_top6 = 0.8631 avg_hit`
- 다중 티켓 다양성 기준 최고 전략:
  - `weighted_sampling = 1.43 mean pairwise overlap`
- `diversified_top_pool`은 `weighted_sampling`보다 overlap이 크고, `greedy_top6`보다 평균 적중도도 낮아 상대적으로 덜 매력적입니다.

해석:
- 대표 티켓 1장을 고를 때는 `greedy_top6`
- 여러 장을 분산해서 추천할 때는 `weighted_sampling`
이 가장 합리적입니다.

### 실행 가이드

이 프로젝트는 notebook-first 구조입니다. 권장 흐름은 다음과 같습니다.

1. Docker 실행 후 Jupyter 접속
2. notebook 안에서 `src.pipelines`의 공용 helper 사용
3. 수집, 전처리, feature 생성, 모델링을 재현 가능한 공용 함수로 유지
4. 공식 feature set은 `src.features`에서 관리
5. notebook은 orchestration, 해석, export 중심으로 사용

#### Docker 실행

```powershell
docker compose -f docker/docker-compose.yml up --build
```

그 다음 Jupyter 접속:

- `http://localhost:8888`

#### notebook 공통 import

```python
from src.pipelines import (
    run_sync_pipeline,
    run_data_pipeline,
    run_feature_pipeline,
    run_weather_fetch_pipeline,
    run_weather_build_pipeline,
    run_weather_sync_pipeline,
    run_model_pipeline,
)
```

#### 로또 sync

canonical raw workbook:

- `app/data/raw/lotto_history_latest.xlsx`

notebook 셀에서:

```python
run_sync_pipeline()
```

#### 전처리

processed 출력:

- `app/data/processed/lotto_cleaned.csv`

notebook 셀에서:

```python
run_data_pipeline(source="excel")
```

#### feature 생성

feature 출력:

- `app/data/processed/lotto_features.csv`

notebook 셀에서:

```python
feature_df = run_feature_pipeline(window=20)
feature_df.head()
```

현재 공식 feature set 이름:

- `base`
- `base_plus_pattern`
- `base_plus_context`
- `full_feature_set`

#### 날씨 sync

canonical weather 파일:

- `app/data/external/draw_metadata.csv`
- `app/data/external/weather_observations.csv`
- `app/data/external/weather_draw_context.csv`

환경변수:

```env
KMA_AUTH_KEY=your-issued-key
```

권장 흐름:

```python
weather_fetch_bundle = run_weather_fetch_pipeline()
weather_build_bundle = run_weather_build_pipeline()
```

#### 모델링

모델 출력:

- `app/models/results/`
- `app/models/artifacts/`

notebook 셀에서:

```python
results = run_model_pipeline(
    window=20,
    test_ratio=0.2,
    random_seed=42,
    backtest_initial_train_size=600,
    backtest_test_size=30,
    backtest_step_size=30,
    feature_set_name="base",
)
```

최근 구간 학습 실험은:

```python
results = run_model_pipeline(
    feature_set_name="full_feature_set",
    train_window_size=500,
)
```

처럼 사용할 수 있습니다.

### 운영 팁

- 평소 운영용으로는 `15_next_draw_recommendation.ipynb`만 다시 실행하는 흐름이 가장 가볍습니다.
- 이 notebook은 로또 sync, 전처리, feature 재생성, 선택적 날씨 refresh, 다음 회차 추천까지 한 번에 처리하도록 설계돼 있습니다.

### 저장소 구조

```text
002_lottoAnalysis/
├─ app/
│  ├─ data/
│  │  ├─ raw/
│  │  ├─ processed/
│  │  └─ external/
│  ├─ notebooks/
│  ├─ reports/
│  ├─ src/
│  └─ tests/
├─ docker/
├─ Makefile
└─ README.md
```

### 날씨 데이터 출처 및 공개 정책

날씨 데이터는 KMA API Hub를 통해 수집됩니다.

참고:

- KMA API Hub info: https://apihub.kma.go.kr/apiInfo.do
- KMA copyright policy: https://www.kma.go.kr/kma/guide/copyright.jsp

실무적으로는:

- 가공된 날씨 CSV는 출처 표기를 유지하면 공개 가능성이 높음
- 이 README와 `app/data/external/README.md`에 출처 유지
- `.env` 및 private API key는 절대 커밋하지 않기

### 현재 프로젝트 결론

이 프로젝트는 이제 “정확히 번호 6개를 맞히는 시스템”보다는 **확률 랭킹 + 추천 티켓 전략 비교 프로젝트**로 보는 것이 더 적절합니다.

- 최종 대표 모델:
  - `full_feature_set + soft_voting_ensemble + last_500`
- 최종 단일 티켓 전략:
  - `greedy_top6`
- 최종 다중 티켓 전략:
  - `weighted_sampling`

핵심 해석:

- 외부 날씨/달력보다 내부 temporal / pattern feature가 더 중요
- 최근 `500`회 학습이 가장 유망한 recent-window 설정
- 최종 모델은 exact prediction보다 ranking engine으로 해석하는 편이 더 설득력 있음

### 다음 추천 작업

- `soft_voting_ensemble + full_feature_set + last_500` 중심 추가 튜닝
- coefficient / feature importance 요약 추가
- `classifier_chain`, `mlp`와 calibration 및 ranking 품질 비교
- `weighted_sampling` 개선 및 더 나은 diversity-aware generator 탐색
- 다음 회차 추천용 export 또는 경량 배포 흐름 정리

---

## English

### Overview

This project studies Korean Lotto 6/45 history through a notebook-first workflow. The current scope covers:

- raw lotto sync and preprocessing
- exploratory and statistical analysis
- calendar and weather context analysis
- baseline and extended model comparison
- recent-window validation and final candidate selection
- probability ranking and ticket-generation strategy analysis
- report-ready figure and table exports

### Current Notebook Layout

The project currently includes notebook steps `01` through `15`.

- `01_data_collection.ipynb`: data collection overview
- `02_eda.ipynb`: exploratory visualization
- `03_statistical_tests.ipynb`: randomness-oriented tests
- `04_feature_engineering.ipynb`: temporal feature construction
- `05_model_baseline.ipynb`: baseline model training and artifact export
- `06_model_evaluation.ipynb`: evaluation plots and summaries
- `07_contextual_analysis.ipynb`: calendar context analysis
- `08_weather_context_analysis.ipynb`: weather context analysis
- `09_weather_feature_modeling.ipynb`: weather-aware feature comparison
- `10_model_family_comparison.ipynb`: feature-set and model-family comparison
- `11_recent_window_modeling.ipynb`: recent-history window comparison
- `12_final_candidate_selection.ipynb`: final candidate shortlist and recommendation
- `13_probability_ranking_analysis.ipynb`: probability calibration and ranking diagnostics
- `14_ticket_generation_strategy.ipynb`: ticket-construction strategy comparison
- `15_next_draw_recommendation.ipynb`: operational next-draw recommendation notebook

### Current Key Findings

#### Baseline Modeling

- best holdout `avg_hit`: `classifier_chain = 0.9000`
- next best holdout `avg_hit`: `logistic_regression = 0.8708`
- best rolling backtest `mean_avg_hit`: `random_baseline = 0.8719`
- `subset_accuracy` remains `0.0` across saved runs

Interpretation:
- The project still does not show strong evidence that learned models reliably outperform a random-style baseline in a robust way.

#### Weather Context

- `rain_flag`: `94` draws, about `7.72%`
- `snow_flag`: `6` draws, about `0.49%`

Current weather-group tests do not show strong evidence that rain, snow, temperature bins, or humidity bins materially shift draw patterns.

#### Weather-Aware Feature Modeling

From `09_weather_feature_modeling.ipynb`:

- holdout `avg_hit`
  - `base = 0.8708`
  - `base_plus_calendar = 0.8583`
  - `base_plus_calendar_weather = 0.8458`
  - `calendar_weather_only = 0.8292`
- backtest `mean_avg_hit`
  - `base = 0.8389`
  - `base_plus_calendar_weather = 0.8139`
  - `base_plus_calendar = 0.7944`
  - `calendar_weather_only = 0.7722`

Interpretation:
- Calendar and weather variables are currently more useful as explanatory context than as performance-improving predictors.

#### Model-Family Comparison

From `10_model_family_comparison.ipynb`:

- best holdout combination: `base_plus_pattern + classifier_chain = 0.9042`
- next best holdout combination: `base + classifier_chain = 0.9000`
- best backtest combinations are still close to heuristic or weakly-structured baselines
- `base_plus_context + mlp` reached top-tier average scores, but should still be interpreted cautiously because of fold variability and small-sample instability

Interpretation:
- Richer internal pattern features look more promising than calendar/weather context.
- Robust superiority over random-style baselines remains limited.

#### Recent-Window Modeling

From `11_recent_window_modeling.ipynb`:

- the most promising recent-only regime is `last_500`
- strongest holdout candidate: `base_plus_context + soft_voting_ensemble + last_500 = 0.9046`
- strongest shortlist candidates are concentrated around `soft_voting_ensemble`, `mlp`, and `classifier_chain`
- backtest improvements remain moderate rather than decisive

Interpretation:
- Restricting training to the most recent `500` draws is more promising than `last_300` or `last_200`, but the gain is not large enough to overturn the project’s conservative conclusion.

#### Final Candidate and Probability Ranking

From `12_final_candidate_selection.ipynb` and `13_probability_ranking_analysis.ipynb`:

- final recommended candidate
  - `feature_set = full_feature_set`
  - `model = soft_voting_ensemble`
  - `training_regime = last_500`
- ranking summary
  - `avg_hit = 0.8631`
  - `precision_at_6 = 0.1438`
  - `brier_score = 0.1952`
- mean predicted probability
  - positives `= 0.2774`
  - negatives `= 0.2680`

Interpretation:
- The final candidate is more honestly interpreted as a number-ranking model than as a sharp exact-prediction model.

#### Ticket Strategy Comparison

From `14_ticket_generation_strategy.ipynb`:

- best single-ticket strategy:
  - `greedy_top6 = 0.8631 avg_hit`
- best diversity-oriented multi-ticket strategy:
  - `weighted_sampling = 1.43 mean pairwise overlap`
- `diversified_top_pool` is currently less compelling because it overlaps much more while still trailing `greedy_top6` on average hit

Interpretation:
- Use `greedy_top6` for one representative ticket.
- Use `weighted_sampling` for a diversified multi-ticket set.

### Run Guide

This project is notebook-first. The preferred workflow is:

1. start Docker and open Jupyter
2. use shared pipeline helpers from `src.pipelines` inside notebooks
3. keep raw collection, preprocessing, feature generation, and modeling reproducible through those shared functions
4. build official feature sets through `src.features`
5. use notebooks for orchestration, interpretation, and export

#### Start Docker

```powershell
docker compose -f docker/docker-compose.yml up --build
```

Then open Jupyter:

- `http://localhost:8888`

#### Standard Notebook Imports

```python
from src.pipelines import (
    run_sync_pipeline,
    run_data_pipeline,
    run_feature_pipeline,
    run_weather_fetch_pipeline,
    run_weather_build_pipeline,
    run_weather_sync_pipeline,
    run_model_pipeline,
)
```

#### Lotto Sync

Canonical raw workbook:

- `app/data/raw/lotto_history_latest.xlsx`

Run from a notebook cell:

```python
run_sync_pipeline()
```

#### Preprocessing

Processed output:

- `app/data/processed/lotto_cleaned.csv`

Run from a notebook cell:

```python
run_data_pipeline(source="excel")
```

#### Feature Generation

Feature output:

- `app/data/processed/lotto_features.csv`

Run from a notebook cell:

```python
feature_df = run_feature_pipeline(window=20)
feature_df.head()
```

Official feature-set names:

- `base`
- `base_plus_pattern`
- `base_plus_context`
- `full_feature_set`

#### Weather Sync

Canonical weather files:

- `app/data/external/draw_metadata.csv`
- `app/data/external/weather_observations.csv`
- `app/data/external/weather_draw_context.csv`

Environment setup:

```env
KMA_AUTH_KEY=your-issued-key
```

Recommended flow:

```python
weather_fetch_bundle = run_weather_fetch_pipeline()
weather_build_bundle = run_weather_build_pipeline()
```

#### Modeling

Model outputs:

- `app/models/results/`
- `app/models/artifacts/`

Run from a notebook cell:

```python
results = run_model_pipeline(
    window=20,
    test_ratio=0.2,
    random_seed=42,
    backtest_initial_train_size=600,
    backtest_test_size=30,
    backtest_step_size=30,
    feature_set_name="base",
)
```

For recent-window experiments:

```python
results = run_model_pipeline(
    feature_set_name="full_feature_set",
    train_window_size=500,
)
```

### Operating Tip

For regular operation after a new draw arrives, rerunning `15_next_draw_recommendation.ipynb` is usually enough. It is designed to perform lotto sync, preprocessing refresh, feature refresh, optional weather refresh, and next-draw recommendation in one place.

### Repository Layout

```text
002_lottoAnalysis/
├─ app/
│  ├─ data/
│  │  ├─ raw/
│  │  ├─ processed/
│  │  └─ external/
│  ├─ notebooks/
│  ├─ reports/
│  ├─ src/
│  └─ tests/
├─ docker/
├─ Makefile
└─ README.md
```

### Weather Data Source Policy

Weather data in `app/data/external/` comes from KMA API Hub.

References:

- KMA API Hub info: https://apihub.kma.go.kr/apiInfo.do
- KMA copyright policy: https://www.kma.go.kr/kma/guide/copyright.jsp

Practical guidance:

- derived weather CSV outputs are likely acceptable to share when source attribution is preserved
- keep attribution in this README and `app/data/external/README.md`
- never commit `.env` or any private API key

### Project Conclusion

The project is now best framed as a **probability-ranking and ticket-strategy study** rather than a direct exact-number prediction system.

- final representative model:
  - `full_feature_set + soft_voting_ensemble + last_500`
- final single-ticket strategy:
  - `greedy_top6`
- final diversified multi-ticket strategy:
  - `weighted_sampling`

Core interpretation:

- internal temporal and pattern features matter more than external calendar/weather context
- `last_500` is the most useful recent-window regime tested so far
- the final model is more persuasive as a ranking engine than as an exact prediction engine

### Suggested Next Steps

- tune `soft_voting_ensemble + full_feature_set + last_500`
- add coefficient and feature-importance summaries
- compare calibration and ranking quality against lighter top candidates such as `classifier_chain` and `mlp`
- refine `weighted_sampling` and explore stronger diversity-aware generators
- add a cleaner export or lightweight deployment path for next-draw recommendation
