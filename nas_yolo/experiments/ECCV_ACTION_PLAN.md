# ECCV 2026 제출 15일 액션 플랜

> Registration: 2월 26일 | Submission: 3월 5일

---

## Week 1: 실험 (2/18 ~ 2/25)

### Day 1-2 (2/18-19): 환경 세팅 + 학습 시작
- [ ] GPU 서버 확보 (A100 x 1~4 권장)
- [ ] COCO 2017 다운로드 + 경로 설정
- [ ] `pip install -r nas_yolo/requirements.txt`
- [ ] `make smoke-test` — 모델 동작 확인
- [ ] `make train-full GPU=0` 시작 (NAS-YOLO-s, ~3일 소요)
- [ ] 다른 GPU에서 `make train-nano GPU=1` 병렬 시작

### Day 3-4 (2/20-21): Baseline 수집 + Ablation 학습 시작
- [ ] `pip install ultralytics` → YOLOv8/v9/v10 pre-trained 다운로드
- [ ] Baseline mAP 측정: `bash nas_yolo/experiments/run_baselines.sh`
- [ ] Baseline BSI 측정: `python -m nas_yolo.experiments.eval_baseline_bsi`
- [ ] Ablation 학습 시작 (3개 모델, 각 GPU에 분배):
  - `make train-ablation-no-temporal GPU=2`
  - `make train-ablation-no-gate GPU=3`
  - `make train-ablation-no-tcl GPU=0` (full 학습 끝나면)

### Day 5 (2/22): 중간 결과 확인 + Corruption 평가
- [ ] NAS-YOLO-s 학습 ~100 epoch 지점 intermediate eval
- [ ] Baseline corruption 평가 실행 (COCO-C)
- [ ] 초기 결과로 테이블 뼈대 채우기

### Day 6 (2/23): 결과 수집 시작
- [ ] NAS-YOLO-s 학습 완료 예상 → `make eval-full`
- [ ] NAS-YOLO-n 학습 완료 → `make eval-nano`
- [ ] 전체 mAP-C 평가 실행 (15 corruption × 5 severity = 75회)

### Day 7 (2/24): 결과 정리 + Registration
- [ ] 모든 ablation 학습 완료 → `make eval-ablations`
- [ ] `make tables` — LaTeX 테이블 자동 생성
- [ ] `make benchmark` — 속도 벤치마크
- [ ] 결과 검토: 스토리가 되는지 확인

### Day 7.5 (2/25): OpenReview Registration
- [ ] **ECCV 2026 OpenReview 등록** (2/26 마감!)
- [ ] 제목, 저자, abstract 등록
- [ ] 저자 OpenReview 프로필 확인

---

## Week 2: 논문 작성 (2/25 ~ 3/5)

### Day 8-9 (2/25-26): 결과 테이블 완성 + Figure 제작
- [ ] Table 1 (Main Results): 숫자 채우기
- [ ] Table 2 (Corruption): per-category breakdown
- [ ] Table 3 (BSI): stability comparison
- [ ] Table 4 (Ablation): 모든 variant 결과
- [ ] Figure 1: Architecture diagram (draw.io / TikZ)
- [ ] Figure 2: Severity response curves (matplotlib)
- [ ] Figure 3: Box flicker comparison (trajectory overlay)

### Day 10-11 (2/27-28): 논문 본문 작성
- [ ] Introduction: 결과 숫자로 claim 확정
- [ ] Method: 수식 검토 + notation 통일
- [ ] Experiments: 결과 해석 + discussion
- [ ] Related Work: 최신 논문 추가 (2024-2025)
- [ ] Conclusion: 한 페이지 내로 마무리

### Day 12 (3/1): 1차 완성 + 교정
- [ ] 전체 논문 14페이지 이내 확인
- [ ] 모든 테이블/그림에 올바른 숫자 확인
- [ ] Notation consistency check
- [ ] Reference 누락 확인

### Day 13 (3/2): Supplementary 준비
- [ ] 추가 ablation 결과 (window length detail)
- [ ] Per-class AP breakdown
- [ ] 더 많은 visualization
- [ ] 코드 공개 계획 명시

### Day 14 (3/3): 리뷰 + 수정
- [ ] 공저자 리뷰 반영
- [ ] Figure 해상도 확인 (300 DPI)
- [ ] 문법/영어 검토
- [ ] Abstract 최종 수정

### Day 15 (3/4): 최종 제출
- [ ] 최종 PDF 생성: `make paper`
- [ ] Supplementary PDF 생성
- [ ] OpenReview 업로드
- [ ] Conflict of interest 확인
- [ ] **제출 완료!** (3/5 11:00 PM CET 마감)

---

## 핵심 실험 체크리스트

### 반드시 필요한 결과 (MUST)
- [ ] Table 1: NAS-YOLO vs baselines (mAP, mAP-C, BSI, Params, FPS)
- [ ] Table 2: Ablation study (S3M, Gate, TCL 각각의 기여)
- [ ] Table 3: Per-corruption category breakdown
- [ ] Table 4: BSI component analysis
- [ ] Figure: Architecture diagram
- [ ] Figure: Severity response curves

### 있으면 좋은 결과 (NICE-TO-HAVE)
- [ ] ExDark low-light 결과
- [ ] ACDC driving 결과
- [ ] Feature map visualization (before/after NAS)
- [ ] Box flicker GIF/figure
- [ ] Temporal window ablation detail
- [ ] FLOPs/energy comparison

---

## 빠른 커맨드 모음

```bash
# 1. 환경 설정
pip install -r nas_yolo/requirements.txt
make smoke-test

# 2. 학습 (병렬 가능)
make train-full GPU=0 &
make train-nano GPU=1 &

# 3. Ablation (full 학습 후)
make train-ablations GPU=0

# 4. 평가
make eval-all GPU=0
make benchmark GPU=0

# 5. 논문 재료
make tables
make figures
make paper
```
