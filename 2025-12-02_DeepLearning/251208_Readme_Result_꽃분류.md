# 결과 요약 및 회고

## 1.개요
- 목적: Transfer Learning을 활용하여 5종의 꽃 이미지를 85% 이상의 정확도로 분류하는 모델 개발
- 꽃 클래스: 총 5개 (Daisy, Dandelion, Roses, Sunflowers, Tulips)

## 2.최종 모델 요약
<img width="860" height="358" alt="image" src="https://github.com/user-attachments/assets/f3f837a1-7064-4bfc-a79d-4ce328058743" />

## 3.모델 성능 검증 단계별 시도 사항 및 요약
| 단계 | 시도 사항 및 주요 전략 | 베이스 모델 | 결과 (Test Acc) | 소요 시간(Epoch당) |
|:-----|:------------------------|:------------|:------------------|:---------------------|
| 1차 | 기본 전이 학습 | VGG16 | 71.88% | 약 11분 |
| 2차 | 경량화 모델로 변경 및 재학습 | MobileNetV2 | 81.25% | 약 4~5초 |
| 3차 | Fine-tuning | MobileNetV2 | 91.83% | 약 4~5초 |

## 4.학습 과정 세부
[1차 시도]
- 고양이 vs. 강아지 분류 모델 학습 시의 VGG16 모델 구조 그대로 적용
- train 정확도는 준수 but 파라미터의 양이 많아 학습 속도가 매우 느려 효율적인 실험 불가
- 특히 이후 파인튜닝 고려 시, 해당 모델로는 진행 불가하다고 판단했음

[2차 시도]
- 경량화 모델이라는 MobileNetV2로 교체하여 연산량 최적화. Dropout(0.2)을 추가하여 과적합 방지 시도
- 학습 시간 대폭 단축 및 1차 71% 대비 10%p 향상된 81% 기록

[3차 시도]
- 목표 점수인 test accuracy 85% 도달을 위해 파인 튜닝 진행
- BaseModel.trainable = True로 변경, 학습 진행 시 성능 달성 후 조기 종료(early stopping), 학습률을 매우 낮게 조정
- train vs. validation 정확도 및 손실값 간의 그래프 확인하여 과적합 여부 확인 -> 안정성 / 일반화 성능 확인
  <img width="2601" height="1093" alt="image" src="https://github.com/user-attachments/assets/a0980156-6ac4-4557-a6f1-27855e84342b" />
- 혼동행렬 히트맵 확인을 통한 클래스별 혼동된 꽃 분류 체크 \
  (단, 본 프로젝트에서는 혼동행렬 확인만 하고 추가 디벨롭 하지 않음)
  <img width="1560" height="1401" alt="image" src="https://github.com/user-attachments/assets/cda3867b-8f8d-4a3e-a59c-0596f666f4ab" />


## 5.결론
<img width="3477" height="2779" alt="image" src="https://github.com/user-attachments/assets/697cf7a3-a13b-4caf-bc4b-8a096a5d0014" />

- MobileNetV2는 모바일 환경에서 경량화된 모델이지만 VGG16 대비 성능도 뒤떨어지지 않아 이번 모델에 적합했다고 판단됨
- 또한 fine-tunning의 Layer 범위를 80까지 확대한 것이 유효했음
- 5가지 꽃 중에서 튤립에 대한 특징을 가장 잘 파악하여, 오답이 많이 없는 편이었음
- 단, 사람도 헷갈릴만한 일부 꽃 사진(튤립vs.장미 / 민들레vs.해바라기)의 정확한 분류를 위해 모델 성능을 디벨롭하는 방법을 아직 모름.. \
  (튤립 같이 생긴 장미는 사람도 헷갈릴 지경이라 이 모델도 잘못 예측하는 경우가 6건 있음 \
  또한 너무 해바라기처럼 활짝 펴 있는 민들레 사진도 오해할만 하나.. 성능을 좀 더 높일 수 있는 방법 모색 필요)
