# AutoInt+ 기반 개인화 영화 추천 시스템 메인 프로젝트

## 1. 개요
MovieLens-1M 데이터셋을 활용하여 사용자 맞춤형 영화 추천 서비스를 구축해보기
- 일자: 2026년 1월 8일 목요일 ~ 1월 10일 토요일
- 데이터셋: MovieLens 1M (User: 6,040명, Movie: 3,706개, Rating: 100만 건)
- 사용 기술: Python, TensorFlow (Keras), Pandas, Streamlit, Scikit-learn
- 사용 모델: AutoInt 기본 모델 및 피처 간의 상호작용뿐만 아니라 비선형적 패턴까지 학습하도록 개선을 위해 MLP (Multi-Layer Perceptron)를 결합한 AutoInt+ 사용

## 2. 프로젝트 파일 구조
📊 데이터 파이프라인

* 깃허브 각 폴더 경로에도 기재해두었음

## 3. 모델 아키텍처
### 1️⃣ AutoInt (베이스 모델)
- 특징: Multi-head Self-Attention을 이용해 피처 간의 고차원 상호작용 자동 학습
- 한계: 명시적인 상호작용 외의 복잡한 비선형 패턴 학습에 한계가 있음

### 2️⃣ AutoInt+ (AutoInt + MLP)
- 개선점: Attention 구조에 DNN을 병렬로 결합하여 표현력 강화
- 구조:
  - Embedding Layer: 고차원 희소 데이터를 저차원 밀집 벡터로 변환
  - Attention Part: 피처 간의 연관성(Explicit Interaction) 학습
  - MLP Part: 데이터의 내재된 패턴(Implicit Interaction) 학습 (Dense + ReLU)
  - Output: 두 파트의 결과를 합산(Sum)하여 최종 예측
 
## 4. 이번 프로젝트에서 가장 힘들었던.. 트러블 슈팅!!
🤦🏻‍♀️ 1. 레이어 네임 매칭 실패로 인한 가중치 로드 실패
- 현상: 주피터 노트북과 로컬 VSCode 실행 시 세션 차이로 인해 저장된 가중치(.h5) 파일의 레이어 이름이 계속 mismatch 되어, streamlit 화면을 보지도 못하고 계속 오류 발생
- 해결 방안
  - Gemini와의 대화 끝에 .h5의 복잡한 이름 매칭 방식을 버리고, 가중치 값 자체를 Numpy 리스트로 추출하여 .pkl 파일로 저장 후 시도 -> 로딩 완료
  - 실행 시 `model.set_weights()`를 통해 순수 데이터만 주입하는 방식으로 호환성 확보

🤦🏻‍♀️ 2. Streamlit 무한 로딩
- 현상: Mac M4 환경에서 1) 사용자가 input 시, 전체 데이터를 계산하는데 시간이 너무 오래 소요됨 2) TensorFlow가 GPU를 점유하려다 충돌 발생
- 해결:
  - 1) 우선 안정적인 실행을 위해 전체 데이터가 아닌 샘플링으로 진행 -> 그에 따라 일부 사용자 넘버는 실행이 안되는 이슈가 발생되긴 함..
  - 2) os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 설정을 통해 강제로 CPU 모드로 실행
   
## 5. 성능 평가
### 1️⃣ 정량 평가
| 지표 (Metrics) | 점수 (Score) | 평가 |
|:--|:--:|:--|
| NDCG@10 | 0.6626 | 추천 순위의 정확도가 준수한 편 |
| HitRate@10 | 0.6329 | 상위 10개 추천 중 약 63% 확률로 사용자 선호 영화 포함하여, 매우 만족스럽진 못하지만 오류를 다시 해결할 자신이 없어 더 개선하지 못함 |
| Loss | 0.54 | 심각한 과적합 없이 나름의 안정적인 학습 상태 확인 |

### 2️⃣ 정성 평가
- 개인화: 사용자 ID 변경 시 추천 리스트가 즉각적으로 변화하며 개인 취향 반영 -> 단, 이것은 데이터 무한 로딩의 늪에 빠지지 않기 위해 샘플링의 영향이라 아쉬움
- 장르 연관성: 시청 이력(예: 액션, SF)과 유사하거나 연관된 장르가 추천되는 점 확인 완료 (다양한 장르의 영화를 본 사용자 대상으로는 그에 맞게 다양하게 추천 / 추천 점수도 높은 편)
- 다양성: 인기 영화뿐만 아니라, 사용자의 숨겨진 취향에 맞는 독특한 영화도 추천

### 🔥 Streamlit 캡처 이미지
#### 1️⃣ AutoInt 기본 모델
<img width="520" height="540" alt="image" src="https://github.com/user-attachments/assets/5130d222-0da9-49ba-bafd-db14b6b4693f" />

<img width="532" height="540" alt="image" src="https://github.com/user-attachments/assets/efc3f290-3243-4a31-b6f5-342913635c19" />

<img width="504" height="540" alt="image" src="https://github.com/user-attachments/assets/32812515-8f3a-467f-9f98-cc712b463b1f" />

#### 2️⃣ AutoInt+ MLP 모델
<img width="1494" height="1524" alt="image" src="https://github.com/user-attachments/assets/db52a88c-ad02-4cbe-945b-4d18e2ef389f" />

<img width="1498" height="1550" alt="image" src="https://github.com/user-attachments/assets/6ef897e8-83fe-43c4-97a6-c526346ebb03" />

<img width="1496" height="1592" alt="image" src="https://github.com/user-attachments/assets/f543dc55-5fd9-466f-8f75-dcce7dad0765" />

<img width="1506" height="1552" alt="image" src="https://github.com/user-attachments/assets/09a15c83-2fa9-4bfa-b2d9-355df124f516" />

