# 🚀 MZ 밈 통역기 PRO (MZ Meme Translator)

> **"소통의 단절을 넘어, 세대 간의 다리가 되다."** > RAG(Retrieval-Augmented Generation) 기술을 활용한 지능형 MZ 신조어/밈 해석 서비스

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-v1.38.0-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-v0.2-1C3C3C?style=flat&logo=langchain&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-412991?style=flat&logo=openai&logoColor=white)

## 📖 프로젝트 소개 (Overview)
급변하는 밈(Meme) 문화 속에서 소외감을 느끼는 기성세대와 한국 문화에 낯선 외국인을 위해 개발된 **AI 기반 밈 통역기**입니다.  
단순한 단어 뜻풀이를 넘어, **사용 상황(Context)**과 **적절한 화법(Persona)**까지 코칭하여 실질적인 소통을 돕습니다.

### 🎯 핵심 타겟 (Target Persona)
1.  **👨‍💼 이시대의 낀세대 (Manager Mode):** MZ 사원들과 오해 없이 소통하고 싶은 부장님
2.  **👱 대한외국인 (Foreigner Mode):** K-밈의 뉘앙스를 영어로 이해하고 싶은 외국인 친구
3.  **😎 복학생 (Student Mode):** 유행에 뒤처지기 싫은 밈린이

---

## 🛠️ 기술 아키텍처 (System Architecture)

본 서비스는 **LangChain 프레임워크**를 기반으로 한 **RAG(검색 증강 생성)** 파이프라인으로 구축되었습니다.

### 🔍 v9.0 Core Features
**1. Hybrid Search Retriever (정밀 검색)**
   - `BM25` (키워드 매칭) + `ChromaDB` (의미 기반 벡터 검색)를 결합한 **Ensemble Search** 구현.
   - 고유명사(밈 단어)의 정확성과 문맥의 유사성을 동시에 확보 (가중치 3:7 적용).

**2. Query Expansion & Preprocessing (전처리)**
   - 한국어 특성상 붙는 조사(은/는/이/가)를 제거하고 핵심 어근(Stem)만 추출.
   - 1차적으로 LLM이 사용자의 질문 의도를 파악하여 검색어를 최적화하는 **Query Router** 도입.

**3. Safety Filtering System (안전성 확보)**
   - 데이터 수집 단계에서 `SafetyLevel` 메타데이터(Safe/Caution/Danger) 태깅.
   - **User Persona**에 따라 위험한 밈(비속어 등)이 검색 단계에서부터 노출되지 않도록 **Metadata Filtering** 적용.

**4. Persona Injection (답변 생성)**
   - **System Prompt Engineering**을 통해 각 모드별 어조와 포맷(DO/DON'T 가이드)을 강제.
   - 외국인 모드 시 한국어/영어 동시 출력 기능 구현.

---

## 📂 프로젝트 구조 (Directory Structure)

```bash
├── 2025-12-30_RAG_랭체인톤_밈통역기/
│   ├── app.py                # 메인 애플리케이션 (Streamlit Frontend & Backend)
│   ├── mz_meme_data_v3.csv   # 밈 데이터베이스 (Knowledge Base)
│   └── requirements.txt      # 프로젝트 의존성 패키지 목록
└── README.md                 # 프로젝트 설명서

# 레포지토리 클론
git clone [https://github.com/MJ-GINI/AiffelRepository.git](https://github.com/MJ-GINI/AiffelRepository.git)
cd 2025-12-30_RAG_랭체인톤_밈통역기

# 가상환경 생성 및 활성화 (선택 사항)
python -m venv venv
source venv/bin/activate  # Mac/Linux
# venv\Scripts\activate   # Windows

pip install -r requirements.txt

OPENAI_API_KEY=sk-your-api-key-here

streamlit run app.py

# 향후 로드맵
1. Data Pipeline Automation
웹 크롤링 및 자동 임베딩 파이프라인 구축을 통한 실시간 트렌드(Real-time Trend) 반영.

2. Multimodal & Advanced Search
이미지(Vision) 인식 기능 도입 및 AI 에이전트 기반의 정밀한 외부 정보(Youtube, News) 큐레이션 구현.

3. User Feedback Loop
사용자 평가(RLHF) 기반 모델 고도화 및 기업 맞춤형 커스텀 페르소나(Custom Persona) 지원.
