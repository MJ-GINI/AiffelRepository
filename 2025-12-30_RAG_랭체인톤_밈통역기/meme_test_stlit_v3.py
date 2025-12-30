import os
import streamlit as st
import pandas as pd
import re
import time
import random
from io import BytesIO
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.docstore.document import Document
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# --- [페이지 설정] ---
st.set_page_config(page_title="MZ 밈 통역기 PRO v9.0", layout="wide")

# --- 1. HTML 변환 유틸리티 (UI 렌더링 엔진 - v8.6 Final) ---
def convert_st_to_html(text):
    if not text: return ""

    # [Sanitizer] 마크다운 볼드체(**) 기호 무조건 삭제
    text = text.replace("**", "")

    def clean_md(t):
        # 색상 태그 및 앞뒤 특수문자 제거
        t = re.sub(r'(?i):?green\s*\[(.*?)\]', r'\1', t)
        t = re.sub(r'(?i):?red\s*\[(.*?)\]', r'\1', t)
        t = re.sub(r'(?i):?blue\s*\[(.*?)\]', r'\1', t)
        t = re.sub(r'^[\*\-\s]+', '', t) 
        return t.strip()

    lines = text.split('\n')
    html_lines = []
    
    html_lines.append('<div class="response-container">')

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        
        # [Clean] 시스템 태그 제거
        if "[SearchKeyword]" in line:
            i += 1
            continue
        line = re.sub(r'(?i)^\[?closing\]?\s*:?', '', line).strip()
        if not line: 
            i += 1
            continue

        # 헤더 처리
        if line.startswith("🚀") or line.startswith("🧐") or "원포인트 코칭" in line or line.startswith("###") or re.match(r'^\d+\.', line):
            content = line.replace("###", "").strip()
            content = re.sub(r'^\d+\.\s*의미와 유래:', '', content).strip()
            html_lines.append(f'<div class="section-header">{clean_md(content)}</div>')
            i += 1
        
        # [Fix] 원포인트 코칭(신호등) 처리
        elif line.startswith("🟢") or line.startswith("🟡") or line.startswith("🔴") or line.startswith("💡"):
             html_lines.append(f'<div class="section-header" style="margin-top:15px; font-size:1em;">{clean_md(line)}</div>')
             i += 1

        # [Fix] Don't / Do 박스 처리
        elif re.search(r"(?i)(do|don'?t)", line):
            is_dont = bool(re.search(r"(?i)don'?t", line))
            box_class = "dont-box" if is_dont else "do-box"
            badge_class = "dont-badge" if is_dont else "do-badge"
            badge_text = "DON'T" if is_dont else "DO"
            
            # 본문 발라내기
            match = re.search(r"(?i)(do|don'?t)", line)
            if match:
                start_idx = match.end()
                raw_content = line[start_idx:].strip()
            else:
                raw_content = line
                
            raw_content = re.sub(r"^[:\-\s]+", "", raw_content).strip()
            
            if raw_content.lower().startswith("n't"):
                raw_content = raw_content[3:].strip()
            if raw_content.startswith(":"):
                raw_content = raw_content[1:].strip()

            next_line = ""
            if i + 1 < len(lines) and (lines[i+1].strip().startswith('"') or lines[i+1].strip().startswith("'")):
                next_line = lines[i+1].strip()
                i += 1
            
            full_content = raw_content + " " + next_line
            
            situation = ""
            dialogue = ""
            
            if ":" in full_content:
                parts = full_content.split(":", 1)
                situation = parts[0].strip()
                dialogue = parts[1].strip()
            elif '"' in full_content: 
                parts = full_content.split('"', 1)
                situation = parts[0].strip()
                dialogue = '"' + parts[1] 
            elif "'" in full_content:
                parts = full_content.split("'", 1)
                situation = parts[0].strip()
                dialogue = "'" + parts[1]
            else:
                situation = full_content 
            
            situation = situation.replace("(상황)", "").strip()
            if not situation: situation = "상황 예시"

            html_lines.append(f'<div class="guide-box {box_class}"><span class="badge {badge_class}">{badge_text}</span><span class="situation">{clean_md(situation)}</span><br><div class="dialogue" style="color:{"#ea4335" if is_dont else "#28a745"}; font-weight:bold;">{clean_md(dialogue)}</div></div>')
            i += 1
            
        elif "사이드바" in line or "sidebar" in line:
            html_lines.append(f'<div class="sidebar-link">{clean_md(line)}</div>')
            i += 1
        else:
            if "실전 사용 가이드" in line or line == "---":
                i += 1
                continue
            html_lines.append(f'<div class="text-content">{clean_md(line)}</div>')
            i += 1
            
    html_lines.append('</div>')
    return "".join(html_lines)

# --- 2. 데이터 로드 및 검색 엔진 ---
def kiwi_tokenize(text):
    text = re.sub(r'[^가-힣a-zA-Z0-9]', ' ', text)
    tokens = text.split() 
    bi_grams = [] 
    for token in tokens:
        if len(token) > 1:
            for i in range(len(token)-1):
                bi_grams.append(token[i:i+2])
    return tokens + bi_grams

@st.cache_resource
def get_search_engine():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "mz_meme_data_v3.csv")
    
    if not os.path.exists(file_path):
        st.error(f"'{file_path}' 파일을 찾을 수 없습니다!")
        return None, None, None
    
    try:
        df = pd.read_csv(file_path, encoding='utf-8-sig').fillna("")
    except:
        df = pd.read_csv(file_path).fillna("")

    documents = []
    for _, row in df.iterrows():
        safety = row.get('SafetyLevel', '')
        
        def extract_level(text, key):
            match = re.search(rf"\[{key}:\s*(Safe|Caution|Danger)\]", text, re.IGNORECASE)
            return match.group(1) if match else "Unknown"

        level_manager = extract_level(safety, "낀세대")
        level_foreigner = extract_level(safety, "외국인")
        level_student = extract_level(safety, "복학생")

        is_safe_manager = level_manager != "Danger"
        is_safe_foreigner = level_foreigner != "Danger"
        is_safe_student = True 

        content = (
            f"Meme: {row.get('Meme','')}\n"
            f"유사어: {row.get('ExpectedQueries','')}\n"
            f"의미: {row.get('Meaning','')}\n"
            f"맥락: {row.get('Context','')}\n"
            f"예시: {row.get('Example','')}\n"
            f"금지상황(AntiPattern): {row.get('AntiPattern','')}\n"
            f"유래(Origin): {row.get('Origin','')}\n"
            f"안전도: {safety}\n"
            f"타겟연령: {row.get('TargetAge','')}"
        )
        
        metadata = {
            "meme": row.get('Meme',''),
            "is_safe_manager": is_safe_manager,
            "is_safe_foreigner": is_safe_foreigner,
            "is_safe_student": is_safe_student,
            "level_manager": level_manager,
            "level_foreigner": level_foreigner,
            "level_student": level_student
        }
        documents.append(Document(page_content=content, metadata=metadata))
    
    vector_db = Chroma.from_documents(documents, OpenAIEmbeddings(model='text-embedding-3-small'))
    
    try:
        bm25_retriever = BM25Retriever.from_documents(
            documents, 
            preprocess_func=kiwi_tokenize
        )
        bm25_retriever.k = 3
    except ImportError:
        bm25_retriever = None

    return vector_db, bm25_retriever, df

# --- 3. UI 스타일 및 세션 ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_processed_query" not in st.session_state:
    st.session_state.last_processed_query = None

st.markdown("""
<style>
@import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');
html, body, [class*="css"] { font-family: 'Pretendard', sans-serif; }
.text-content { margin-bottom: 6px; line-height: 1.6; }
.sidebar-link { margin-top: 15px; font-size: 0.9em; color: #666; font-weight: 500; border-top: 1px solid #eee; padding-top:10px; }
.section-header { display: block; font-weight: 700; font-size: 1.05em; margin-top: 18px; margin-bottom: 8px; color: #333; clear: both; }
.guide-box { display: block; padding: 12px 16px; border-radius: 12px; margin-top: 6px; margin-bottom: 6px; font-size: 0.95em; line-height: 1.5; clear: both; }
.do-box { background-color: rgba(40, 167, 69, 0.08); border: 1px solid rgba(40, 167, 69, 0.25); }
.dont-box { background-color: rgba(234, 67, 53, 0.08); border: 1px solid rgba(234, 67, 53, 0.25); }
.situation { font-weight: 700; color: #555; font-size: 0.9em; margin-left: 6px; }
.dialogue { margin-top: 6px; display: block; word-break: break-word; }
.badge { display: inline-block; padding: 3px 8px; border-radius: 6px; font-size: 0.75em; font-weight: 800; margin-right: 4px; vertical-align: middle; }
.do-badge { background-color: #28a745; color: white; }
.dont-badge { background-color: #ea4335; color: white; }
.chat-wrap { margin-bottom: 30px; }
.user-bubble { background: #4f46e5; color: white; padding: 12px 20px; border-radius: 20px 20px 0px 20px; max-width: 70%; margin-left: auto; margin-bottom: 10px; font-weight: 500; }
.ai-bubble { background: #ffffff; color: #111111; padding: 24px; border-radius: 20px 20px 20px 0px; max-width: 95%; border: 1px solid rgba(0,0,0,0.08); box-shadow: 0 4px 15px rgba(0,0,0,0.05); }
.onboarding-container { padding: 25px; background-color: rgba(0,0,0,0.02); border-radius: 15px; border: 1px solid rgba(0,0,0,0.05); margin-bottom: 30px; }
.onboarding-title { margin-bottom: 20px; font-weight: 800; color: #111; font-size: 1.5em; }
.onboarding-text { color: #444; line-height: 1.6; }
.onboarding-tip { margin-top: 25px; font-size: 0.95em; color: #666; border-top:1px dashed #ddd; padding-top:15px; font-weight: 500; }
[data-testid="stSidebar"] hr { margin: 10px 0 !important; }
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { 
    padding-top: 0.5rem !important; padding-bottom: 0.2rem !important; margin-bottom: 0.2rem !important;
}
[data-testid="stSidebar"] .stRadio { margin-top: 0rem !important; }
@media (prefers-color-scheme: dark) {
    .ai-bubble { background: #1f2937; color: #f9fafb; border: 1px solid rgba(255,255,255,0.1); }
    .section-header { color: #eee; }
    .situation { color: #ccc; }
    .sidebar-link { color: #aaa; border-top: 1px solid #444; }
    .onboarding-container { background-color: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); }
    .onboarding-title { color: #ffffff !important; }
    .onboarding-text { color: #dddddd !important; }
    .onboarding-tip { color: #aaaaaa !important; border-top:1px dashed #555 !important; }
}
</style>
""", unsafe_allow_html=True)

# --- 4. 로직 및 프롬프트 설정 ---
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
vector_db, bm25_retriever, raw_df = get_search_engine()

target = "이시대의 낀세대" 

def get_persona_settings(target_key):
    sidebar_msg = "👈🏻 관련 정보는 좌측 사이드바를 확인하세요!"
    if target_key == "이시대의 낀세대":
        return {
            "prompt": "인자하고 전문적인 부장님 말투. 정중한 존댓말. 영어 금지.",
            "filter": {"is_safe_manager": True},
            "meta_key": "level_manager", 
            "scenario_guide": "Do 상황: 회식 자리 건배사, 자녀와의 대화, 등산 동호회 뒷풀이, 마트에서 장볼 때, 부하직원 격려. Don't 상황: 엄숙한 임원 보고, 중요 계약 미팅, 결재 서류 작성, 장례식장. (주의: 학교나 교수님 상황은 절대 생성 금지)",
            "ending": sidebar_msg
        }
    elif target_key == "대한외국인":
        return {
            "prompt": """
            역할: 한국어 잘하는 외국인 친구.
            [CRITICAL] Translate EVERY sentence into English in parentheses.
            Example: 진짜? (Really?)
            """,
            "filter": {"is_safe_foreigner": True},
            "meta_key": "level_foreigner",
            "scenario_guide": "Do Situation: Chatting with Korean friends, ordering food at a restaurant, asking directions, Convenience store. Don't Situation: Speaking to elders, Formal Business meetings.",
            "ending": "👈🏻 Check the sidebar! (사이드바 확인해!)"
        }
    else: # 침착맨/복학생
        return {
            "prompt": "나른하고 논리적인데 킹받는 침착맨 이말년 말투. '~함', '~임'체 사용.",
            "filter": None, 
            "meta_key": "level_student",
            "scenario_guide": "Do 상황: 동아리방, 인스타 댓글, 친구랑 카톡, 편의점 알바 중, PC방. Don't 상황: 교수님 면담, 예비군 훈련장 조교한테 개기기, 상견례, 엄숙한 장례식장.",
            "ending": sidebar_msg
        }

# [핵심] 통합 생성 함수 (v8.6 Final)
def generate_meme_response(user_query, target_persona_key):
    if not vector_db: return "DB Error", ""

    # 0. Intent Detection
    strong_trend_triggers = [
        "최신 밈", "최신 유행", "최신 신조어", "요즘 밈", "요즘 유행", "요즘 신조어",
        "트렌드 알려줘", "트렌드 밈", "유행하는 거", "인싸 용어", "인싸 밈",
        "최신 밈이 궁금해요", "최신 밈 알려주세요", "요즘 뭐 유행해", "완전 최신"
    ]
    trend_words = ["최신", "요즘", "유행", "트렌드", "신조어"]
    request_words = ["알려줘", "추천", "뭐야", "뭐 있어", "소개", "알고 싶어", "궁금"]
    
    is_strong_trigger = any(t in user_query for t in strong_trend_triggers)
    is_weak_trigger = any(t in user_query for t in trend_words) and any(r in user_query for r in request_words)
    is_curation_request = is_strong_trigger or is_weak_trigger

    # 1. Track A: 큐레이션
    if is_curation_request:
        try:
            filtered_df = raw_df[
                raw_df['ExpectedQueries'].str.contains('완전최신', na=False) | 
                raw_df['Meme'].str.contains('완전최신', na=False)
            ]
            if len(filtered_df) < 3: selected_memes = filtered_df
            else: selected_memes = filtered_df.sample(n=3)
            
            if selected_memes.empty: return "아직 최신 트렌드 데이터가 충분하지 않아요! 😅", "데이터 부족"

            meme_info_text = ""
            for _, row in selected_memes.iterrows():
                meme_info_text += f"이름: {row['Meme']}\n의미: {row['Meaning']}\n예시: {row['Example']}\n\n"
            
            curator_persona = ""
            output_instruction = ""
            closing_guide = ""
            
            if target_persona_key == "이시대의 낀세대":
                curator_persona = "친절하고 유쾌한 부장님 톤"
                closing_guide = "허허, 이 정도만 알아도 다음 회식 때 '센스쟁이' 소리 듣습니다. 연습해두세요! 화이팅!"
            elif target_persona_key == "대한외국인":
                curator_persona = "Excited Trend Setter Tone"
                output_instruction = """
                [CRITICAL RULE for Foreigner Mode]
                - You MUST write the content in **Korean first**, followed by the **English translation in parentheses**.
                - Format: "Korean Sentence. (English Translation.)"
                """
                closing_guide = "Wow! Korean trend is so fast! 이거 외워서 우리 같이 '인싸' 됩시다! (Let's memorize this and become an 'Insider' together!)"
            else:
                curator_persona = "귀찮지만 팩트만 말하는 침착맨(이말년) 톤"
                closing_guide = "뭐... 이거 3개만 알면 어디 가서 '화석' 취급은 안 받음. 아님 말고."

            curator_prompt = ChatPromptTemplate.from_template("""
            당신은 MZ 트렌드 큐레이터입니다. 아래 3개의 최신 밈 정보를 바탕으로 매거진 스타일의 리포트를 작성하세요.
            
            [페르소나]: {persona}
            {translation_instruction}
            
            [작성 양식]
            🚀 [MZ 트렌드 속보]
            
            ### 1. (밈 이름)
               - 🧐 핵심: (한 줄 요약)
               - 🗣 실전: (대화 예시)
            
            ### 2. (밈 이름)
               ...
            
            ### 3. (밈 이름)
               ...
            
            **[Closing]**:
            (주의: '[Closing]' 라벨 출력 금지)
            가이드: "{closing_guide}"
            
            [밈 정보]:
            {meme_info}
            """)
            
            chain = curator_prompt | llm | StrOutputParser()
            response = chain.invoke({
                "persona": curator_persona,
                "translation_instruction": output_instruction,
                "meme_info": meme_info_text,
                "closing_guide": closing_guide
            })
            return response, "🔥 최신 트렌드"
        except Exception as e:
            return f"큐레이션 오류: {str(e)}", "Error"

    # 2. Track B: 일반 검색
    query_expander = ChatPromptTemplate.from_template(
        """
        질문에서 사용자가 알고 싶어하는 '밈 단어'를 딱 1개만 추출해.
        [규칙]
        1. 문장, 조사(은/는/이/가/을/를/이/가), 문장부호 절대 금지.
        2. 오직 '명사' 형태의 단어 하나만 출력.
        3. 만약 '캘박이' 처럼 조사가 붙어있으면 '캘박'만 출력.
        
        예시:
        "캘박이 뭐야?" -> 캘박
        "중꺾마 뜻" -> 중꺾마
        "농협은행이 뭐임" -> 농협은행
        "느좋은 무슨 줄임말이야" -> 느좋
        
        질문: {q}
        출력:"""
    ) | llm | StrOutputParser()
    
    expanded_keyword = query_expander.invoke({"q": user_query}).strip()
    
    if " " in expanded_keyword:
        expanded_keyword = expanded_keyword.split()[0]
    if len(expanded_keyword) > 8:
        expanded_keyword = expanded_keyword[:8]

    search_query = expanded_keyword if expanded_keyword else user_query
    
    settings = get_persona_settings(target_persona_key)
    filter_condition = settings.get("filter")
    
    search_kwargs = {'k': 6, 'fetch_k': 20, 'lambda_mult': 0.6}
    if filter_condition: search_kwargs['filter'] = filter_condition

    chroma_retriever = vector_db.as_retriever(search_type="mmr", search_kwargs=search_kwargs)
    
    if bm25_retriever:
        bm25_retriever.k = 6
        ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, chroma_retriever], weights=[0.3, 0.7])
        docs = ensemble_retriever.invoke(search_query)
    else:
        docs = chroma_retriever.invoke(search_query)

    traffic_light_icon = "🟢"
    final_keyword = expanded_keyword
    
    if docs:
        top_doc = docs[0]
        final_keyword = top_doc.metadata.get("meme", expanded_keyword)
        meta_key = settings.get("meta_key", "level_manager")
        level = top_doc.metadata.get(meta_key, "Safe")
        if level == "Danger": traffic_light_icon = "🔴"
        elif level == "Caution": traffic_light_icon = "🟡"

    final_input_query = user_query
    if target_persona_key == "대한외국인":
        final_input_query += " (모든 문장 영어 번역 필수)"

    qa_prompt = ChatPromptTemplate.from_template("""
    당신은 대한민국 최고의 MZ 밈 전문가입니다. [문서 내용]을 참고하여 답변하세요.
    정보가 없으면 "데이터가 없습니다."라고만 하세요.

    [작성 절대 규칙]
    1. **의미와 유래**: 
       - [매우 중요] 절대 '의미와 유래:' 같은 제목을 출력하지 마세요.
       - 바로 🧐 이모지로 시작하여 의미를 설명하세요.
       - [매우 중요] 설명 내에 마크다운 볼드체(**)를 절대 사용하지 마세요.
    
    2. **외국인 모드**: 페르소나가 '대한외국인'이면:
       - 본문 설명뿐만 아니라 **DO/DON'T의 대사(Dialogue) 부분도 반드시 괄호 안에 영어 번역을 넣으세요.**
       - 예시: "안녕하세요! (Hello!)"
    
    3. **가이드 작성법 (필수 포함)**: 
       - **[중요] 답변에 Do 섹션과 Don't 섹션을 무조건 포함해야 합니다. 생략 금지.**
       - `scenario_guide` 중 하나를 골라 상황을 채워넣으세요.
       - **[경고] Do 와 Don't 뒤에는 반드시 콜론(:)을 사용해 상황과 대사를 구분하세요.**
       - 형식 예시: 
         **Do : 회식 자리 : "부장님, 오늘 텐션 저세상이네요!"**
         **Don't : 임원 보고 중 : "상무님, 텐션 무엇?"**
    
    4. **원포인트 코칭**: {target_traffic_light} 아이콘 사용. 구체적 행동 가이드 제시.

    5. **[매우 중요] 내부 처리용 키워드 출력**:
       - 답변의 맨 마지막 줄에 **반드시** `[SearchKeyword]: (설명한_밈_이름_단어)` 를 적어주세요.

    [상황 가이드]: {scenario_guide}
    [페르소나]: {persona_info}
    [문서 내용]: {context}
    [질문]: {user_input}
    """)

    chain = qa_prompt | llm | StrOutputParser()
    full_res = chain.invoke({
        "context": docs, 
        "user_input": final_input_query,
        "persona_info": settings["prompt"],
        "scenario_guide": settings["scenario_guide"],
        "target_traffic_light": traffic_light_icon 
    })
    
    extracted_keyword = ""
    clean_res = full_res
    
    if "[SearchKeyword]:" in full_res:
        parts = full_res.split("[SearchKeyword]:")
        clean_res = parts[0].strip()
        extracted_keyword = parts[1].strip()
    
    if extracted_keyword:
        final_keyword = extracted_keyword.split(',')[0].strip()

    if "데이터가 없습니다" in clean_res or "Data not found" in clean_res:
        fallback_triggers = ["최신", "요즘", "유행", "트렌드"]
        if any(t in user_query for t in fallback_triggers):
            clean_res = "찾으시는 밈 데이터가 없네요. 혹시 **'최신 밈 알려줘'** 라고 물어보시면 요즘 뜨는 걸 알려드릴게요! 😉"
        else:
            clean_res = "데이터가 없습니다."
    
    response_for_batch = clean_res.replace(settings["ending"], "").strip()

    return response_for_batch, final_keyword

# --- 5. 사이드바 ---
with st.sidebar:
    try:
        st.image("logo.png", use_container_width=True)
    except:
        st.markdown("## 🚀 MZ 밈 통역기")
    
    st.markdown("---")
    st.header("👤 통역 스타일 선택")
    
    persona_map = {
        "👨‍💼 부장님 모드": "이시대의 낀세대",
        "👱 외국인 친구": "대한외국인",
        "😎 복학생 모드": "침착맨 스타일"
    }
    
    selected_label = st.radio(
        "통역 스타일 선택", 
        options=list(persona_map.keys()),
        index=0,
        label_visibility="collapsed"
    )
    target = persona_map[selected_label]
    
    st.markdown("---")
    st.subheader("🚦 안전도 가이드")
    st.markdown("""
    <div style="font-size: 0.85rem;">
        <div style="margin-bottom: 6px; display:flex; align-items:center;">
            <span style="font-size:1.2em; margin-right:6px;">🟢</span> 
            <b>전체이용가</b> <span style="color:#888; margin-left:6px; font-size:0.9em;">(안심 사용)</span>
        </div>
        <div style="margin-bottom: 6px; display:flex; align-items:center;">
            <span style="font-size:1.2em; margin-right:6px;">🟡</span> 
            <b>주의 필요</b> <span style="color:#888; margin-left:6px; font-size:0.9em;">(친한 사이)</span>
        </div>
        <div style="margin-bottom: 6px; display:flex; align-items:center;">
            <span style="font-size:1.2em; margin-right:6px;">🔴</span> 
            <b>사용 위험</b> <span style="color:#888; margin-left:6px; font-size:0.9em;">(속으로만)</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # [v8.4] 사용자 요청으로 배치 테스트 기능 복구 (관리자용)
    with st.expander("🔐 관리자 전용 (Batch Test)", expanded=False):
        uploaded_file = st.file_uploader("엑셀/CSV 업로드", type=["xlsx", "csv"])
        if uploaded_file and st.button("🚀 실행 시작"):
            try:
                if uploaded_file.name.endswith('.csv'):
                    df_test = pd.read_csv(uploaded_file)
                else:
                    df_test = pd.read_excel(uploaded_file)
                
                q_col = next((c for c in df_test.columns if '질문' in str(c) or 'Question' in str(c)), None)
                if not q_col:
                    st.error("파일에 '질문' 또는 'Question' 칼럼이 있어야 합니다.")
                else:
                    st.info(f"총 {len(df_test)}개의 데이터를 처리합니다.")
                    progress_bar = st.progress(0)
                    results = []
                    keywords = []
                    
                    for idx, row in df_test.iterrows():
                        q = str(row[q_col])
                        p_col = next((c for c in df_test.columns if '페르소나' in str(c) or 'Persona' in str(c)), None)
                        p_key = row[p_col] if p_col else target
                        
                        if not q or q.lower() == 'nan':
                            results.append("")
                            keywords.append("")
                            continue

                        res, kw = generate_meme_response(q, p_key)
                        results.append(res)
                        keywords.append(kw)
                        progress_bar.progress((idx + 1) / len(df_test))
                    
                    df_test['생성된 답변'] = results
                    df_test['검색된 키워드'] = keywords
                    
                    def to_excel(df):
                        output = BytesIO()
                        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                            df.to_excel(writer, index=False)
                        return output.getvalue()
                    
                    st.success("완료!")
                    st.download_button(
                        label="📥 결과 다운로드",
                        data=to_excel(df_test),
                        file_name='meme_batch_result.xlsx',
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                    )
            except Exception as e:
                st.error(f"오류 발생: {e}")

    st.markdown("---")

    if st.session_state.messages:
        curr = st.session_state.messages[0]
        # [v8.6 Fix] 상태 판별 및 키워드 로직 보완
        is_unknown = "데이터가 없습니다" in curr['ai'] or "Data not found" in curr['ai']
        is_trend = "🚀" in curr['ai'] or "트렌드 속보" in curr['ai']
        
        kw = curr.get('keyword', '')
        
        if is_unknown:
            st.header(f"📍 관련 정보") 
            st.info("아래 버튼을 눌러 직접 검색해보세요!") 
            search_query = curr['user']
            google_url = f"https://www.google.com/search?q={search_query}+밈+뜻"
            youtube_url = f"https://www.youtube.com/results?search_query={search_query}+밈+원본+영상"
        elif is_trend:
            st.header(f"📍 관련 정보")
            st.info("아래 버튼을 눌러 검색해보세요!")
            search_query = "최신 MZ 밈 트렌드"
            google_url = f"https://www.google.com/search?q={search_query}"
            youtube_url = f"https://www.youtube.com/results?search_query={search_query}"
        else:
            # 키워드 정제
            clean_kw = re.sub(r'\(.*?\)', '', kw).strip()
            # [v8.6 Fix] 조사 제거 (은,는,이,가,을,를,의 등)
            clean_kw = re.sub(r'(은|는|이|가|을|를|의|과|와|야|이여|여)$', '', clean_kw).strip()
            
            # [v8.6 Fix] 키워드가 없거나 공백이 포함된 경우 처리
            if not clean_kw:
                clean_kw = curr['user'].split()[0]
            elif " " in clean_kw:
                clean_kw = clean_kw.split()[0]
            
            # 너무 길면 자르기 (UI 깨짐 방지)
            if len(clean_kw) > 10:
                display_kw = clean_kw[:8] + ".."
            else:
                display_kw = clean_kw
            
            st.header(f"📍 '{display_kw}' 더 보기")
            google_url = f"https://www.google.com/search?q={clean_kw}+밈+뜻"
            youtube_url = f"https://www.youtube.com/results?search_query={clean_kw}+밈+원본+영상"

        st.link_button("🔍 구글 검색", google_url)
        st.link_button("▶️ 유튜브 검색", youtube_url)
        
        st.markdown(
            '<div style="color:#999999; font-size:11px; margin-top:-5px; margin-bottom:15px; text-align:right;">'
            '⚠️검색 성능이 아직 부족할 수 있어요😓'
            '</div>', 
            unsafe_allow_html=True
        )
    else:
        st.header("📍 관련 정보")
        st.caption("검색 결과가 여기에 표시됩니다.")

# --- 6. 메인 로직 ---
st.title("👨‍💼 어서와 MZ는 처음이지")

scenario = st.text_input("궁금한 밈이나 상황을 입력하세요 👇", key="main_search")

if scenario and scenario != st.session_state.last_processed_query:
    if vector_db:
        with st.spinner("해독 중..."):
            final_response, final_keyword_for_sidebar = generate_meme_response(scenario, target)
            
            settings = get_persona_settings(target)
            
            if "🚀" in final_response:
                display_response = final_response
            else:
                display_response = final_response + "\n\n" + settings["ending"]

            st.session_state.messages.insert(0, {
                "user": scenario, 
                "ai": display_response, 
                "keyword": final_keyword_for_sidebar 
            })
            st.session_state.last_processed_query = scenario
            st.rerun()

# [화면 출력]
if not st.session_state.messages:
    onboarding_html_list = [
        '<div class="onboarding-container">',
        '<h3 class="onboarding-title">👋 검색한 결과는 이렇게 해석하세요!</h3>',
        '<div class="section-header">🧐 밈의 뜻과 유래</div>',
        '<div class="onboarding-text">해당 밈이 무슨 뜻인지, 어디서 왔는지 요약해서 알려드려요.</div>',
        '<br>',
        '<div class="guide-box do-box">',
        '    <span class="badge do-badge">DO</span>',
        '    <span class="situation">사용해도 되는 경우</span><br>',
        '    <div class="dialogue" style="color:#28a745; font-weight:bold;">"상황에 맞는 적절한 사용 예시를 보여드려요."</div>',
        '</div>',
        '<div class="guide-box dont-box">',
        '    <span class="badge dont-badge">DON\'T</span>',
        '    <span class="situation">사용하면 안 되는 경우</span><br>',
        '    <div class="dialogue" style="color:#ea4335; font-weight:bold;">"분위기 싸해지는 오남용 예시를 경고해드려요."</div>',
        '</div>',
        '<br>',
        '<div class="section-header">💡 원포인트 코칭 (사용 가능 여부 판독 신호등)</div>',
        '<div class="onboarding-text">',
        '    🟢 <b>전체이용가</b> : 남녀노소 누구나 사용 가능한 전국민 밈<br>',
        '    🟡 <b>주의 필요</b> : 친구나 가까운 사이에서만 권장하는 밈<br>',
        '    🔴 <b>사용 위험</b> : 특정 커뮤니티 용어거나 오해의 소지가 있는 밈',
        '</div>',
        '<div class="onboarding-tip">',
        '    👈🏻 <b>Tip:</b> 더 자세한 정보나 영상이 궁금하다면 왼쪽의 <b>[관련 정보]</b> 버튼을 눌러보세요!',
        '</div>',
        '</div>'
    ]
    st.markdown("".join(onboarding_html_list), unsafe_allow_html=True)
else:
    st.subheader("💬 통역 히스토리")
    for msg in st.session_state.messages:
        html_res = convert_st_to_html(msg['ai'])
        st.markdown(f"""
        <div class="chat-wrap">
            <div class="user-bubble">📌 {msg['user']}</div>
            <div class="ai-bubble">{html_res}</div>
        </div>
        """, unsafe_allow_html=True)
