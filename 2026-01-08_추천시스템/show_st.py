# 파일명: show_st.py
import os

# [핵심 1] GPU 강제 비활성화 (Mac 충돌 방지 필수)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

# autoint.py 파일 체크
try:
    from autoint import AutoIntModel
except ImportError:
    st.error("❌ 'autoint.py' 파일이 없습니다.")
    st.stop()

st.set_page_config(page_title="영화 추천 시스템", page_icon="🎬", layout="wide")

# 1. 정적 데이터 로드
@st.cache_data
def load_static_data():
    project_path = os.path.abspath(os.getcwd())
    data_path = os.path.join(project_path, 'data')
    
    required_files = ['field_dims.npy', 'label_encoders.pkl']
    for f in required_files:
        if not os.path.exists(os.path.join(data_path, f)):
            st.error(f"❌ 필수 파일 없음: {f}")
            st.stop()

    try:
        field_dims = np.load(os.path.join(data_path, 'field_dims.npy'))
        ratings_df = pd.read_csv(os.path.join(data_path, 'ml-1m', 'ratings_prepro.csv'))
        movies_df = pd.read_csv(os.path.join(data_path, 'ml-1m', 'movies_prepro.csv'))
        users_df = pd.read_csv(os.path.join(data_path, 'ml-1m', 'users_prepro.csv'))
        label_encoders = joblib.load(os.path.join(data_path, 'label_encoders.pkl'))
        
        return field_dims, ratings_df, movies_df, users_df, label_encoders, project_path
    except Exception as e:
        st.error(f"데이터 로드 에러: {e}")
        st.stop()

# 2. 모델 로드 (Fresh Load)
def load_model_fresh(field_dims, project_path):
    tf.keras.backend.clear_session()
    model_path = os.path.join(project_path, 'model')
    
    embed_dim = 16
    dropout = 0.4
    
    try:
        # 모델 생성
        model = AutoIntModel(field_dims, embed_dim, att_layer_num=3, att_head_num=2, att_res=True,
                             l2_reg_dnn=0, l2_reg_embedding=1e-5, dnn_use_bn=False, dnn_dropout=dropout, init_std=0.0001)
        
        # 더미 데이터로 빌드 (int64)
        dummy_input = tf.zeros((1, len(field_dims)), dtype=tf.int64)
        model(dummy_input)
        
    except Exception as e:
        st.error(f"모델 빌드 실패: {e}")
        st.stop()

    # 가중치 파일 찾기
    weights_candidates = [
        os.path.join(model_path, 'autoInt_model_weights.weights.h5'),
        os.path.join(model_path, 'autoInt_model_weights.h5')
    ]
    
    weights_path = None
    for w in weights_candidates:
        if os.path.exists(w):
            weights_path = w
            break
            
    if weights_path:
        try:
            model.load_weights(weights_path)
        except Exception as e:
            st.error(f"가중치 로드 실패: {e}")
            st.stop()
    else:
        st.error("❌ 가중치 파일 없음")
        st.stop()
        
    return model

# 3. 추천 로직 (★ 핵심 수정 적용됨 ★)
def get_recommendations(user_id, r_year, r_month, users_df, movies_df, ratings_df, model, label_encoders):
    status_placeholder = st.empty() # 진행상황 표시용
    
    # (1) 필터링
    status_placeholder.text("🔍 1/4. 후보 영화 추리는 중...")
    seen_movies = set(ratings_df[ratings_df['user_id'] == user_id]['movie_id'].unique())
    all_movies = set(movies_df['movie_id'].unique())
    candidate_movies = list(all_movies - seen_movies)
    
    if not candidate_movies: return pd.DataFrame()

    # 샘플링 (50개)
    sample_size = 50 
    if len(candidate_movies) > sample_size:
        candidate_movies = np.random.choice(candidate_movies, sample_size, replace=False)

    # (2) 데이터 준비
    status_placeholder.text("📊 2/4. 입력 데이터 만드는 중...")
    candidates_df = pd.DataFrame({'movie_id': candidate_movies})
    candidates_df['user_id'] = user_id
    
    merge_data = pd.merge(candidates_df, movies_df, on='movie_id', how='left')
    merge_data = pd.merge(merge_data, users_df, on='user_id', how='left')
    
    # 피처 생성
    r_year = int(r_year)
    r_month = int(r_month)
    merge_data['rating_year'] = r_year
    merge_data['rating_month'] = r_month
    merge_data['rating_decade'] = str(r_year - (r_year % 10)) + 's'
    merge_data.fillna('no', inplace=True)
    
    # 컬럼 정렬 & 인코딩
    input_cols = ['user_id', 'movie_id','movie_decade', 'movie_year', 'rating_year', 'rating_month', 'rating_decade', 'genre1','genre2', 'genre3', 'gender', 'age', 'occupation', 'zip']
    merge_data = merge_data[input_cols]
    
    for col, encoder in label_encoders.items():
        if col not in merge_data.columns: continue
        target_type = type(encoder.classes_[0])
        merge_data[col] = merge_data[col].astype(target_type)
        mask = merge_data[col].isin(encoder.classes_)
        merge_data.loc[mask, col] = encoder.transform(merge_data.loc[mask, col])
        merge_data.loc[~mask, col] = 0 
        
    # (3) 예측 (★ 여기가 핵심 수정입니다 ★)
    status_placeholder.text("🤖 3/4. AI 예측 시작 (Direct Call)...")
    
    # 텐서 변환
    input_tensor = tf.convert_to_tensor(merge_data.values, dtype=tf.int64)
    
    # [수정] model.predict() -> model() 직접 호출
    # 이 방식은 Keras API 오버헤드를 건너뛰어 Mac에서의 멈춤 현상을 방지합니다.
    pred_scores = model(input_tensor, training=False)
    
    # numpy()로 변환 후 1차원 배열로 평탄화
    pred_scores = pred_scores.numpy().flatten()
    
    status_placeholder.text("✅ 4/4. 결과 정리 중...")
    
    # (4) 결과 정리
    candidates_df['score'] = pred_scores
    top_10 = candidates_df.sort_values('score', ascending=False).head(10)
    
    result = pd.merge(top_10, movies_df, on='movie_id')
    status_placeholder.empty() # 텍스트 제거
    return result[['title', 'genre1', 'movie_year', 'score']]

# --- 메인 실행 ---

# 데이터 로드
field_dims, ratings_df, movies_df, users_df, label_encoders, p_path = load_static_data()

# 모델 로드
model = load_model_fresh(field_dims, p_path)

# [사이드바]
with st.sidebar:
    st.header("📝 정보 입력")
    min_uid, max_uid = int(users_df['user_id'].min()), int(users_df['user_id'].max())
    user_id = st.number_input("사용자 ID", min_value=min_uid, max_value=max_uid, value=min_uid)
    
    years = sorted(list(label_encoders['rating_year'].classes_)) if 'rating_year' in label_encoders else [2000]
    months = sorted(list(label_encoders['rating_month'].classes_)) if 'rating_month' in label_encoders else [1]
    
    r_year = st.selectbox("관람 연도", years, index=len(years)-1)
    r_month = st.selectbox("관람 월", months, index=0)

    run_btn = st.button("🚀 추천 결과 보기", type="primary")

# [메인 화면]
st.title("🎬 AutoInt 영화 추천 서비스")

st.subheader("👤 사용자 정보")
u_info = users_df[users_df['user_id'] == user_id]
st.table(u_info)

st.subheader("👀 최근 재미있게 본 영화 (4점↑)")
user_history = ratings_df[(ratings_df['user_id'] == user_id) & (ratings_df['rating'] >= 4)]

if not user_history.empty:
    history_view = pd.merge(user_history, movies_df, on='movie_id').tail(5)
    display_data = history_view[['title', 'genre1', 'movie_year', 'rating']].copy()
    display_data['rating'] = display_data['rating'].apply(lambda x: f"⭐ {x}")
    display_data.columns = ['제목', '장르', '개봉년도', '평점']
    st.table(display_data)
else:
    st.info("이력이 없습니다.")

st.divider()

if run_btn:
    st.subheader("🤖 AI 맞춤 추천 Top 10")
    
    try:
        recom_df = get_recommendations(user_id, r_year, r_month, users_df, movies_df, ratings_df, model, label_encoders)
        
        if not recom_df.empty:
            recom_df['score'] = recom_df['score'].apply(lambda x: f"{x:.2f}")
            recom_df.columns = ['제목', '장르', '개봉년도', '예측점수']
            st.table(recom_df)
        else:
            st.warning("추천할 영화가 없습니다.")
            
    except Exception as e:
        st.error(f"에러 발생: {e}")