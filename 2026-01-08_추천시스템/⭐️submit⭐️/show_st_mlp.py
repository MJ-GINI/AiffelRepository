# 파일명: show_st_mlp.py
import os
import sys

# [핵심 1] 현재 실행 중인 파일의 폴더 경로를 파이썬 검색 경로에 강제 추가
# 이 코드가 있으면 터미널 위치와 상관없이 옆에 있는 autoint_mlp.py를 잘 찾아냅니다.
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# [핵심 2] GPU 강제 비활성화 (Mac 충돌 방지)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

# autoint_mlp.py 파일 불러오기
try:
    from autoint_mlp import AutoIntMLPModel
except ImportError:
    # 경로를 추가했는데도 안 되면 정말 파일이 없는 것입니다.
    st.error(f"❌ 'autoint_mlp.py' 파일을 찾을 수 없습니다.\n현재 경로: {current_dir}")
    st.stop()

# 페이지 설정
st.set_page_config(page_title="영화 추천 시스템 (AutoInt+)", page_icon="🎬", layout="wide")

# 1. 정적 데이터 로드
@st.cache_data
def load_static_data():
    project_path = os.path.dirname(os.path.abspath(__file__)) # 현재 파일 기준 경로
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

# 2. 모델 로드 (MLP 모델)
def load_model_fresh(field_dims, project_path):
    tf.keras.backend.clear_session()
    model_path = os.path.join(project_path, 'model')
    
    embed_dim = 16
    dropout = 0.4
    
    try:
        # 모델 뼈대 생성 (autoint_mlp.py와 동일)
        model = AutoIntMLPModel(
            field_dims=field_dims, 
            embedding_size=embed_dim, 
            att_layer_num=3, 
            att_head_num=2, 
            att_res=True,
            dnn_hidden_units=(32, 32),
            dnn_activation='relu',      
            l2_reg_dnn=0, 
            l2_reg_embedding=1e-5, 
            dnn_use_bn=False, 
            dnn_dropout=dropout, 
            init_std=0.0001
        )
        
        # 더미 데이터로 빌드 (뼈대 완성)
        dummy_input = tf.zeros((1, len(field_dims)), dtype=tf.int64)
        model(dummy_input)
        
    except Exception as e:
        st.error(f"모델 빌드 실패: {e}")
        st.stop()

    # [핵심 변경] .h5 대신 .pkl 파일을 로드하여 set_weights로 주입
    # 이 방식은 레이어 이름이 달라도 구조(순서)만 같으면 무조건 성공합니다.
    weights_pkl_path = os.path.join(model_path, 'autoInt_model_weights.pkl')
    
    if os.path.exists(weights_pkl_path):
        try:
            weights_list = joblib.load(weights_pkl_path)
            model.set_weights(weights_list)
            # st.success("✅ 가중치(PKL) 로드 성공!") 
        except Exception as e:
            st.error(f"가중치 주입 실패: {e}\n(모델 구조가 학습 때와 다른지 확인하세요)")
            st.stop()
    else:
        # 혹시 몰라 h5 파일이 있다면 시도하도록 예비 코드 남김 (선택사항)
        weights_h5_path = os.path.join(model_path, 'autoInt_model_weights.weights.h5')
        if os.path.exists(weights_h5_path):
             try:
                model.load_weights(weights_h5_path)
             except:
                st.error(f"❌ .pkl 파일이 없습니다: {weights_pkl_path}")
                st.stop()
        else:
            st.error(f"❌ 가중치 파일(.pkl)을 찾을 수 없습니다: {weights_pkl_path}")
            st.stop()
        
    return model

# 3. 추천 로직
def get_recommendations(user_id, r_year, r_month, users_df, movies_df, ratings_df, model, label_encoders):
    # 진행 상태 표시줄 (빈 공간 확보)
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    status_text.text("🔍 1/4. 후보 영화 추리는 중...")
    progress_bar.progress(25)
    
    seen_movies = set(ratings_df[ratings_df['user_id'] == user_id]['movie_id'].unique())
    all_movies = set(movies_df['movie_id'].unique())
    candidate_movies = list(all_movies - seen_movies)
    
    if not candidate_movies: return pd.DataFrame()

    # 샘플링
    sample_size = 50 
    if len(candidate_movies) > sample_size:
        candidate_movies = np.random.choice(candidate_movies, sample_size, replace=False)

    status_text.text("📊 2/4. 입력 데이터 만드는 중...")
    progress_bar.progress(50)
    
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
    
    input_cols = ['user_id', 'movie_id','movie_decade', 'movie_year', 'rating_year', 'rating_month', 'rating_decade', 'genre1','genre2', 'genre3', 'gender', 'age', 'occupation', 'zip']
    merge_data = merge_data[input_cols]
    
    for col, encoder in label_encoders.items():
        if col not in merge_data.columns: continue
        target_type = type(encoder.classes_[0])
        merge_data[col] = merge_data[col].astype(target_type)
        mask = merge_data[col].isin(encoder.classes_)
        merge_data.loc[mask, col] = encoder.transform(merge_data.loc[mask, col])
        merge_data.loc[~mask, col] = 0 
        
    status_text.text("🤖 3/4. AI 예측 시작 (Direct Call)...")
    progress_bar.progress(75)
    
    input_tensor = tf.convert_to_tensor(merge_data.values, dtype=tf.int64)
    pred_scores = model(input_tensor, training=False)
    pred_scores = pred_scores.numpy().flatten()
    
    status_text.text("✅ 4/4. 완료!")
    progress_bar.progress(100)
    
    candidates_df['score'] = pred_scores
    top_10 = candidates_df.sort_values('score', ascending=False).head(10)
    
    result = pd.merge(top_10, movies_df, on='movie_id')
    
    # UI 정리
    status_text.empty()
    progress_bar.empty()
    
    return result[['title', 'genre1', 'movie_year', 'score']]

# --- 메인 실행 ---

# 데이터 로드
field_dims, ratings_df, movies_df, users_df, label_encoders, p_path = load_static_data()

# 모델 로드
model = load_model_fresh(field_dims, p_path)

# [사이드바] - 입력창이 중복되지 않도록 확실히 분리
with st.sidebar:
    st.header("📝 정보 입력")
    min_uid, max_uid = int(users_df['user_id'].min()), int(users_df['user_id'].max())
    user_id = st.number_input("사용자 ID", min_value=min_uid, max_value=max_uid, value=min_uid)
    
    years = sorted(list(label_encoders['rating_year'].classes_)) if 'rating_year' in label_encoders else [2000]
    months = sorted(list(label_encoders['rating_month'].classes_)) if 'rating_month' in label_encoders else [1]
    
    r_year = st.selectbox("관람 연도", years, index=len(years)-1)
    r_month = st.selectbox("관람 월", months, index=0)

    # 버튼에 유니크 키를 주어 충돌 방지
    run_btn = st.button("🚀 추천 결과 보기", type="primary", key="run_recommendation")

# [메인 화면]
st.title("🎬 AutoInt+ 영화 추천 서비스")

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
