import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="작물 추천 시스템", layout="centered")

@st.cache_resource
def load_model():
    return joblib.load("v109_crop_model.pkl")  # 파일명 반드시 일치

@st.cache_data
def load_fertilizer_table():
    df = pd.read_csv("fertilizer_table.csv")
    return df.set_index("작물")

model = load_model()
fertilizer_table = load_fertilizer_table()

st.title("토양 분석 기반 작물 추천 시스템")
st.write("토양 성분 데이터를 입력하세요:")

ph = st.number_input("pH", value=6.5)
om = st.number_input("유기물", value=20.0)
cec = st.number_input("치환성양분", value=0.5)
ca = st.number_input("Ca", value=6.0)
mg = st.number_input("Mg", value=2.0)
p_avail = st.number_input("유효인산", value=50.0)
ec = st.number_input("전기전도도", value=0.8)
k = st.number_input("K", value=0.3)
drainage = st.selectbox("배수상태", ["양호", "불량", "보통"])

if "top_classes" not in st.session_state:
    st.session_state.top_classes = []
if "selected_crop" not in st.session_state:
    st.session_state.selected_crop = None

if st.button("작물 추천"):
    try:
        input_data = pd.DataFrame([{
            "pH": ph,
            "유기물": om,
            "치환성양분": cec,
            "Ca": ca,
            "Mg": mg,
            "유효인산": p_avail,
            "전기전도도": ec,
            "K": k,
            "배수상태": drainage
        }])

        pred_probs = model.predict_proba(input_data)[0]
        class_labels = model.classes_
        class_prob_dict = dict(zip(class_labels, pred_probs))
        top3 = sorted(class_prob_dict.items(), key=lambda x: x[1], reverse=True)[:3]

        st.session_state.top_classes = sorted([c[0] for c in top3])
        st.session_state.selected_crop = st.session_state.top_classes[0]

    except Exception as e:
        st.error(f"예측 중 오류 발생: {e}")

selected_crop = None
if st.session_state.top_classes:
    st.success("✅ 추천 작물:")
    selected_crop = st.radio(
        "추천된 작물 중 하나를 선택하세요",
        st.session_state.top_classes,
        index=st.session_state.top_classes.index(
            st.session_state.selected_crop or st.session_state.top_classes[0]
        )
    )
    st.session_state.selected_crop = selected_crop

if st.session_state.selected_crop:
    st.markdown(f"## 🌱 '{st.session_state.selected_crop}'의 표준 시비처방")
    try:
        std_fert = fertilizer_table.loc[st.session_state.selected_crop][["질소(N)", "인산(P₂O₅)", "칼륨(K₂O)"]]
        st.markdown("### 📌 표준 시비처방량 (10a당)")
        st.dataframe(std_fert.to_frame())

        current_levels = {
            "질소(N)": 0,
            "인산(P₂O₅)": int(p_avail / 7),
            "칼륨(K₂O)": int(k / 0.03)
        }
        additional = {
            nutrient: max(0, std_fert[nutrient] - current_levels[nutrient])
            for nutrient in ["질소(N)", "인산(P₂O₅)", "칼륨(K₂O)"]
        }
        add_df = pd.DataFrame.from_dict(additional, orient="index", columns=["추가 필요량 (kg/10a)"])
        st.markdown("### ➕ 추가로 필요한 시비량 (kg/10a 기준):")
        st.dataframe(add_df)

    except Exception as e:
        st.error(f"시비처방 출력 오류: {e}")

st.write("🔥 로딩된 모델 타입:", type(model))
