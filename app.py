import time
import pickle
import numpy as np
import pandas as pd
import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for UI styling
st.markdown("""
<style>
    /* Main Background & Fonts */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Card Container */
    .metric-card {
        background-color: #ffffff;
        padding: 24px;
        border-radius: 16px;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.08);
        border: 1px solid #e1e8ed;
        text-align: center;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.12);
    }
    
    /* Headers */
    .title-header {
        font-family: 'Inter', sans-serif;
        color: #1E293B;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .subtitle-header {
        color: #64748B;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Predict Button Styling */
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #4F46E5 0%, #7C3AED 100%);
        color: white;
        font-size: 18px;
        font-weight: 600;
        padding: 12px 30px;
        border-radius: 12px;
        border: none;
        width: 100%;
        box-shadow: 0 4px 15px rgba(79, 70, 229, 0.4);
        transition: all 0.3s ease;
    }
    div.stButton > button:first-child:hover {
        background: linear-gradient(90deg, #4338CA 0%, #6D28D9 100%);
        box-shadow: 0 6px 20px rgba(79, 70, 229, 0.6);
        transform: scale(1.02);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load the serialized KNN model."""
    with open("model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading `model.pkl`: {e}")
    st.stop()

# Header Section
st.markdown("<h1 class='title-header'>🎓 Academic Performance Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle-header'>Enter the individual subject marks below to estimate the total score using your KNN model.</p>", unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Model Details")
    st.info("""
    **Model Type:** K-Neighbors Classifier  
    **Algorithm:** KD-Tree  
    **Input Features:** 6 Subjects  
    **Target:** Predicted Total Score  
    """)
    st.markdown("---")
    st.caption("🚀 Built with Streamlit & Scikit-Learn")

# Input Form
st.markdown("### 📝 Input Subject Scores")

col1, col2, col3 = st.columns(3)

with col1:
    hindi = st.number_input("Hindi Marks", min_value=0, max_value=100, value=75, step=1)
    english = st.number_input("English Marks", min_value=0, max_value=100, value=80, step=1)

with col2:
    science = st.number_input("Science Marks", min_value=0, max_value=100, value=85, step=1)
    maths = st.number_input("Maths Marks", min_value=0, max_value=100, value=90, step=1)

with col3:
    history = st.number_input("History Marks", min_value=0, max_value=100, value=70, step=1)
    geography = st.number_input("Geography Marks", min_value=0, max_value=100, value=78, step=1)

st.markdown("<br>", unsafe_allow_html=True)

# Feature alignment matching model's feature_names_in_
features = np.array([[hindi, english, science, maths, history, geography]])

# Prediction Button & Animation Effects
if st.button("🔮 Predict Total Score"):
    # Animated Loading State
    with st.spinner("Processing features and querying KNN model..."):
        time.sleep(0.6)  # Brief delay to enhance effect
        prediction = model.predict(features)[0]

    # Trigger celebration effects
    st.balloons()

    st.markdown("---")
    
    # Results Display
    res_col1, res_col2, res_col3 = st.columns([1, 2, 1])

    with res_col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #64748B; margin-bottom: 8px;">Predicted Total Score</h3>
            <h1 style="color: #4F46E5; font-size: 54px; margin: 0;">{prediction}</h1>
            <p style="color: #10B981; font-weight: 600; margin-top: 8px;">
                Average Score: {np.mean(features):.2f} / 100
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Visual Breakdown Table
    st.markdown("### 📊 Subject Breakdown")
    input_data = pd.DataFrame({
        "Subject": ["Hindi", "English", "Science", "Maths", "History", "Geography"],
        "Score": [hindi, english, science, maths, history, geography]
    })
    st.dataframe(input_data, use_container_width=True, hide_index=True)
