import time
import pickle
import numpy as np
import pandas as pd
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Student Result Predictor",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom Styling: Vertical Form Layout with Shadows & Modern Card Design
st.markdown("""
<style>
    /* Global Background */
    .stApp {
        background: #F8FAFC;
        font-family: 'Inter', sans-serif;
    }
    
    /* Main Container Padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 680px;
    }
    
    /* Card Layout with Shadows */
    .custom-card {
        background-color: #FFFFFF;
        padding: 28px;
        border-radius: 16px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.08), 0 8px 10px -6px rgba(0, 0, 0, 0.04);
        border: 1px solid #E2E8F0;
        margin-bottom: 24px;
    }

    /* Prediction Result Card */
    .result-card {
        background: linear-gradient(135deg, #4F46E5 0%, #3B82F6 100%);
        color: #FFFFFF;
        padding: 30px;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 12px 20px -3px rgba(79, 70, 229, 0.35);
        margin-top: 16px;
    }

    /* Headers */
    .main-title {
        color: #0F172A;
        font-size: 28px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 6px;
    }
    .sub-title {
        color: #64748B;
        font-size: 15px;
        text-align: center;
        margin-bottom: 24px;
    }

    /* Styled Prediction Button */
    div.stButton > button:first-child {
        background: #4F46E5;
        color: #FFFFFF;
        font-size: 16px;
        font-weight: 600;
        padding: 12px 24px;
        border-radius: 10px;
        border: none;
        width: 100%;
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3);
        transition: all 0.2s ease-in-out;
    }
    div.stButton > button:first-child:hover {
        background: #4338CA;
        box-shadow: 0 6px 16px rgba(79, 70, 229, 0.45);
        transform: translateY(-1px);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load the trained KNN model."""
    with open("model.pkl", "rb") as file:
        return pickle.load(file)

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model file (`model.pkl`): {e}")
    st.stop()

# Header Section
st.markdown("<h1 class='main-title'>🎓 Student Result Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Enter the student's scores across all subjects below to calculate the predicted total.</p>", unsafe_allow_html=True)

# Main Form Card (Vertical Stack)
st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
st.subheader("📚 Subject Marks (0 - 100)")

# Vertical Form Inputs
hindi = st.number_input("Hindi", min_value=0, max_value=100, value=75, step=1)
english = st.number_input("English", min_value=0, max_value=100, value=80, step=1)
science = st.number_input("Science", min_value=0, max_value=100, value=85, step=1)
maths = st.number_input("Maths", min_value=0, max_value=100, value=90, step=1)
history = st.number_input("History", min_value=0, max_value=100, value=70, step=1)
geography = st.number_input("Geography", min_value=0, max_value=100, value=78, step=1)

# Format features into a Pandas DataFrame to preserve feature structure & names
feature_dict = {
    "Hindi": hindi,
    "English": english,
    "Science": science,
    "Maths": maths,
    "History": history,
    "Geography": geography
}

# If the model expects specific feature names, align DataFrame columns
if hasattr(model, "feature_names_in_"):
    input_df = pd.DataFrame([feature_dict])[list(model.feature_names_in_)]
else:
    input_df = pd.DataFrame([feature_dict])

st.markdown("<br>", unsafe_allow_html=True)
predict_clicked = st.button("🔮 Predict Total Result")
st.markdown("</div>", unsafe_allow_html=True)

# Execution and Result Handling
if predict_clicked:
    with st.spinner("Calculating result..."):
        time.sleep(0.4)
        try:
            prediction = model.predict(input_df)[0]
            
            # Display Prediction Card
            st.markdown(f"""
            <div class="result-card">
                <p style="font-size: 16px; margin-bottom: 4px; opacity: 0.9;">Predicted Total Score</p>
                <h1 style="font-size: 48px; margin: 0; font-weight: 800;">{prediction}</h1>
                <p style="font-size: 14px; margin-top: 8px; opacity: 0.85;">
                    Average Input Mark: {np.mean(list(feature_dict.values())):.1f} / 100
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Display Verification Confirmation
            st.caption("✅ Prediction processed successfully.")

        except Exception as err:
            st.error(f"Prediction failed: {err}")
