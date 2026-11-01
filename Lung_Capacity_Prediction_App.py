
# ============================================================
# LUNG CAPACITY PREDICTION SYSTEM
# DATAQUEST SOLUTIONS
# Python | Streamlit | Scikit-learn
# ============================================================

import streamlit as st
import pandas as pd
import math
import textwrap

from joblib import load
from datetime import datetime
from pathlib import Path


## 1. PAGE CONFIGURATION

st.set_page_config(
    page_title="Lung Capacity Prediction | DataQuest Solutions",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)


## 2. BRAND COLORS

NAVY = "#0B1F3A"
NAVY_LIGHT = "#173B62"
ORANGE = "#F28C28"
WHITE = "#FFFFFF"
LIGHT_BG = "#F4F6FA"
TEXT = "#263445"
MUTED = "#718096"


## 3. HTML RENDERING HELPER

def render_html(content):
    clean_content = "\n".join(
        line.lstrip()
        for line in content.splitlines()
    ).strip()

    st.html(clean_content)


## 4. CUSTOM CSS AND APPLICATION DESIGN

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background-color: #F4F6FA;
    color: #263445;
}

/* MAIN DASHBOARD CONTAINER */

.block-container {
    max-width: 1800px !important;
    width: 100% !important;
    padding-top: 1.6rem !important;
    padding-bottom: 2.5rem !important;
    padding-left: 1.5rem !important;
    padding-right: 1.5rem !important;
}

/* SIDEBAR */

[data-testid="stSidebar"] {
    background-color: #0B1F3A;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
[data-testid="stSidebar"] [data-testid="stWidgetLabel"],
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p {
    color: #FFFFFF !important;
}

[data-testid="stSidebar"] input {
    background-color: #FFFFFF !important;
    color: #0B1F3A !important;
}

[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background-color: #FFFFFF !important;
    color: #0B1F3A !important;
}

[data-testid="stSidebar"] [data-testid="stFormSubmitButton"] button {
    background-color: #F28C28 !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 9px !important;
    font-weight: 700 !important;
}

/* MAIN HEADER */

.hero {
    width: 100%;
    box-sizing: border-box;
    background: linear-gradient(
        120deg,
        #0B1F3A 0%,
        #173B62 100%
    );
    padding: 34px 38px;
    border-radius: 18px;
    color: #FFFFFF;
    margin-top: 2rem;
    margin-bottom: 20px;
    border-left: 7px solid #F28C28;
    box-shadow: 0 8px 24px rgba(11, 31, 58, 0.10);
}

.hero h1 {
    color: #FFFFFF !important;
    font-size: clamp(25px, 3vw, 36px);
    font-weight: 800;
    line-height: 1.2;
    margin: 0 0 12px 0;
}

.hero p {
    color: #DCE6F1 !important;
    font-size: 15px;
    line-height: 1.7;
    margin: 0 0 20px 0;
}

.hero-tag {
    display: inline-block;
    background: #F28C28;
    color: #FFFFFF;
    padding: 7px 13px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 700;
    margin: 0 7px 5px 0;
}

.hero-tag-secondary {
    display: inline-block;
    background: rgba(255,255,255,0.15);
    color: #FFFFFF;
    padding: 7px 13px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    margin-bottom: 5px;
}

/* SECTION TITLES */

.section-title {
    color: #0B1F3A;
    font-size: 20px;
    font-weight: 800;
    margin: 22px 0 14px 0;
}

/* DASHBOARD CARDS */

.info-card {
    background: #FFFFFF;
    border: 1px solid #E7ECF2;
    border-radius: 15px;
    padding: 22px;
    min-height: 125px;
    box-shadow: 0 3px 12px rgba(11, 31, 58, 0.045);
}

.info-card-label {
    color: #718096;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.5px;
    margin-bottom: 9px;
}

.info-card-value {
    color: #0B1F3A;
    font-size: 23px;
    font-weight: 800;
}

.info-card-description {
    color: #718096;
    font-size: 12px;
    margin-top: 6px;
}

/* PATIENT PROFILE */

.profile-detail {
    margin-top: 15px;
    color: #0B1F3A !important;
    line-height: 1.9;
    font-size: 14px;
}

.profile-detail p,
.profile-detail p span {
    color: #F28C28 !important;
    margin: 7px 0;
}

.profile-detail strong,
.profile-detail b {
    color: #0B1F3A !important;
    font-weight: 700;
}

/* Streamlit paragraph override */

.profile-detail .st-emotion-cache-cmmld9,
.profile-detail .st-emotion-cache-cmmld9 p {
    color: #F28C28 !important;
}

/* PREDICTION RESULT CARD */

.result-card {
    background: linear-gradient(
        135deg,
        #0B1F3A,
        #173B62
    );
    border-radius: 18px;
    padding: 28px;
    color: #FFFFFF;
    border-bottom: 5px solid #F28C28;
    box-shadow: 0 8px 22px rgba(11, 31, 58, 0.10);
    min-height: 205px;
}

.result-card h3 {
    color: #FFFFFF !important;
    font-size: 17px;
    font-weight: 600;
    margin: 0;
}

.result-number {
    color: #F28C28;
    font-size: clamp(34px, 4vw, 46px);
    font-weight: 800;
    margin: 15px 0 8px 0;
}

.result-note {
    color: #DCE6F1;
    font-size: 13px;
    line-height: 1.6;
    margin: 0;
}

/* BUTTONS */

.stButton > button,
.stDownloadButton > button {
    border-radius: 9px;
    font-weight: 700;
    min-height: 43px;
}

.stButton > button {
    background-color: #F28C28;
    color: #FFFFFF;
    border: 1px solid #F28C28;
}

.stButton > button:hover {
    background-color: #D97012;
    color: #FFFFFF;
    border-color: #D97012;
}

.stDownloadButton > button {
    background-color: #0B1F3A;
    color: #FFFFFF;
    border: 1px solid #0B1F3A;
}

.stDownloadButton > button:hover {
    background-color: #173B62;
    color: #FFFFFF;
    border-color: #173B62;
}

/* METRICS */

[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #E7ECF2;
    border-radius: 12px;
    padding: 14px;
}

/* METRICS AND PATIENT PROFILE */

[data-testid="stMetric"] {
    background: #FFFFFF !important;
    border: 1px solid #E7ECF2;
    border-left: 4px solid #F28C28;
    border-radius: 12px;
    padding: 14px;
}

/* Patient metric labels */
[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] p {
    color: #0B1F3A !important;
    font-weight: 700 !important;
}

/* Patient metric values */
[data-testid="stMetricValue"],
[data-testid="stMetricValue"] div {
    color: #F28C28 !important;
    font-weight: 800 !important;
}

/* DATA TABLE */

[data-testid="stDataFrame"] {
    border: 1px solid #E7ECF2;
    border-radius: 10px;
    overflow: hidden;
}

/* FOOTER */

.footer-full-width {
    width: calc(100% + 3rem);
    margin-left: -1.5rem !important;
    margin-right: -1.5rem !important;
    margin-bottom: -2.5rem !important;
    box-sizing: border-box;
}

.footer {
    width: 100%;
    box-sizing: border-box;
    background-color: #0B1F3A;
    padding: 47px 18px;
    border-radius: 0 !important;
    text-align: center;
    margin: 0 !important;
}

.footer h3 {
    color: #FFFFFF !important;
    font-size: 19px;
    font-weight: 800;
    margin: 0 0 9px 0;
}

.footer h3 span {
    color: #F28C28;
}

.footer p {
    color: #DCE6F1 !important;
    font-size: 12px;
    margin: 5px 0;
}

.footer .copyright {
    margin-top: 13px;
    color: #AFC0D2 !important;
}

.hero-tag-link {
    text-decoration: none !important;
    display: inline-block;
    cursor: pointer;
}

.hero-tag-secondary {
    cursor: pointer;
    transition: all 0.3s ease;
}

.hero-tag-link:hover .hero-tag-secondary {
    color: #F28C28;
    text-decoration: underline;
}

/* CLINICAL DISCLAIMER */

.clinical-disclaimer {
    background: #0B1F3A !important;
    border-left: 6px solid #F28C28 !important;
    border-radius: 12px;
    padding: 20px 24px;
    margin-top: 15px;
}

.clinical-disclaimer h4,
.clinical-disclaimer h4 span {
    color: #F28C28 !important;
    font-size: 16px;
    font-weight: 800;
    margin: 0 0 10px 0;
}

.clinical-disclaimer p,
.clinical-disclaimer p span,
.clinical-disclaimer div {
    color: #FFFFFF !important;
    font-size: 13px;
    line-height: 1.8;
}

.clinical-disclaimer * {
    color: #FFFFFF !important;
}

.clinical-disclaimer h4 {
    color: #F28C28 !important;
}

/* RESPONSIVE DESIGN */

@media (max-width: 768px) {

    .hero {
        padding: 25px 22px;
    }

    .hero h1 {
        font-size: 25px;
    }

    .info-card {
        padding: 17px;
    }

}

</style>
""", unsafe_allow_html=True)


## 5. LOAD TRAINED MACHINE LEARNING MODEL

@st.cache_resource
def load_prediction_model():

    model_path = (
        Path(__file__).resolve().parent
        / "Best_Regression_Model.joblib"
    )

    return load(model_path)


try:

    model = load_prediction_model()

    model_available = True

    model_error = None

except Exception as error:

    model = None

    model_available = False

    model_error = str(error)


## 6. SIDEBAR: PATIENT INFORMATION

with st.sidebar:

    render_html("""
    <div style="padding:8px 0 4px 0;">

        <h2 style="color:#F28C28 !important; margin-bottom:4px;">
            🫁 LUNG HEALTH
        </h2>

        <p style="color:#DCE6F1 !important; font-size:13px;">
            Patient assessment panel
        </p>

    </div>
    """)

    st.divider()

    st.markdown("### Patient Information")

    with st.form("patient_form"):

        age = st.number_input(
            "Age (Years)",
            min_value=3,
            max_value=25,
            value=12,
            step=1
        )

        height = st.number_input(
            "Height (Inches)",
            min_value=40.0,
            max_value=80.0,
            value=60.0,
            step=0.1
        )

        gender = st.selectbox(
            "Gender",
            ["Female", "Male"]
        )

        smoke = st.selectbox(
            "Smoking Status",
            ["No", "Yes"]
        )

        caesarean = st.selectbox(
            "Caesarean Birth",
            ["No", "Yes"]
        )

        st.divider()

        predict_button = st.form_submit_button(
            "🔍 Predict Lung Capacity",
            use_container_width=True
        )

    st.markdown("---")

    render_html("""
    <div style="text-align:center; padding:8px 0;">

        <p style="
            color:#F28C28 !important;
            font-weight:800;
            letter-spacing:0.5px;">
            DATAQUEST SOLUTIONS
        </p>

        <p style="
            font-size:11px;
            color:#DCE6F1 !important;">
            Health Data Science &amp; AI
        </p>

    </div>
    """)


## 7. MAIN DASHBOARD HEADER

render_html("""
<div class="hero">

    <h1>🫁 Lung Capacity Prediction System</h1>

    <p>
        An intelligent machine learning application designed to
        estimate lung capacity using selected demographic and
        health-related characteristics.
    </p>

    <span class="hero-tag">
        MACHINE LEARNING
    </span>

    <a href="https://www.dataquestsolutions.co.ke/project.html"
   target="_blank"
   class="hero-tag-link">

    <span class="hero-tag-secondary">
        DATAQUEST SOLUTIONS STUDENTS PROJECT
    </span>

</a>

</div>
""")


## 8. MODEL STATUS

if not model_available:

    st.error(
        "The trained model could not be loaded. "
        "Place Best_Regression_Model.joblib in the same folder "
        "as this Python application, then restart Streamlit. "
        f"Technical details: {model_error}"
    )


## 9. DASHBOARD OVERVIEW

render_html("""
<div class="section-title">Dashboard Overview</div>
""")

overview_col1, overview_col2, overview_col3 = st.columns(3)

with overview_col1:

    render_html("""
    <div class="info-card">

        <div class="info-card-label">
            APPLICATION
        </div>

        <div class="info-card-value">
            Lung Health
        </div>

        <div class="info-card-description">
            Predictive analytics system
        </div>

    </div>
    """)


with overview_col2:

    render_html("""
    <div class="info-card">

        <div class="info-card-label">
            MODEL TYPE
        </div>

        <div class="info-card-value">
            Regression
        </div>

        <div class="info-card-description">
            Supervised machine learning
        </div>

    </div>
    """)


with overview_col3:

    status_color = (
        "#16A36A" if model_available else "#D64545"
    )

    status_text = (
        "Active" if model_available else "Unavailable"
    )

    render_html(f"""
    <div class="info-card">

        <div class="info-card-label">
            MODEL STATUS
        </div>

        <div class="info-card-value" style="color:{status_color};">
            {status_text}
        </div>

        <div class="info-card-description">
            Trained model loading status
        </div>

    </div>
    """)


## 10. PATIENT ASSESSMENT SUMMARY

render_html("""
<div class="section-title">Patient Assessment Summary</div>
""")

gender_value = 1 if gender == "Male" else 0

smoke_value = 1 if smoke == "Yes" else 0

caesarean_value = 1 if caesarean == "Yes" else 0


input_data = pd.DataFrame({

    "Age": [age],

    "Height": [height],

    "Smoke": [smoke_value],

    "Gender": [gender_value],

    "Caesarean": [caesarean_value]

})


profile_col1, profile_col2 = st.columns(2)

with profile_col1:

    render_html(f"""
    <div class="info-card">

        <div class="info-card-label">
            DEMOGRAPHIC INFORMATION
        </div>

        <div class="profile-detail">
            <p><strong>Age:</strong> {age} years</p>
            <p><strong>Height:</strong> {height:.1f} inches</p>
            <p><strong>Gender:</strong> {gender}</p>
        </div>

    </div>
    """)


with profile_col2:

    render_html(f"""
    <div class="info-card">

        <div class="info-card-label">
            HEALTH-RELATED INFORMATION
        </div>

        <div class="profile-detail">
            <p><strong>Smoking Status:</strong> {smoke}</p>
            <p><strong>Caesarean Birth:</strong> {caesarean}</p>
        </div>

    </div>
    """)


## 11. PREDICTION ENGINE

if predict_button:

    if model_available:

        try:

            prediction_value = float(
                model.predict(input_data)[0]
            )

            if not math.isfinite(prediction_value):

                raise ValueError(
                    "The model returned an invalid prediction."
                )

            st.session_state["prediction"] = prediction_value

            st.session_state["patient_details"] = {

                "Age": age,

                "Height": height,

                "Gender": gender,

                "Smoking Status": smoke,

                "Caesarean Birth": caesarean

            }

            st.session_state["prediction_time"] = (
                datetime.now().strftime(
                    "%d %B %Y, %H:%M:%S"
                )
            )

        except Exception as error:

            st.error(f"Prediction failed: {error}")


## 12. DISPLAY PREDICTION RESULTS

if (
    "prediction" in st.session_state
    and "patient_details" in st.session_state
):

    prediction = st.session_state["prediction"]

    patient = st.session_state["patient_details"]

    st.markdown("---")

    render_html("""
    <div class="section-title">Prediction Results</div>
    """)

    result_col, patient_col = st.columns([1.5, 1])

    with result_col:

        render_html(f"""
        <div class="result-card">

            <h3>
                🫁 Estimated Lung Capacity
            </h3>

            <div class="result-number">
                {prediction:.3f}
            </div>

            <p class="result-note">
                Machine learning model estimate.
                The measurement unit depends on the target
                variable used to train the model.
            </p>

            <hr style="border-color:#53677D; margin:18px 0;">

            <p class="result-note">
                Prediction generated successfully.
            </p>

        </div>
        """)

    with patient_col:

        render_html("""
        <div class="section-title" style="margin-top:0;">
            Patient Profile
        </div>
        """)

        st.metric(
            "Age",
            f"{patient['Age']} years"
        )

        st.metric(
            "Height",
            f"{patient['Height']:.1f} inches"
        )

        st.metric(
            "Gender",
            patient["Gender"]
        )


    ## 13. PREDICTION REPORT

    render_html("""
    <div class="section-title">Prediction Report</div>
    """)

    report = pd.DataFrame([{

        "Prediction Date": st.session_state["prediction_time"],

        "Age": patient["Age"],

        "Height (Inches)": patient["Height"],

        "Gender": patient["Gender"],

        "Smoking Status": patient["Smoking Status"],

        "Caesarean Birth": patient["Caesarean Birth"],

        "Predicted Lung Capacity": prediction

    }])

    st.dataframe(
        report,
        use_container_width=True,
        hide_index=True
    )

    ## DOWNLOAD REPORT

    csv_data = report.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(

        label="⬇ Download Prediction Report (CSV)",

        data=csv_data,

        file_name="lung_capacity_prediction_report.csv",

        mime="text/csv",

        use_container_width=True

    )


else:

    st.info(
        "Enter the patient information in the sidebar and click "
        "'Predict Lung Capacity' to generate a prediction."
    )


## 14. CLINICAL DISCLAIMER

render_html("""
<div class="clinical-disclaimer">

    <h4>
        ⚕️ Clinical Disclaimer
    </h4>

    <p>
        This application provides a machine learning-based estimate
        for educational and research purposes. It is not a substitute
        for clinical spirometry, medical assessment, or professional
        diagnosis. Interpret predictions within the limitations of
        the training data, model validation and appropriate clinical
        reference standards.
    </p>

</div>
""")

## 15. DATAQUEST SOLUTIONS FOOTER

st.markdown(
    """
    <style>
    [data-testid="stBottomBlockContainer"] {
        padding-bottom: 0 !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)

render_html("""
<div class="footer-full-width">

    <div class="footer">

        <h3>
            DATAQUEST <span>SOLUTIONS</span>
        </h3>

        <p>
            Health Data Science | Machine Learning | Artificial Intelligence
        </p>

        <p>
            Developed using Python, Streamlit and Scikit-learn
        </p>

        <p class="copyright">
            © 2026 DataQuest Solutions. All Rights Reserved.
        </p>

    </div>

</div>
""")
