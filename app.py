from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Student Lifestyle GPA Estimator", layout="centered")

# Custom Style
st.markdown(
    """
    <style>
    html, body, [class*="st-"] { font-family: "Aptos", "Segoe UI", sans-serif; }
    .stApp { background: #171411; color: #eee7db; }
    [data-testid="stHeader"] { background: rgba(23, 20, 17, 0); }
    [data-testid="stWidgetLabel"] p, .stMarkdown, label { color: #eee7db !important; }
    [data-testid="stNumberInput"] input { background: #211d18; color: #f7f1e7; border-color: #806343; }
    [data-testid="stNumberInput"] button { color: #ead9bb; background: #382b20; }
    .block-container, [data-testid="stMainBlockContainer"] { width: 100%; max-width: 760px;
                       box-sizing: border-box; margin-left: auto; margin-right: auto;
                       min-height: 100vh; padding-top: 2rem; padding-bottom: 2rem; text-align: left;
                       display: flex; flex-direction: column; align-items: center; justify-content: center; }
    [data-testid="stVerticalBlockBorderWrapper"] { background: #201b16;
        border: 1px solid #55432f; border-radius: 18px; padding: 1.1rem; }
    h1, h2, h3 { color: #d8bc8c !important; font-family: Georgia, "Times New Roman", serif; }
    [data-testid="stButton"] button[kind="primary"] { background: #98764b; color: #fff8ec;
                                                        border: 1px solid #c1a274; font-weight: 700; }
    [data-testid="stButton"] button[kind="primary"]:hover { background: #ad8959; color: #fff8ec; }
    .result { padding: 1rem 1.25rem; border-radius: 16px; background: #292219;
              border: 1px solid #806343; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Student Lifestyle GPA Estimator")
st.write("Enter the average number of hours spent in each activity per day.")

MODEL_PATH = Path(__file__).with_name("model.joblib")
FEATURES = [
    "Study_Hours_Per_Day",
    "Extracurricular_Hours_Per_Day",
    "Sleep_Hours_Per_Day",
    "Social_Hours_Per_Day",
    "Physical_Activity_Hours_Per_Day",
]

try:
    model = joblib.load(MODEL_PATH)
except Exception:
    st.error(
        "The prediction model is unavailable. Make sure model.joblib is beside app.py, "
        "then restart or refresh the app."
    )
    st.stop()

activities = [
    ("study_hours", "Study (hours/day)", 6.0),
    ("extracurricular_hours", "Extracurricular activities (hours/day)", 2.0),
    ("social_hours", "Social activities (hours/day)", 3.0),
    ("physical_hours", "Physical activity (hours/day)", 1.0),
    ("sleep_hours", "Sleep (hours/day)", 8.0),
]

# Calculate all limits before rendering widgets so every value updates together
for key, _, default in activities:
    if key not in st.session_state:
        st.session_state[key] = default

max_hours = {}
hours_used = 0.0
for key, label, default in activities:
    hours_left = max(0.0, round(24.0 - hours_used, 1))
    current_value = float(st.session_state[key])
    st.session_state[key] = min(max(current_value, 0.0), hours_left)
    max_hours[key] = hours_left
    hours_used += float(st.session_state[key])

with st.container(border=True):
    st.markdown("#### Your typical day")
    col1, col2 = st.columns(2, gap="large")
    with col1:
        study = st.number_input("Study (hours/day)", min_value=0.0,
            max_value=max_hours["study_hours"] or 24.0,
            step=0.5, key="study_hours", disabled=max_hours["study_hours"] == 0)
        extracurricular = st.number_input("Extracurricular activities (hours/day)", min_value=0.0, max_value=max_hours["extracurricular_hours"] or 24.0, step=0.5,
            key="extracurricular_hours", disabled=max_hours["extracurricular_hours"] == 0)
    with col2:
        social = st.number_input("Social activities (hours/day)", min_value=0.0,
            max_value=max_hours["social_hours"] or 24.0, step=0.5,
            key="social_hours", disabled=max_hours["social_hours"] == 0)
        physical = st.number_input("Physical activity (hours/day)", min_value=0.0,
            max_value=max_hours["physical_hours"] or 24.0, step=0.5,
            key="physical_hours", disabled=max_hours["physical_hours"] == 0)
    
    sleep = st.number_input("Sleep (hours/day)", min_value=0.0,
        max_value=max_hours["sleep_hours"] or 24.0, step=0.5,
        key="sleep_hours", disabled=max_hours["sleep_hours"] == 0)

    result_area = st.empty()
    submitted = st.button("Estimate GPA", type="primary", use_container_width=True)

values = {
    "study_hours": study,
    "extracurricular_hours": extracurricular,
    "sleep_hours": sleep,
    "social_hours": social,
    "physical_hours": physical,
}

if submitted:
    st.session_state.pop("gpa_estimate", None)
    row = pd.DataFrame([{
        "Study_Hours_Per_Day": values["study_hours"],
        "Extracurricular_Hours_Per_Day": values["extracurricular_hours"],
        "Sleep_Hours_Per_Day": values["sleep_hours"],
        "Social_Hours_Per_Day": values["social_hours"],
        "Physical_Activity_Hours_Per_Day": values["physical_hours"],
    }], columns=FEATURES)

    try:
        estimate = float(model.predict(row)[0])
        if not pd.notna(estimate):
            raise ValueError("The prediction was not a number.")
        st.session_state["gpa_estimate"] = estimate
    except Exception:
        st.error("We couldn’t calculate an estimate. Please check the inputs and try again.")

if "gpa_estimate" in st.session_state:
    with result_area.container():
        st.write(f"Estimated GPA: {st.session_state['gpa_estimate']:.2f}")

# Credits :)
st.markdown(
    """
    <div style="background:#292219; color:#e8d7b5; padding:10px 14px;
                border-radius:8px; border:1px solid #806343;">
        Please don't treat the estimated GPA as guaranteed; it is only an estimate based on patterns in the dataset used to train the model.
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown('<div style="margin-top: 5px;"></div>', unsafe_allow_html=True)
st.caption("Made by Yousif Alhammadi.")