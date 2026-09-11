import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


# ============================================================
# TPMS AI PROOF OF CONCEPT
# ============================================================
# This dashboard uses a trained Random Forest model to estimate
# whether the current vehicle sensor readings indicate a
# potential TPMS alert.
#
# IMPORTANT:
# - This is a Proof of Concept (POC).
# - The model does NOT directly use Tire_Pressure as an input.
# - The target used during training was derived from tire pressure.
# - This should not be treated as a real vehicle safety system.
# ============================================================


# ------------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------------

st.set_page_config(
    page_title="TPMS AI POC",
    page_icon="🚗",
    layout="wide"
)


# ------------------------------------------------------------
# LOAD TRAINED MODEL
# ------------------------------------------------------------
# __file__ points to dashboard/app.py.
# parent.parent takes us to the project root.
#
# Therefore the model path works regardless of the directory
# from which Streamlit is launched.

MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "random_forest_model.pkl"
)

try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    st.error(
        "Random Forest model could not be found.\n\n"
        f"Expected location: {MODEL_PATH}"
    )
    st.stop()


# ------------------------------------------------------------
# MODEL FEATURES
# ------------------------------------------------------------
# These MUST match the features used when the Random Forest
# model was trained.

FEATURES = [
    "SOC",
    "SOH",
    "Battery_Temp",
    "Motor_RPM",
    "Motor_Torque",
    "Motor_Temp",
    "Brake_Pad_Wear",
    "Charging_Voltage"
]


# ------------------------------------------------------------
# DASHBOARD HEADER
# ------------------------------------------------------------

st.title("🚗 TPMS AI Proof of Concept")

st.write(
    "Use vehicle sensor readings to estimate whether the vehicle "
    "may be at risk of a TPMS alert."
)

st.info(
    "This is a machine-learning Proof of Concept. "
    "The prediction is based on vehicle sensor patterns and "
    "should not be considered a replacement for the vehicle's "
    "actual TPMS or a physical tire-pressure check."
)


# ------------------------------------------------------------
# SENSOR INPUTS
# ------------------------------------------------------------

st.header("📊 Vehicle Sensor Readings")

st.caption(
    "Enter the current vehicle sensor values and click "
    "**Predict TPMS Status**."
)


# Use two columns to make the dashboard easier to read.

col1, col2 = st.columns(2)


# ------------------------------------------------------------
# COLUMN 1 — BATTERY / VEHICLE VALUES
# ------------------------------------------------------------

with col1:

    SOC = st.number_input(
        "SOC (%)",
        min_value=0.0,
        max_value=100.0,
        value=50.0,
        step=1.0,
        help="State of Charge of the battery."
    )

    SOH = st.number_input(
        "SOH (%)",
        min_value=0.0,
        max_value=100.0,
        value=95.0,
        step=1.0,
        help="State of Health of the battery."
    )

    Battery_Temp = st.number_input(
        "Battery Temperature (°C)",
        value=30.0,
        step=1.0,
        help="Battery temperature in degrees Celsius."
    )

    Motor_RPM = st.number_input(
        "Motor RPM",
        min_value=0.0,
        value=1000.0,
        step=100.0,
        help="Motor rotational speed."
    )


# ------------------------------------------------------------
# COLUMN 2 — MOTOR / BRAKE / CHARGING VALUES
# ------------------------------------------------------------

with col2:

    Motor_Torque = st.number_input(
        "Motor Torque",
        min_value=0.0,
        value=200.0,
        step=50.0,
        help="Motor torque value."
    )

    Motor_Temp = st.number_input(
        "Motor Temperature (°C)",
        value=70.0,
        step=1.0,
        help="Motor temperature in degrees Celsius."
    )

    Brake_Pad_Wear = st.number_input(
        "Brake Pad Wear (%)",
        min_value=0.0,
        max_value=100.0,
        value=50.0,
        step=1.0,
        help="Estimated brake pad wear percentage."
    )

    Charging_Voltage = st.number_input(
        "Charging Voltage (V)",
        min_value=0.0,
        value=240.0,
        step=10.0,
        help="Vehicle charging voltage."
    )


# ------------------------------------------------------------
# PREDICTION BUTTON
# ------------------------------------------------------------

st.divider()

predict_button = st.button(
    "🔍 Predict TPMS Status",
    type="primary",
    use_container_width=True
)


# ------------------------------------------------------------
# MAKE PREDICTION
# ------------------------------------------------------------

if predict_button:

    # Create a DataFrame containing exactly the eight features
    # expected by the trained Random Forest model.

    input_data = pd.DataFrame([{
        "SOC": SOC,
        "SOH": SOH,
        "Battery_Temp": Battery_Temp,
        "Motor_RPM": Motor_RPM,
        "Motor_Torque": Motor_Torque,
        "Motor_Temp": Motor_Temp,
        "Brake_Pad_Wear": Brake_Pad_Wear,
        "Charging_Voltage": Charging_Voltage
    }])

    # Make sure the feature order is exactly the same as
    # the order used during model training.

    input_data = input_data[FEATURES]


    # --------------------------------------------------------
    # MODEL PREDICTION
    # --------------------------------------------------------

    prediction = model.predict(input_data)[0]

    # Probability of class 1 = TPMS alert.

    probability = model.predict_proba(input_data)[0][1]


    # --------------------------------------------------------
    # DISPLAY PREDICTION
    # --------------------------------------------------------

    st.header("📊 AI Prediction")

    if prediction == 1:

        st.error(
            "🔴 TPMS ALERT",
            icon="🚨"
        )

        st.write(
            "The AI model predicts that the current vehicle "
            "sensor pattern may be associated with a TPMS alert."
        )

    else:

        st.success(
            "🟢 NORMAL",
            icon="✅"
        )

        st.write(
            "The AI model does not detect a strong TPMS alert "
            "pattern in the current sensor readings."
        )


    # --------------------------------------------------------
    # PROBABILITY DISPLAY
    # --------------------------------------------------------

    st.subheader("🎯 TPMS Alert Probability")

    st.metric(
        label="Model Estimated Probability",
        value=f"{probability:.1%}"
    )

    # Visual probability bar.

    st.progress(
        min(max(float(probability), 0.0), 1.0)
    )


    # --------------------------------------------------------
    # AI INTERPRETATION
    # --------------------------------------------------------

    st.subheader("💡 AI Interpretation")

    if prediction == 1:

        st.warning(
            f"The Random Forest model estimates a "
            f"{probability:.1%} probability of the TPMS alert class. "
            "As a precaution, check the vehicle's tire pressure and "
            "TPMS system."
        )

    else:

        st.info(
            f"The Random Forest model estimates a "
            f"{probability:.1%} probability of the TPMS alert class. "
            "The current sensor pattern is classified as normal "
            "by the model."
        )


    # --------------------------------------------------------
    # SENSOR VALUES SENT TO MODEL
    # --------------------------------------------------------

    with st.expander("📋 View Sensor Values Sent to Model"):

        display_data = input_data.T.copy()

        display_data.columns = ["Value"]

        st.dataframe(
            display_data,
            use_container_width=True
        )


# ------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------

st.divider()

st.caption(
    "TPMS AI POC | Random Forest | Machine-learning demonstration only"
)
