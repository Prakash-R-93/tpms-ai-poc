## TPMS AI Proof of Concept

A machine-learning Proof of Concept (POC) for predicting potential Tire Pressure Monitoring System (TPMS) alerts from EV vehicle sensor data.

**Overview**

This project explores whether vehicle-level sensor information can be used to predict a potential TPMS alert.

The project uses four EV user-profile datasets and trains three classification models:

```text
Logistic Regression
Random Forest
XGBoost
```

The Random Forest model was selected as the final model based on its F1 score.

A Streamlit dashboard allows users to enter vehicle sensor values and receive:
```text
TPMS prediction
TPMS alert probability
AI interpretation of the prediction
Sensor values submitted to the model
```

**Important: This is a machine-learning Proof of Concept and not a production vehicle safety system. Actual tire pressure and the vehicle's TPMS should always be checked directly.**

## 📁Project Structure

```text
tpms-ai-poc/
│
├── dashboard/
│   └── app.py
│
├── data/
│   └── raw/
│       ├── dataset1/
│       │   ├── daily_user.csv
│       │   ├── heavy_user.csv
│       │   ├── moderate_user.csv
│       │   └── rare_user.csv
│       └── README.md
│
├── docs/
│   └── dataset-licenses.md
│
├── models/
│   ├── features.json
│   └── random_forest_model.pkl
│
├── notebooks/
│   └── 01_data_understanding.ipynb
│
├── .gitignore
├── README.md
└── requirements.txt
```

## Dataset

The project uses four EV user-profile datasets:
```text
rare_user.csv
heavy_user.csv
daily_user.csv
moderate_user.csv
```
Each dataset contains 43,800 rows and 12 original columns.

After combining the four datasets:

**Total rows: 175,200**

The original sensor fields include:
```text
SOC
SOH
Charging Cycles
Battery Temperature
Motor RPM
Motor Torque
Motor Temperature
Brake Pad Wear
Charging Voltage
Tire Pressure
DTC
```
The dataset also contains a timestamp column.

Dataset licensing and attribution information is documented in:

docs/dataset-licenses.md

TPMS Target Definition

For this POC, a TPMS alert target was created from the Tire_Pressure value.

TPMS threshold = 25.1


The target is defined as:

TPMS_alert = 1 if Tire_Pressure < 25.1
TPMS_alert = 0 otherwise

Target distribution
Class	Meaning	Rows	Percentage
0	Normal	172,571	98.50%
1	TPMS Alert	2,629	1.50%

This creates a significantly imbalanced classification problem, so precision, recall, and F1 score are considered alongside accuracy.

## Machine Learning Features

The final Random Forest model uses these eight features:
```text
SOC
SOH
Battery_Temp
Motor_RPM
Motor_Torque
Motor_Temp
Brake_Pad_Wear
Charging_Voltage
```
Why is Tire_Pressure not an input?

Tire_Pressure was used to create the TPMS target.

It is intentionally excluded from the model inputs so that the POC attempts to predict a potential TPMS alert from other vehicle sensor information.

DTC was also excluded from the final model because it contains text values such as P0MR, while the selected Random Forest model expects numerical inputs.

## Train/Test Split

The data was divided into:
```text
Training data: 140,160 rows
Testing data : 35,040 rows
```

**A stratified 80/20 train/test split was used to preserve the minority-class proportion.**

train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

## Models Tested

Three classification models were evaluated.

**Logistic Regression**
Accuracy : 54.12%
Precision: 1.60%
Recall   : 48.86%
F1 Score : 3.10%

**Random Forest**
Accuracy : 99.04%
Precision: 84.06%
Recall   : 44.11%
F1 Score : 57.86%

**XGBoost**
Accuracy : 74.46%
Precision: 4.50%
Recall   : 79.28%
F1 Score : 8.52%

## Model Comparison
Model	Accuracy	Precision	Recall	F1 Score
Logistic Regression	54.12%	1.60%	48.86%	3.10%
Random Forest	99.04%	84.06%	44.11%	57.86%
XGBoost	74.46%	4.50%	79.28%	8.52%
Selected Model
Random Forest

The Random Forest model was selected because it achieved the highest F1 score:

**F1 Score: 57.86%**

Its test-set confusion matrix was:

                 Predicted
                 Normal  Alert

Actual Normal     34470     44
Actual Alert        294    232


This means:
```text
True Negatives: 34,470
False Positives: 44
False Negatives: 294
True Positives: 232
```
The model correctly detected:

**232 of 526 actual TPMS alert cases.**

## Dashboard

The project includes a Streamlit dashboard located at:

dashboard/app.py


The dashboard loads the saved Random Forest model and accepts the eight model features as user inputs.

It displays:
```text
🟢 NORMAL
🔴 TPMS ALERT
TPMS Alert Probability
AI Interpretation
Current sensor values sent to the model
```
## Running the Dashboard
1. Install dependencies
From the project root:
pip install -r requirements.txt

2. Start Streamlit
streamlit run dashboard/app.py

3. Open the dashboard
Streamlit will display a local URL in the terminal.

Open that URL in a browser.

**Saved Model**

The final trained model is stored at:
models/random_forest_model.pkl

The corresponding feature list is stored at:
models/features.json

Keeping the feature list separately helps ensure that the dashboard uses the same features as the trained model.

## Notebook

The complete data-understanding and model-development workflow is documented in:
notebooks/01_data_understanding.ipynb


The notebook covers:
```text
Dataset loading
Dataset inspection
Dataset combination
Basic data-quality checks
TPMS target creation
Feature selection
Train/test split
Model training
Model evaluation
Model comparison
Random Forest selection
Model saving
Model verification
Dashboard prediction validation
```
## Limitations

This POC has several important limitations.

1. TPMS target is derived from tire pressure

The target label is created using a fixed tire-pressure threshold rather than an actual TPMS warning signal.

2. Tire pressure is not an input feature

The model attempts to predict the target using other vehicle sensor values. This means its predictions should not be interpreted as direct measurements of tire pressure.

3. Imbalanced dataset

Only approximately 1.5% of observations belong to the TPMS alert class.

Therefore, accuracy alone is not sufficient to evaluate the model.

4. False negatives

The Random Forest model missed 294 of the 526 actual alert cases in the test set.

For a real automotive safety application, this would require significant further investigation and improvement.

5. POC only

The model has not been validated against real-world vehicle data, real TPMS hardware, or safety requirements.

## Future Improvements

Potential next steps include:
```text
Collecting real TPMS warning data
Including actual tire-pressure measurements during inference
Engineering time-series features
Investigating tire-pressure trends rather than individual observations
Tuning the Random Forest decision threshold
Testing additional ensemble models
Cross-validation
Hyperparameter optimization
Explainable AI / feature importance analysis
Monitoring model performance over time
Testing on completely independent vehicle datasets
Building a production-grade inference API
Conclusion
```
**This project demonstrates a complete end-to-end machine-learning POC:**
```text
EV Sensor Data
      ↓
Data Understanding
      ↓
TPMS Target Creation
      ↓
Feature Selection
      ↓
Model Training
      ↓
Model Evaluation
      ↓
Random Forest Selection
      ↓
Saved Model
      ↓
Streamlit Dashboard
      ↓
TPMS Prediction
```

**The final Random Forest model achieved:**

99.04% accuracy

84.06% precision

44.11% recall

57.86% F1 score

on the held-out test set.

The POC successfully integrates the trained model into a working Streamlit dashboard that produces both NORMAL and TPMS ALERT predictions.

**Disclaimer**

This project is intended for educational, experimental, and Proof-of-Concept purposes only.

It must not be used as a substitute for a vehicle manufacturer's TPMS, tire-pressure monitoring equipment, vehicle diagnostics, or professional automotive safety systems.

:::

