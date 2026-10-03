import json
import joblib
import pandas as pd
import streamlit as st
from sklearn.datasets import load_breast_cancer

st.set_page_config(page_title="Breast Tumor Classifier", page_icon="🔬")

st.title("Breast Tumor Classification Demo")
st.write(
    "An educational machine-learning demonstration using the "
    "scikit-learn breast cancer dataset."
)

st.warning(
    "Not a medical device. This demo is not for diagnosis, screening, "
    "or treatment decisions."
)

model = joblib.load("model.joblib")

with open("feature_names.json", "r") as f:
    feature_names = json.load(f)

data = load_breast_cancer(as_frame=True)
reference = data.frame.drop(columns=["target"])

st.subheader("Enter sample measurements")
st.write(
    "The default values are the dataset medians. Replace them with "
    "measurements if you are testing the interface."
)

inputs = {}

with st.form("prediction_form"):
    for start in range(0, len(feature_names), 2):
        columns = st.columns(2)

        for offset, column in enumerate(columns):
            index = start + offset
            if index >= len(feature_names):
                break

            feature = feature_names[index]
            values = reference[feature]
            low = float(values.min())
            high = float(values.max())
            default = float(values.median())

            step = max((high - low) / 100, 0.01)

            inputs[feature] = column.number_input(
                feature,
                min_value=low,
                max_value=high,
                value=default,
                step=step,
                format="%.4f"
            )

    submitted = st.form_submit_button("Predict")

if submitted:
    input_frame = pd.DataFrame([inputs], columns=feature_names)
    prediction = int(model.predict(input_frame)[0])
    probabilities = model.predict_proba(input_frame)[0]

    malignant_index = list(model.classes_).index(0)
    benign_index = list(model.classes_).index(1)

    st.subheader("Demo result")
    st.write(
        "Predicted class:",
        "**" + ("malignant" if prediction == 0 else "benign") + "**"
    )
    st.metric(
        "Estimated malignant probability",
        f"{probabilities[malignant_index]:.1%}"
    )
    st.metric(
        "Estimated benign probability",
        f"{probabilities[benign_index]:.1%}"
    )

    st.caption(
        "These values are model outputs on a small educational dataset. "
        "They are not a person's real-world medical risk."
    )

with st.expander("About the data and model"):
    st.write(
        "The dataset contains numeric features computed from images of "
        "fine-needle aspirates of breast masses. Target code 0 is malignant; "
        "target code 1 is benign."
    )
    st.write(
        "The model is logistic regression with feature standardization. "
        "It was trained on a training split and evaluated on a held-out test split."
    )
