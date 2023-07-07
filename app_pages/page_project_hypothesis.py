import streamlit as st
import matplotlib.pyplot as plt


def page_project_hypothesis_body():
    st.write("### Project Hypothesis and Validation")

    st.info(
        f"The purpose of this project is to develop a predictive model "
        f"for identifying Powdery Mildew in cherry leaves. "
        f"The hypothesis for this project is based on the assumption "
        f"that machine learning algorithms, "
        f"trained on a dataset of cherry leaf images, can accurately classify "
        f"leaves as either healthy or "
        f"infected with Powdery Mildew based on their visual features. "
        f"The hypothesis is as follows: "
    )

    st.write("### Methodology")

    st.info(
        f"1. ### Data Collection:\n"
        f"A dataset of cherry leaf images was collected, consisting of both "
        f"healthy leaves and leaves infected with Powdery Mildew. \n"
        f"2. ### Data Preprocessing:\n"
        f"The collected images are preprocessed to ensure consistency in "
        f"size, format, and quality. "
        f"This step also involved involved resizing to fit the model. "
        f"Mean and variability of images per label plot were distinguished to "
        f"contrast between powdery mildew-contained and healthy "
        f"cherry leaf images. \n"
        f"3. ### Model Development:\n"
        f"The machine learning model, was a convolutional neural network "
        f"(CNN), which was developed and trained on the preprocessed dataset. "
        f"The model was designed to classify cherry leaves as healthy or "
        f"infected with Powdery Mildew based on their visual features. \n"
        f"4. ### Model Evaluation:\n"
        f"Each version of the nine models was trained on a training set "
        f"and evaluated on a separate test set. The evaluation included "
        f"various metrics such as accuracy, precision, recall, and F1-score "
        f"to assess the model's ability to accurately classify cherry leaves "
        f"as healthy or infected with Powdery Mildew. \n"
        f"5. ### Comparison and Selection:\n"
        f"The performance of each model version was compared, considering "
        f"the evaluation metrics. The best-performing model version "
        f"(version 2), was based on the predefined criteria, "
        f"and will be selected for further analysis. \n"
    )

    st.success(
        f"6. ### Hypothesis Testing:\n"
        f"The performance of the selected model (version 2) achieved a "
        f"significantly higher accuracy 99.88% and a low loss of 0.0041 "
        f"which made this the more favourable hypothesis."
    )

    st.success(
        f"### Outcome \n"
        f"Based on a rigorous evaluation of multiple model versions, "
        f"it is expected that the hypothesis demonstrates a high level of "
        f"accuracy, precision, recall, and F1-score in distinguishing between "
        f"healthy cherry leaves and leaves infected with Powdery Mildew."
    )

    st.write(
        f"  # Significance \n"
        f"The hypothesis confirms, the developed model, with optimized "
        f"hyperparameters and a comprehensive evaluation, could serve as a "
        f"reliable tool for identifying Powdery Mildew in cherry orchards. "
        f"The model's ability to accurately classify cherry leaves based on "
        f"visual cues could aid in early detection and prompt management "
        f"strategies, potentially reducing the spread of the disease and "
        f"minimizing crop losses."
    )

    st.write(
        f"By conducting this hypothesis-driven project with a focus on model "
        f"selection and evaluation, valuable insights can be gained regarding "
        f"the effectiveness of different model variations and the impact of "
        f"hyperparameters on the performance of the classification task."
    )

    st.write(
        f"For additional information, regarding each individual version and "
        f"the process behind each models training, please visit and **read** "
        f"the [Project README file](https://github.com/Huddy2022/milestone-"
        f"project-mildew-detection-in-cherry-leaves/blob/main/README.md)."
    )
