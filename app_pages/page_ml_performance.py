import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.image import imread
from src.machine_learning.evaluate_clf import load_test_evaluation


def page_ml_performance_metrics():
    version = 'v2'

    st.write("### Train, Validation and Test Set: Labels Frequencies")
    col1, col2 = st.beta_columns(2)
    with col1:
        labels_distribution = plt.imread(
            f"outputs/{version}/labels_distribution.png")
        st.image(labels_distribution,
                 caption='Labels Distribution on Train, '
                         'Validation and Test Sets')
    with col2:
        sets_distribution = plt.imread(
            f"outputs/{version}/sets_distribution_pie.png")
        st.image(sets_distribution,
                 caption='Datasets Distribution on Train, '
                         'Validation and Test Sets')
    st.write("---")

    st.write("### Model History")
    col1, col2 = st.beta_columns(2)
    with col1:
        model_acc = plt.imread(f"outputs/{version}/model_training_acc.png")
        st.image(model_acc, caption='Model Traninig Accuracy')
    with col2:
        model_loss = plt.imread(f"outputs/{version}/model_training_losses.png")
        st.image(model_loss, caption='Model Traninig Losses')
    st.write("---")

    st.write("### Generalised Performance on Test Set")
    st.dataframe(pd.DataFrame(load_test_evaluation(
        version), index=['Loss', 'Accuracy']))
    col1, col2 = st.beta_columns(2)
    with col1:
        confusion_matrix = plt.imread(
            f"outputs/{version}/confusion_matrix.png")
        st.image(confusion_matrix, caption="Confusion Matrix")
    with col2:
        clf_report = plt.imread(f"outputs/{version}/clf_report.png")
        st.image(clf_report, caption="classification report")

    st.write(
        f"* The overall accuracy of 99.88% and a loss of 0.0041 on "
        f"identifying infected cherry leaves indicate the successful "
        f"performance of this machine learning model in determining the "
        f"presence of powdery mildew. These results align with the defined "
        f"success metrics in the ML business requirements, with a minimum "
        f"degree of accuracy of 97% on identifying infected cherry leaves.")
