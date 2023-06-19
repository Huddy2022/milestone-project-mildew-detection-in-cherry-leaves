import streamlit as st
import matplotlib.pyplot as plt


def page_project_hypothesis_body():
    st.write("### Project Hypothesis and Validation")

    st.success(
        f"* We suspect powdery mildew infected leaves have clear white lines/strips, "
        f"typically in the middle of the leaf, that can differentiate, from a healthy leaf. \n\n"
        f"* An Image Montage, shows that typically a powdery mildew lead has white lines across. "
        f"Average Image, Variability Image and Difference between Averages studies didn't reveal "
        f"any clear pattern to differentiate one to another."

    )
