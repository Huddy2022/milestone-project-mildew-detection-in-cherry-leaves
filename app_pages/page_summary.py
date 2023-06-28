import streamlit as st
import matplotlib.pyplot as plt


def page_summary_body():

    st.write("### Quick Project Summary")

    st.info(
        f"**General Information**\n\n"
        f"Powdery mildew is a parasitic fungal disease caused by Podosphaera "
        f"clandestina in cherry trees."
        f" When the fungus begins to take over the plants, a layer of mildew "
        f"made up of many spores forms across the top of the leaves."
        f" The disease is particularly severe on new growth, can slow down the"
        f" growth of the plant, and can infect fruit as well, causing direct "
        f"crop loss."
        f"\n\nVisual criteria used to detect infected leaves are:\n\n"
        f"* Light-green, circular lesion on either leaf surface.\n"
        f"* Subtle white cotton-like growth develops in the infected area on "
        f"either leaf surface and on the fruits, thus reducing yield and "
        f"quality."
        f"\n\n"
        f"**Powdery Mildew Information**\n\n"
        f"* Powdery Mildew is a disease infecting herbaceous and woody "
        f"plants, and can result "
        f"in a low fruit yield in the case of Cherry Trees.\n"
        f"* The current process is manual verification if a given cherry "
        f"tree contains powdery"
        f"mildew.\n"
        f"* An employee spends around 30 minutes in each tree, taking a "
        f"few samples of "
        f"tree leaves and verifying visually if the leaf tree is healthy "
        f"or has powdery" f"mildew.\n"
        f"* According to the [Connecticut Portal](https://portal.ct.gov"
        f"/CAES/Fact-Sheets/Plant-Pathology/Powdery-Mildew),"
        f" powdery mildews are easily recognized by the white, powdery "
        f"growth of the fungus on infected portions of the plant host."
        f" The powdery appearance results from the superficial growth of the "
        f"fungus as thread-like strands (hyphae) over the plant surface"
        f" and the production of chains of spores (conidia). Colonies can "
        f"vary in appearance from fluffy and white to sparse and gray."
    )

    st.write(
        f"* For additional information, please visit and **read** the "
        f"[Project README file](https://github.com/Huddy2022/milestone-"
        f"project-mildew-detection-in-cherry-leaves/blob/main/README.md).")

    st.success(
        f"**Project Objectives**\n\n"
        f"The project aims to achieve the following objectives:\n\n"
        f"1. Conduct a study to visually differentiate a healthy leaf from "
        f"an infected leaf."
        f"\n\n"
        f"2. Develop an accurate prediction model to determine whether a "
        f"given leaf is infected by powdery mildew or not."
        f"\n\n"
        f"3. Provide the ability to download a prediction report of "
        f"the examined leaves."
        f"\n"
    )
