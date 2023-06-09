import streamlit as st
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.image import imread

import itertools
import random


def page_cherry_leaves_visualizer_body():
    st.write("### Cherry Leaves Visualizer")
    st.info(
        f"* The client is interested to have a study to visually differentiate"
        f" between an infected powdery mildew cherry leaf and healthy "
        f"cherry leaf.")

    version = 'v2'
    if st.checkbox("Difference between average and variability image"):

        avg_powdery_mildew = plt.imread(
            f"outputs/{version}/avg_var_powdery_mildew.png")
        avg_healthy = plt.imread(f"outputs/{version}/avg_var_healthy.png")

        st.warning(
            f"Our analysis suggests that cherry leaves affected by powdery "
            f"mildew exhibit distinct visual characteristics."
            f" The first noticeable sign is the presence of light-green, "
            f"circular lesions on either leaf surface, which are followed by "
            f"the emergence of a subtle white cotton-like growth in the "
            f"infected area."
            f"\n\n"
            f"In our data visualization notebook, we have visually explored "
            f"and confirmed the existence of these markers through the "
            f"analysis of leaf images."
            f" We observed that healthy leaves exhibit consistent color "
            f"patterns, while leaves affected by powdery mildew show distinct "
            f"irregularities and growth patterns."
            f"\n\n"
            f"To enhance the accuracy of our machine learning model, it is "
            f"crucial to preprocess the images before training."
            f" A vital step in this process is normalization, where we adjust "
            f"the images to a standardized format."
            f" By calculating the mean and standard deviation of the entire "
            f"dataset, we ensure that our model can effectively extract "
            f"features and make accurate predictions."
            f" This normalization procedure was performed on the dataset as "
            f"part of our data visualization and preprocessing efforts.")

        st.image(avg_powdery_mildew,
                 caption='Powdery Mildew cherry leaf - '
                         'Avegare and Variability')
        st.image(avg_healthy, caption='Healthy cherry leaf - '
                                      'Average and Variability')
        st.write("---")

    if st.checkbox("Differences between average powdery mildew cherry "
                   "leaf and average healthy cherry leaf"):
        diff_between_avgs = plt.imread(f"outputs/{version}/avg_diff.png")

        st.warning(
            f"* During our analysis, we observed that the visual differences "
            f"between healthy and powdery mildew-infected cherry leaves were "
            f"not easily differentiated.\n"
            f"* Although there is a slight variation in the appearance of "
            f"the affected area on the cherry leaf, with white stripes."
            f" Both datasets contain a significant number of leaves for "
            f"powdery mildew and healthy leaves."
            f" The ImageDataGenerator task, which includes flipping and "
            f"rotating images, will help the model generalize and learn from "
            f"the available data effectively.")
        st.image(diff_between_avgs,
                 caption='Difference between average images')

    if st.checkbox("Image Montage"):
        st.write("* To refresh the montage, click on 'Create Montage' button")
        my_data_dir = 'inputs/cherry-leaves_dataset/cherry-leaves'
        labels = os.listdir(my_data_dir + '/validation')
        label_to_display = st.selectbox(
            label="Select label", options=labels, index=0)
        if st.button("Create Montage"):
            image_montage(dir_path=my_data_dir + '/validation',
                          label_to_display=label_to_display,
                          nrows=8, ncols=3, figsize=(10, 25))
        st.write("---")


def image_montage(dir_path, label_to_display, nrows, ncols, figsize=(15, 10)):
    sns.set_style("white")
    labels = os.listdir(dir_path)

    # subset the class you are interested to display
    if label_to_display in labels:

        # checks if your montage space is greater than subset size
        # how many images in that folder
        images_list = os.listdir(dir_path+'/' + label_to_display)
        if nrows * ncols < len(images_list):
            img_idx = random.sample(images_list, nrows * ncols)
        else:
            print(
                f"Decrease nrows or ncols to create your montage. \n"
                f"There are {len(images_list)} in your subset. "
                f"You requested a montage with {nrows * ncols} spaces")
            return

        # create list of axes indices based on nrows and ncols
        list_rows = range(0, nrows)
        list_cols = range(0, ncols)
        plot_idx = list(itertools.product(list_rows, list_cols))

        # create a Figure and display images
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        for x in range(0, nrows*ncols):
            img = imread(dir_path + '/' + label_to_display + '/' + img_idx[x])
            img_shape = img.shape
            axes[plot_idx[x][0], plot_idx[x][1]].imshow(img)
            axes[plot_idx[x][0], plot_idx[x][1]].set_title(
                f"Width {img_shape[1]}px x Height {img_shape[0]}px")
            axes[plot_idx[x][0], plot_idx[x][1]].set_xticks([])
            axes[plot_idx[x][0], plot_idx[x][1]].set_yticks([])
        plt.tight_layout()

        st.pyplot(fig=fig)
        # plt.show()

    else:
        print("The label you selected doesn't exist.")
        print(f"The existing options are: {labels}")
