# Cherry Leaf Powdery Mildew Detector

Deployed version : <https://powdery-mildew-detection-86ce1c83ad33.herokuapp.com/>

## Dataset Content

* The dataset is sourced from [Kaggle](https://www.kaggle.com/codeinstitute/cherry-leaves). We then created a fictitious user story where predictive analytics can be applied in a real project in the workplace.
* The dataset contains 4208 images taken from the client's crop fields. The images show healthy cherry leaves and cherry leaves that have powdery mildew, a fungal disease that affects many plant species. The cherry plantation crop is one of the finest products in their portfolio, and the company is concerned about supplying the market with a compromised quality product.

## Business Requirements

The cherry plantation crop from Farmy & Foods is facing a challenge where their cherry plantations have been presenting powdery mildew. Currently, the process is manual verification if a given cherry tree contains powdery mildew. An employee spends around 30 minutes in each tree, taking a few samples of tree leaves and verifying visually if the leaf tree is healthy or has powdery mildew. If there is powdery mildew, the employee applies a specific compound to kill the fungus. The time spent applying this compound is 1 minute.  The company has thousands of cherry trees, located on multiple farms across the country. As a result, this manual process is not scalable due to the time spent in the manual process inspection.

To save time in this process, the IT team suggested an ML system that detects instantly, using a leaf tree image, if it is healthy or has powdery mildew. A similar manual process is in place for other crops for detecting pests, and if this initiative is successful, there is a realistic chance to replicate this project for all other crops. The dataset is a collection of cherry leaf images provided by Farmy & Foods, taken from their crops.

* 1 - The client is interested in conducting a study to visually differentiate a healthy cherry leaf from one with powdery mildew.
* 2 - The client is interested in predicting if a cherry leaf is healthy or contains powdery mildew.

## Rationale to map the business requirements to the Data Visualizations and ML tasks

Business Requirement 1: Conduct a study to visually differentiate a healthy cherry leaf from one with powdery mildew.

* Analyze average images and variability images for each class (healthy or powdery mildew).
* Identify and highlight the differences between average healthy and average powdery mildew cherry leaves.
* Create image montages for each class to visually showcase the characteristics of healthy and powdery mildew cherry leaves.

Business Requirement 2: Develop a predictive model to determine if a cherry leaf is healthy or contains powdery mildew.

* Use Neural Networks to map the relationships between the features (cherry leaf images) and the labels (healthy or powdery mildew).
* Consider the image shape when loading the images into memory for training the model, ensuring it meets the performance requirement.
* Explore different image shape options (e.g., 256x256, 100x100, or 75x75) to find the optimal balance between model size and performance.
* Ensure the trained model achieves a minimum accuracy of 97% to meet the performance goal.

## Hypothesis and how to validate?

Hypothesis 1: Does infected leaves have clear marks differentiating them from the healthy leaves.

* Conduct research about the specific disease (powdery mildew) affecting cherry leaves and gather information about the visual characteristics or marks associated with infected leaves.
* Perform an average image study by analyzing a large number of infected and healthy cherry leaf images to identify consistent visual patterns or distinguishing marks that differentiate infected leaves from healthy ones.
* Compare the identified marks or patterns to validate their significance in differentiating infected and healthy leaves.
* Use statistical analysis or machine learning techniques to quantify the effectiveness of the identified marks in predicting the presence of powdery mildew.

Hypothesis 2: Does softmax perform better than sigmoid as the activation function for the CNN output layer.

* Conducted multiple iterations using different hyperparameters and techniques, with sigmoid as the primary activation function for the CNN output layer.
* Utilized hyperparameter tuning techniques, such as Keras Tuner, to search for optimal hyperparameter configurations.
* Explored various hyperparameters, including image reshaping/sizing, regularizers, batch normalization, and alternative activation functions (e.g., tanh instead of relu).
* Evaluated and compared the performance of each model using appropriate evaluation metrics.
* Identified the best hyperparameter configuration based on the results obtained from the iterations.
* Once the best hyperparameters were determined, tested the model with softmax as the activation function for the CNN output layer.
* Assessed the performance of the model with softmax and compared it to the performance of the models using sigmoid.
* Analyzed the results to determine if softmax outperformed sigmoid as the activation function for the CNN output layer.

## ML Business Case

* Develop an ML model to predict if a cherry leaf is healthy or infected with powdery mildew based on provided image data.
* Objective: Provide a fast and more reliable detection method for powdery mildew in cherry leaves.
* Success Metrics:
  * Accuracy of 97% or above on the test set.
* Model Output: Flag indicating if the cherry leaf is healthy or contains powdery mildew, along with associated probability.
* Heuristics:
  * Current detection method relies on manual visual inspection, which is time-consuming and prone to human error.
  * Training data comes from the provided cherry leaf image database.
* Dataset: Contains 4,208 images of cherry leaves.
* Hypothesis: Infected leaves have clear visual marks differentiating them from healthy leaves.
* Validation Approach:
  * Conduct research on powdery mildew and study average images and variability for each class (healthy and infected).
  * Compare average healthy and infected cherry leaves to identify visual differences.
  * Build and train multiple models using different hyperparameters, including image reshaping, regularizers, batch normalization, and activation functions (e.g., sigmoid and softmax).
  * Evaluate and compare model performance to determine the best hyperparameters and validate the hypothesis.

## Development and Machine Learning Model Iterations

* The Tensorflow binary image classification model went through a series of iterations and hyperparameters in order to produce an optimised model capable of handling the data.
* Version 2 was the eventual accepted production version. Legacy versions of the models and their performance can be found in the 'Outputs' section of this repository.
* The general structure of the model was heavily influenced by Code Institute's Malaria Detector model for binary classification, and adjusted and tested upon to find optimal hyperparameters.
* Before the model trained on it, data was augmented using TensorFlow's ImageDataGenerator, which re shaped, flipped, and rotated the images in order to provide a larger dataset and increase the model's robustness to variations in the input images. This augmentation technique helped prevent overfitting and allowed the model to generalize better to unseen data.
* The initial model was constructed of three consecutive pairs of 2D convolution and pooling layers used to isolate areas of importance and contrast within the image for the model to train itself on. The numbers of filters for the three convolution layers were set to values of 32, 64, and 64 respectively and the kernel sizes at (3,3). Powers of 2 were chosen as filter numbers for the convolution layers to optimise processing. Pooling layer pool sizes were set to the industry standard value of (2,2).
* This is followed by a single dense layer of many neurons (160 in the final production model), a dropout layer of 0.5 in order to avoid overfitting of the model, and a final dense layer containing a single neuron with a sigmoid activation function, as is standard for binary classification models.
* Before running each iteration of the model, the most suitable hyperparameters out of a user-provided selection were attained using the Keras Tuner.
* Two hyperparameters were selected in this search process for optimisation; the neuron count in the main densely connected layer of the neural network, and the learning rate of the model. These values were chosen with guidance from TensorFlow's Keras Tuner Tutorial and later simplified to remove those hyperparameter combinations that performed consistently poorly in order to increase tuning speed.
* The hyperparameter optimisation search was conducted using the Hyperband search algorithm.
* The data was loaded into the model in batch sizes of 20, a relatively small even number which was chosen both to further improve time efficiency of the model and avoid overfitting.
* TensorFlow's EarlyStopping function was included to halt training of the model early when the loss value on validation data was no longer clearly improving (a 'patience' value of 3 was passed in).
* The model was then trained over a possible 25 epochs with steps per epoch set to the train set class lengths divided by 20 (the batch size) and an early stopping function were passed into the fit() function.
* Version 1 (V1) - The initial model was developed without hyperparameter tuning using the original dataset of cherry leaf images to detect powdery mildew. Although the model achieved a high accuracy of 99.64%, it was clear that the dataset size needed improvement for better performance. Hyperparameter tuning was attempted using the Keras Tuner, but the search took over 4 hours, and an error prevented early stopping. Despite the high accuracy, the training time was inefficient for the project.
* Version 2 (V2) - To address the issues encountered in V1, the images were resized to 100x100 to expedite the hyperparameter search. The search time was significantly reduced to under 30 minutes, and the hyperparameter search successfully identified a layer size of 160 and an optimizer value of 0.001. The model achieved an accuracy of 99.88% and a loss of 0.0041, demonstrating improved performance compared to V1.
* Version 3 (V3) - In an attempt to further speed up the search and training process, the image size was reduced to 75x75 while maintaining the same hyperparameter search settings as V2. The search time was further reduced to approximately 17 minutes, and the model obtained a layer size of 224 and an optimizer value of 0.001. However, the accuracy of the trained model decreased to 99.64%, with a loss of 0.016.
* Version 4 (V4) - Returning to the image size of 100x100 used in V2, regularization techniques such as L1 and L2 were introduced to fine-tune the model. Various filter and flatten configurations were attempted, but due to excessive search time, adjustments were made to simplify the model architecture. The resulting search, which lasted about 1 hour and 30 minutes, identified a layer size of 96 and an optimizer value of 0.001. However, the model failed to learn, with an accuracy stuck at 50% and a loss of 0.693.
* Version 5 (V5) - Building upon the regularizers introduced in V4, additional configurations were explored by including the none regularizer at the end. The filter sizes were adjusted, and the model achieved a similar search speed to V4. The hyperparameter search identified a layer size of 224 and an optimizer value of 0.001. Although the model's training time improved compared to V4, the accuracy and loss did not surpass those achieved in V2.
* Version 6 (V6) - Batch normalization was introduced in this version to enhance model training during epochs. The filter and flatten configurations were similar to V5, with a dropout rate of 0.3. The search speed improved, and the hyperparameter search yielded a layer size of 256 and an optimizer value of 0.001. The model achieved an accuracy of 99.64% and a loss of 0.014, showing a slight improvement in loss compared to V5.
* Version 7 (V7) - Continuing with the regularizers and batch normalization, a change was made to the dropout rate, increasing it to 0.4. The hyperparameter search resulted in a layer size of 128 and an optimizer value of 0.0001. However, the model showed signs of overfitting towards the end of training, resulting in a decreased accuracy of 98.93% and an increased loss of 0.057.
* Version 8 (V8) - Regularizers and batch normalization were abandoned in favor of a model architecture similar to V2. The filter sizes and other hyperparameters remained the same, but the activation function was changed to Tanh for a larger range. The hyperparameter search identified a layer size of 224 and an optimizer value of 0.001. The model achieved a similar accuracy of 99.88% to V2, but the loss slightly increased to 0.0010.
* Version 9 (V9) - The final version retained the same architecture as V2, but the activation function was changed to softmax to better suit the multiclass nature of the problem. The hyperparameter search resulted in a layer size of 128 and an optimizer value of 0.001. The model achieved a comparable accuracy of 99.88%, but the loss slightly increased to 0.0096. Notably, the softmax activation had fewer trainable parameters compared to the sigmoid activation used in V2.

## Dashboard Design & features

Page 1: Quick Project Summary

* Summary Page:
  * Provides a quick summary of the project and its objectives.
  * Includes general information about powdery mildew in cherry trees and the visual criteria used to detect infected leaves.
  * Contains links to additional information in the project README file.

Page 2: Cherry Leaves Visualizer

Answers buisness requirements 1

* Allows the visualization of cherry leaves, with a focus on the differences between healthy leaves and leaves infected with powdery mildew.
* Provides options to display the difference between average and variability images, the differences between average powdery mildew cherry leaves and average healthy cherry leaves, and an image montage of cherry leaves.

Page 3: Powdery Mildew Detector

Answers buisness requirements 2

* Used for detecting powdery mildew in cherry leaves.
* Provides the option to upload cherry leaf images for live prediction.
* Processes the uploaded images and uses a machine learning model to predict whether the leaves are healthy or infected with powdery mildew.
* Displays the predictions along with a summary report.

Page 4: Project Hypothesis & validation

* Presents the project hypothesis and validation methodology.
* Explains the steps followed in the project, including data collection, data preprocessing, model development, model evaluation, comparison and selection, and hypothesis testing.
* Discusses the outcome and significance of the project.

Page 5: ML Performance & evaluation

* Provides an overview of the performance of the machine learning model in identifying powdery mildew in cherry leaves.
* Displays model evaluation metrics such as accuracy, precision, recall, and F1-score.
* Includes a confusion matrix visualization to understand the model's performance.

## The CRISP-DM Methodology

My CRISP-DM provides a structured approach for the data mining project. It outlines the different phases of a project, the tasks within each phase, and the relationships between these tasks.

To document this process for the Powdery Mildew detection project, a Kanban Board provided by GitHub was used in the repository's project section. A Kanban board is an agile project management tool that helped visualize the work, limit work-in-progress, and improve efficiency. It uses cards and columns to organize tasks and facilitate continuous improvement.

In this project, the CRISP-DM process was divided into sprints. Each sprint is associated with epics based on the CRISP-DM tasks. These epics were further broken down into individual tasks. Throughout the workflow, tasks can progress through different statuses such as To Do, In Progress, and Done, providing a clear overview of the project's progress.

In addition to the tasks and epics within the CRISP-DM process, the Powdery Mildew detection project also incorporated user stories. User stories represent specific functionalities or features from the perspective of end users.

To capture these user stories, comments were used within the Kanban board to provide detailed information about the tasks. These comments outlined the specific requirements, objectives, and expectations related to each user story.

By including user stories in the comments section, I could ensure that the implementation of each task aligned with the desired functionalities and provided value to the end users. This approach helped to prioritize development efforts, track progress, and maintain a user-centric focus throughout the project.

## Unfixed Bugs

Images producing false predicitons

* In terms of unfixed bugs, one known issue that has been identified is related to the incorrect image appearing in the confusion matrix. The confusion matrix is a visual representation of the performance of a classification model, showing the predicted labels versus the actual labels.
* In this case, there seems to be a bug where an incorrect image is displayed within the confusion matrix, possibly misrepresenting the actual performance of the model. This discrepancy can lead to confusion and misinterpretation of the model's accuracy and effectiveness.
* Healthy images wrongly predicted as Powdery Mildew:
powdery_mildew/4c756b73-5e7d-40ec-9b36-1866c49f2e43___FREC_Pwd.M 5156_flipLR.JPG
* The image looks shadowed which can introduce variations in pixel intensities, which may affect the features extracted by the algorithm and lead to misclassifications.
* To address the issue of misclassification caused by shadowed images, we could consider implementing techniques such as further data augmentation, image enhancement, and fine-tuning the model to improve the accuracy of powdery mildew detection.

## Deployment

### Heroku

* The App live link is: [Cherry leaf powdery mildew detector app](https://powdery-mildew-detection-86ce1c83ad33.herokuapp.com/)

* Set the runtime.txt Python version to a [Heroku-20](https://devcenter.heroku.com/articles/python-support#supported-runtimes) stack currently supported version.
* The project was deployed to Heroku using the following steps.

1. Log in to Heroku and create an App with deseried name
2. At the Deploy tab, select GitHub as the deployment method.
3. Select your repository name and click Search. Once it is found, click Connect
4. Log into Heroku CLI in IDE workspace terminal using the bash command: heroku login -i and enter user credentials
5. In terminal set heroku heroku stack:set heroku-20 -a appname, for compatibility with the Python 3.8.12 version used for this project
6. Select the main branch, then click Deploy Branch.
7. Wait for the logs to run while the dependencies are installed and the app is being built.
8. Once finished and succesfully deployed I could open the app from the button at the top.
9. If the slug size was too large then I added large files not required for the app to the .slugignore file.

## Main Data Analysis and Machine Learning Libraries

* numpy 1.19.2 - Used for converting data to arrays.
* pandas 1.1.2 - Used for creating and saving data as dataframes.
* matplotlib 3.3.1 - Used for plotting the sets' distribution.
* seaborn 0.11.0 - Used for plotting the model's confusion matrix.
* plotly 4.12.0 - Used for plotting the model's learning curve.
* streamlit 0.85.0 - Used for creating the dashboard.
* scikit-learn 0.24.2 - Used for evaluating the model.
* tensorflow-cpu 2.6.0 - Used for creating the model.
* keras 2.6.0 - Used for setting the model's hyperparameters.

### Other technologies used

* Streamlit - Development of dashboard for presentation of data and project delivery
* Heroku - Deployment of dashboard as web application
* Jupiter Notebook - to edit code for this project
* Kaggle - to download datasets for this project
* Git/GitHub - Version control and storage of source code
* Codeanywhere - IDE Workspace in which application was developed

## Testing

### Manual Testing

* I manually tested each page of the dashboard after deployment and documented each test in the kanban board [Kanban](https://github.com/users/Huddy2022/projects/5).
* With each test i documented in the comment section, its feature, action, expected result and actual result.

### Validation

* All of the Python code in this project was validated as conforming to PEP8 standards using [Pep8](https://pep8ci.herokuapp.com/).
* For the Streamlit app pages and source code files, I simply edited the code until no errors were recorded in the Problems section of the codeanywhere workspace.

## Credits

### Content

* The cherry leaves dataset was linked from Kaggle and created by Code Institute.
* The CRISP-DM that was created on my github, i modelled after using this site [Specific Datascience website](https://www.datascience-pm.com/crisp-dm-2/)<>.
* Many of the project's functions were transferred from Code Institute's sample Malaria Detector project.
* For the Keras Tuner i got alot of insight from this website [Tensorflow](https://www.tensorflow.org/tutorials/keras/keras_tuner/).
* When i came to tuning the hyperperamter searching i used inspriation from this website for the regualizers and batch normalization [Keras website](https://keras.io/api/layers/regularizers/).

### Media

*

## Acknowledgements

* My mentor, Mo Shami, for supervising this project.

## Deployed version at [Cherry leaf powdery mildew detector app](https://powdery-mildew-detection-86ce1c83ad33.herokuapp.com/)
