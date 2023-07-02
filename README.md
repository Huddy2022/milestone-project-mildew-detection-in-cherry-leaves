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

* v1-
* v2 -
* v3 -
* v4 -
* v5 -
* v6 -
* v7 -
* v8 -
* v9 -

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


## Unfixed Bugs
* You will need to mention unfixed bugs and why they were unfixed. This section should include shortcomings of the frameworks or technologies used. Although time can be a significant variable for consideration, paucity of time and difficulty understanding implementation is not a valid reason to leave bugs unfixed.

## Deployment
### Heroku

* The App live link is: https://YOUR_APP_NAME.herokuapp.com/ 
* Set the runtime.txt Python version to a [Heroku-20](https://devcenter.heroku.com/articles/python-support#supported-runtimes) stack currently supported version.
* The project was deployed to Heroku using the following steps.

1. Log in to Heroku and create an App
2. At the Deploy tab, select GitHub as the deployment method.
3. Select your repository name and click Search. Once it is found, click Connect.
4. Select the branch you want to deploy, then click Deploy Branch.
5. The deployment process should happen smoothly if all deployment files are fully functional. Click now the button Open App on the top of the page to access your App.
6. If the slug size is too large then add large files not required for the app to the .slugignore file. 


## Main Data Analysis and Machine Learning Libraries
* Here you should list the libraries used in the project and provide an example(s) of how you used these libraries.


## Credits 

* In this section, you need to reference where you got your content, media and from where you got extra help. It is common practice to use code from other repositories and tutorials. However, it is necessary to be very specific about these sources to avoid plagiarism. 
* You can break the credits section up into Content and Media, depending on what you have included in your project. 

### Content 

- The text for the Home page was taken from Wikipedia Article A.
- Instructions on how to implement form validation on the Sign-Up page were taken from [Specific YouTube Tutorial](https://www.youtube.com/).
- The icons in the footer were taken from [Font Awesome](https://fontawesome.com/).

### Media

- The photos used on the home and sign-up page are from This Open-Source site.
- The images used for the gallery page were taken from this other open-source site.



## Acknowledgements (optional)
* Thank the people that provided support throughout this project.
