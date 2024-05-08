# Spam Email Detection using Machine Learning 

<!-- TABLE OF CONTENTS -->
<details>
  <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#data-Description">Data Description</a></li>
    <li><a href="#data-Pre-processing">Data Pre-processing</a></li>
    <li><a href="#model-Training-and-Evaluation">Model Training and Evaluation</a></li>
  
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

Spam detection is the process of identifying and filtering out unwanted or unsolicited messages, typically in the form of emails or text messages. These messages are often sent by spammers or malicious actors with the intent of promoting a product, service, or website, or to trick the recipient into providing personal information or downloading malware. Spam detection typically involves the use of machine learning algorithms that can analyze the content of messages and identify patterns or characteristics that are commonly associated with spam. These algorithms can be trained on large datasets of labeled examples of spam and legitimate messages, allowing them to learn to distinguish between the two with a high degree of accuracy. Effective spam detection is an important task for both individuals and organizations, as it can help to prevent unwanted messages from cluttering inboxes, reduce the risk of phishing attacks, and improve overall cybersecurity.

This repository contains a Python script that uses various machine learning models to classify spam messages from ham messages. The model is trained on a Popular dataset of Spam emails and we use multiple machine learning models for classification.

### Built With

 - NumPy
 
 - Pandas

 - Matplotlib

 - Seaborn

 - Sklearn
 
 <br>
 
 Anyways you can install all the above mentioned libraries at a glance by executing the following command:
 
  ```sh
  pip install -r requirements.txt
  ```

<!-- GETTING STARTED -->
## Getting Started

This is make you understand how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

### Installation

1. Clone the repo

2. Install the required libraries

   ```sh
    pip install -r requirements.txt
   ```
3. Open and execute ```.ipynb``` file (After complete Execution you will get a ```.pkl``` file for project Deployment

# Dataset Description

We have utilized the Email-Spam dataset, which is publicly available on Kaggle. The dataset comprises a collection of 5,572 emails, each having two features: Category and Message. 

```Message```   Message feature contains the actual text of the email. 

```Category```  The Category feature distinguishes between Spam and Ham emails

# Data Pre-processing

### Steps Done:

- The Unnamed: 2, Unnamed: 3, and Unnamed: 4 columns are dropped.

- The Category column is converted to binary values.

- The dataset is split into training and testing sets using train_test_split() function from sklearn.model_selection.

- The emails are transformed into numerical features using the TfidfVectorizer() function from sklearn.feature_extraction.text.
<br>


Intially the 'Unnamed: 2', 'Unnamed: 3', and 'Unnamed: 4' columns are then dropped from the DataFrame and the code checks for null values in the DataFrame using the 'isnull()' method. The 'Category' column in the DataFrame is then converted to numerical values (0 and 1) where 'spam' is replaced with 0 and 'ham' is replaced with 1.
The number of values in each category is printed using the 'value_counts()' method. The X and Y variables are then created where X stores the 'Message' column of the DataFrame, and Y stores the 'Category' column. The code then splits the data into training and testing sets using the 'train_test_split()' method from the scikit-learn library. The TfidfVectorizer is then used to extract features from the text data. The 'min_df' parameter is set to 1, the 'stop_words' parameter is set to 'english', and the 'lowercase' parameter is set to 'True'. The feature extraction is performed on both the training and testing data using the 'fit_transform()' and 'transform()' methods.
Finally, the 'Y_train' and 'Y_test' variables are converted to integers.

# Model Training and Evaluation

As we already splitted the dataset into training and testing parts, the machine learning models can be able to train on the training data by using ```fit()``` method and then we are testing the trained machine learning model by using ```predict()``` method. To know the performance of the  trained machine learning models we are evaluating the predicted data and original data by using evaluation metrics such as accuracy, precision, recall, and F1-score.

The following Machine Learning Algorithms are used:

- Logistic Regression
- K Nearest Neighbors
- Decision Trees
- Random Forest
- Stacking model

