# DS-TermProject
This project is a Term-Project of Data Science Lecture.

We uploaded our final code.
Also, we uploaded our algorithm that can be used in general.

* Covid19.py <-- This file is our final result
* data_analysis.py <-- This is data analysis code using Random Forest Classifier, Logistic Regression, and Decision Tree Classifier
* one_hot_encoidng.py <-- This is one-hot encoding code using Pandas and Numpy


Our team members:
  * 202035509 KIM YEEUN
  * 202035518 NOH HYUNGJU
  * 202035521 PARK JEONGYEON
  
# Project Objection
To help people predict the severity of COVID-19 whith his/her/their symptoms
 
=== WARNING ===
This is NOT an ACCURATE DIAGNOSIS, it is just a PREDICTION based on Symptoms.
It is recommended that you use this for SIMPLE GUESSWORK ONLY, and contact medical professional for accurate diagnosis.


# Used Data
We used 'COVID-19 Symptoms Checker' data from BILAL HUNGUND
Kaggle:
  https://www.kaggle.com/datasets/iamhungundji/covid19-symptoms-checker?select=Cleaned-Data.csv
  

# Used Algorithm
In data preprocessing

  One-hot encoding <-- to get data easy-to-handle
  PCA <-- to make data distribution
 
In data analysis

  Logistic Regression Model <-- to predict severity level using symptoms (data we used is categorical data, so we used logistic)
  Random Forest Classifier <-- to find severity level using symptoms
  Decision Tree Classifier <-- to find severity level using symptoms
  
In data evaluation

  K-fold (test 30%, k=10)
