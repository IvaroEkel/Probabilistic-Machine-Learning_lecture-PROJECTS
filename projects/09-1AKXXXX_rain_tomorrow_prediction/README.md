# Rain Prediction in Australia using Probabilistic Machine Learning  

## Overview  
This project applies probabilistic machine learning techniques to predict whether it will rain tomorrow in Australia, based on historical weather observations. The goal is to compare different modeling approaches and identify the best-performing method for this binary classification task.  

The dataset consists of 145,460 observations collected from various weather stations across Australia over a 10-year period. Predicting rain is not only a common and publicly recognized task but also holds practical importance in agriculture, transportation, and disaster management.  


The jupyter notebook `class_prediction.ipynb` contains the code for data preprocessing, model training, evaluation, and comparison of results.
The project report is in the results folder under the name `09-1AKXXXX_rain_tomorrow_prediction.pdf`.

---


## Research Question  
**What is the best performing model for predicting if it will rain tomorrow given historical weather data?**  

### Hypotheses  
1. **H1:** XGBoost will outperform Logistic Regression in predicting rainfall.  
2. **H2:** Logistic Regression will outperform Bayesian Logistic Regression in terms of accuracy.  
3. **H3:** XGBoost will outperform Bayesian Logistic Regression in terms of accuracy.  

---

## Dataset 


**Source:** Historical weather observations across 49 Australian locations.  
**Size:** 145,460 records, 23 variables (continuous and categorical).  
**Target variable:** `RainTomorrow` (Yes/No).  

---

## Main Steps

### 1. Data Cleaning & Preprocessing  
- Handle missing values  
- Normalize and encode features  
- Remove outliers where necessary  

### 2. Exploratory Data Analysis (EDA)  
- Visualize feature distributions  
- Identify correlations and patterns  
- Explore class balance  

### 3. Models Compared  
- **Logistic Regression** – interpretable baseline model for binary classification  
- **XGBoost** – scalable, regularized gradient boosting with strong performance on tabular data  
- **Bayesian Logistic Regression** – probabilistic model incorporating parameter uncertainty via Laplace approximation  

### 4. Results  
- Model performance metrics  
- Comparative evaluation plots  

### 5. Discussion  
- Interpretation of findings  
- Model trade-offs and limitations  

### 6. References  
- Cited papers  


---
## Project Structure
results/: This directory contains the main written project report PDF (09-1AKXXXX_rain_tomorrow_prediction.pdf) and the generated visualizations. 
Since you mentioned overleaf as a prefered deliverable, you can find the link to the original overleaf [document](https://www.overleaf.com/read/zxqxvwhxwrjj#4c192a) here.

notebooks/: This directory contains scripts and notebooks used for the processing, classification and benchmarking.

data/: This directory currently contains only the data used for generating the map in the written report. The original dataset is not included due to its size and can be used via the link in the class_prediction.ipynb notebook. It refers to my github repository where the original dataset is stored.



## How to use this repository
This repository contains a Jupyter notebook that can be run locally or in a cloud environment. To run the notebook, you will need to have Python and Jupyter installed, along with the required libraries.
The required libraries are listed in the `requirements.txt` file. The main libraries used in this project are:
- pandas
- sklearn
- xgboost
- pymc3
- matplotlib
```bash
python -m venv venv
source venv/bin/activate  # On Windows: `venv\Scripts\activate`
pip install -r requirements.txt
```