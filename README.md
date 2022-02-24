# uber-fare-prediction

### Table of Contents
* Overview
* Motivation
* Technical Aspect
* Requirements
* Run
* Technologies Used
* Credits

### Overview

This is a basic regression model to predict the fare of the uber ride trained on top of Sklearn RandomForestRegressor model. The model would take the different trip parameters (number of passengers, pick up and drop off geographical coordinates, date and time of the trip) as the input and predict the fare amount as output. The model has been trained with 2 lakh records and acquired an R2 score of 0.968.

### Technical Aspect

* Used sklearn RandomForestRegressor as the final model for this problem.
* GridSearchCV was used to find the optimisation of the model.
* seaborn library used to visualise the data.
* Pandas library used to load the data from the csv file and other operations.

### Requirements

* Python (>=3.7) should be installed in your computer. 
* Jupyter notebook or anyother client that support jupyter notebook would be required to open the code.
* You should install the neccessary libraries from the requirement.txt file

### Run

You can use the __UberFarePredictor.pkl__ file to predict the fare.
1. Go to __Uber Fare Prediction.ipynb__ notebook and find the cell containing code to pickle the model (you can find it at almost end of the file)
1. Run the cell, you will get the __UberFarePredictor.pkl__ model.
1. run the below code to see the prediction (<x_values> should be replaced with suitable input values):
```
import pickle
with open('UberFarePredictor.pkl', 'rb') as f:
  model = pickle.load(f)
  model.predict(<x_values>)
```

### Technologies Used

* Python
* Sklearn
* Jupyter Notebook

### Credits

* Problem Statement - https://www.simplilearn.com
* Dataset - https://drive.google.com/file/d/1Yp_1LWg4rtBj6ezbu6AqGRO8LpFUdYiD/view (contains 50 lakh records)

