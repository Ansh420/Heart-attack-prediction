# Heart-attack-prediction

The silver lining is that heart attacks are highly preventable and simple lifestyle modifications(such as reducing alcohol and tobacco use; eating healthily and exercising) coupled with early treatment greatly improves its prognosis. It is, however, difficult to identify high risk patients because of the multi-factorial nature of several contributory risk factors such as diabetes, high blood pressure, high cholesterol, et cetera. This is where machine learning and data mining come to the rescue.

Doctors and scientists alike have turned to machine learning (ML) techniques to develop screening tools and this is because of their superiority in pattern recognition and classification as compared to other traditional statistical approaches.

I will be giving you a walk through on the development of a screening tool for predicting whether a patient has 10-year risk of developing coronary heart disease(CHD) using different Machine Learning techniques.
## Prerequisites
Before you can run the code, ensure you have the following installed:

- Python 3.6 or later
- NumPy
- pandas
- scikit-learn
- matplotlib
- seaborn
  
## Dataset
The dataset used for training and evaluation is assumed to be in a CSV format. It should contain the following columns:

**age**: Age of the patient
**sex**: 1 for male, 0 for female
**cp**: Chest pain type (1-4)
**trestbps**: Resting blood pressure
**chol**: Serum cholesterol
**fbs**: Fasting blood sugar (1 if > 120 mg/dl, 0 otherwise)
**restecg**: Resting electrocardiographic results (0, 1, 2)
**thalach**: Maximum heart rate achieved
**exang**: Exercise-induced angina (1: yes, 0: no)
**oldpeak**: ST depression induced by exercise   
**slope**: Slope of the ST segment (0, 1, 2)
**ca**: Number of major vessels colored by fluoroscopy
**thal**: Thallium stress test result (0, 1, 2, 3)
**target**: 1 for heart attack, 0 for no heart attack
## Usage
Clone the Repository:

 $ Bash
$ git clone https://github.com/Ansh420/heart-attack-prediction.git


## Install Dependencies:

- Bash
$ pip install -r requirements.txt


## Prepare the Data:

Place your dataset in a CSV file named heart.csv.
- Train the Model:

Bash
$ python train.py


This script will load the dataset, preprocess it, and train a machine learning model.


## Model Architecture
The specific machine learning algorithm used can be customized by modifying the train.py script. Possible options include:

- Logistic Regression
- Decision Trees
- Random Forests
- Support Vector Machines
