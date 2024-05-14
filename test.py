import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore", message="X has feature names, but LinearRegression was fitted without feature names")

df=pd.read_csv('diabetes.csv')
df.dropna()

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)
X = df.drop(columns="Outcome")
Y=df["Outcome"]
X=np.array(X)
Y=np.array(Y)
scaler = StandardScaler()
X= scaler.fit_transform(X)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)

# calculating mean squared error
mse = mean_squared_error(Y_test, Y_pred)
print("Mean Squared Error:", mse)
r2 = r2_score(Y_test, Y_pred)
print("R-squared (alternative method):", r2)
for i in range(len(Y_pred)):
    if Y_pred[i] <= 0.5:
        Y_pred[i] = 0
    else:
        Y_pred[i] = 1
accuracy = accuracy_score(Y_test, Y_pred)
print("Accuracy:", accuracy)


while True:
    print("Kindly enter the patient details below:")
    user_ip = {
        'Pregnancies': int(input("Pregnancies: ")),
        'Glucose': int(input("Glucose: ")),
        'BloodPressure': int(input("BloodPressure: ")),
        'SkinThicknesss': int(input("SkinThickness: ")),
        'Insulin': int(input("Insulin: ")),
        'BMI': float(input("BMI: ")),
        'DiabetesPedigreeFunction': float(input("DiabetesPedigreeFunction: ")),
        'Age': int(input("Age: ")),
    }

    # Convert user input to DataFrame
    user_input_d = pd.DataFrame([user_ip])

    # Predict using the model
    prediction = model.predict(user_input_d)

    if prediction<=0.5:
        prediction=0
    else:
        prediction=1

    # Output prediction result
    if prediction == 1:
        print("Can be readmitted")
    else:
        print("You are fit to go")
        
    # to continue predicting after a prediction has been made
    continue_input = input("Do you want to make another prediction? (yes/no): ")
    
    if continue_input.lower() != 'yes':
        break
