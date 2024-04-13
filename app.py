from flask import Flask, render_template,jsonify, request
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler



app = Flask(__name__)

## import logistic regression model and StandardScaler

log_reg=pickle.load(open('model/modelForPrediction.pkl','rb'))
standard_scaler=pickle.load(open('model/standardScalar.pkl','rb'))

## route for home page
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predictdata', methods=['GET','POST'])
def predict_datapoint():
    if request.method =='POST':
        Pregnancies=int(request.form.get('Pregnancies'))
        Glucose=int(request.form.get('Glucose'))
        BloodPressure=int(request.form.get('BloodPressure'))
        SkinThickness=int(request.form.get('SkinThickness'))
        Insulin=int(request.form.get('Insulin'))
        BMI=float(request.form.get('BMI'))
        DiabetesPedigreeFunction=float(request.form.get('DiabetesPedigreeFunction'))
        Age=int(request.form.get('Age'))

        new_scale_data=standard_scaler.transform([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
        predict=log_reg.predict(new_scale_data)

        if predict[0]==1:
            result= 'Diabetic'
        else:
            result= 'Non-Diabetic'    

        return render_template('single_prediction.html', result=result)


    else:
        return render_template('home.html')

if __name__=="__main__":
    app.run(host="0.0.0.0")
