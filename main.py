import pandas as pd
import pickle
from flask import Flask, render_template, request
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
data = pd.read_csv("brain_stoke.csv")
pipe = pickle.load(open("XgBoostClassifier.pkl","rb"))


@app.route('/')
def index():
    gender = sorted(data['gender'].unique())
    married = sorted(data['ever_married'].unique())
    work = sorted(data['work_type'].unique())
    residence = sorted(data['Residence_type'].unique())
    smoke = sorted(data['smoking_status'].unique())
    return render_template('index.html', gender = gender, married = married, work = work, residence = residence, smoke = smoke)


@app.route('/predict', methods=["POST"])
def predict():
    a = request.form.get("gender")	
    b = request.form.get("age")
    c = request.form.get("hypertension")
    d = request.form.get("heart_disease")
    e = request.form.get("married")
    f = request.form.get("work")
    g = request.form.get("residence")
    h = request.form.get("level")
    i = request.form.get("bmi")
    j = request.form.get("smoke")
    
    print(a,b,c,d,e,f,g,h,i,j)
    encoder = LabelEncoder()
    input = pd.DataFrame([[a,b,c,d,e,f,g,h,i,j]], columns = ["gender","age", "hypertension","heart_disease","ever_married","work_type","Residence_type","avg_glucose_level","bmi","smoking_status"])
    input["gender"] = encoder.fit_transform(input["gender"])
    input["heart_disease"] = encoder.fit_transform(input["heart_disease"])
    input["ever_married"] = encoder.fit_transform(input["ever_married"])
    input["work_type"] = encoder.fit_transform(input["work_type"])
    input["Residence_type"] = encoder.fit_transform(input["Residence_type"])
    input["smoking_status"] = encoder.fit_transform(input["smoking_status"])
    prediction = pipe.predict(input)[0]
    if prediction == 0:
        return "He or She is not Suffering from No Brain Stokes"
    elif prediction == 1:
        return "He or She is Suffering from Brain Stokes"
    return str(prediction)


if __name__ == "__main__":
    app.run(debug=True, port=5001)