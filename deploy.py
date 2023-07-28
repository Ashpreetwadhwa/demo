from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# load the model
model = joblib.load('saved_model1.joblib')

@app.route('/')
def home():
    result = ''
    return render_template('index.html', **locals())

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])
        input_features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        result = model.predict(input_features)[0]
        return render_template('index.html', result=result)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
