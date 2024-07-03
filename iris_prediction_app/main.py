from flask import Flask, render_template, request
import pickle
import numpy as np
import os


app = Flask(__name__)

# Load the model
model_path = os.path.join(os.path.dirname(__file__), 'model/iris.pkl')
with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    sepal_length = float(request.form['sepalLength'])
    sepal_width = float(request.form['sepalWidth'])
    petal_length = float(request.form['petalLength'])
    petal_width = float(request.form['petalWidth'])

    # Prepare the features for prediction
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
    # Make a prediction
    prediction = model.predict(features)[0]
    return render_template('index.html', prediction=prediction)

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)
