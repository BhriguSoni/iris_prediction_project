from flask import Flask, render_template, request
import joblib
app = Flask(__name__)
model = joblib.load('iris.pkl')
@app.route('/')
def home():
    return render_template('main.html')
@app.route('/predict', methods=['POST'])
def predict():
    sepal_length = float(request.form['sepalLength'])
    sepal_width = float(request.form['sepalWidth'])
    petal_length = float(request.form['petalLength'])
    petal_width = float(request.form['petalWidth'])
    prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    species = prediction[0]
    return render_template('main.html', prediction=species)
if __name__ == '__main__':
    app.run()