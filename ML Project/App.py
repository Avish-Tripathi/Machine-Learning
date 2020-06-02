from flask import Flask, render_template, url_for, request
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('Home.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    final = [np.array(features)]
    predict = model.predict(final)

    output = round(predict[0], 4)

    return render_template('Home.html', prediction_text='Predicted Percentile is {}'.format(output))

if __name__ == '__main__':
    app.run(debug=True)