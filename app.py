from flask import *
import pickle
import  numpy as np
model = pickle.load(open("forest_fire_model.pkl", 'rb'))
app = Flask(__name__)

@app.route("/")
def welcome():
    return render_template("index.html")

@app.route("/predict", methods = ['POST', 'GET'])
def predict():
    int_feautres = [int(x) for x in request.form.values()]
    final = [np.array(int_feautres)]
    print(int_feautres)
    print(final)
    prediction = model.predict(final)
    print(prediction[0])
    output = prediction[0]
    print(output)
    if output == 1:
        return render_template('index.html',pred='Your Forest is in Danger')
    elif output == 0:
        return render_template('index.html',pred='Your Forest is safe.')

if __name__ == '__main__':
    app.run(debug = True)
