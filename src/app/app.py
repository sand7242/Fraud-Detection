from flask import Flask, render_template, request, jsonify
import pickle
from FraudPredictor import FraudPredictor
import json

app = Flask(__name__)


# with open('static/model.pkl') as f:
#     model = pickle.load(f)

model_fp = FraudPredictor('model.pkl')
model_fp.load_model()



@app.route('/', methods=['GET', 'POST'])
def index():
    """Render a simple splash page."""
    return render_template('form/index.html')

# @app.route('/submit', methods=['GET'])
# def submit():
#     """Render a page containing a textarea input where the user can paste a
#     sequence number to be identified as fraud or not fraud.  """
#     return render_template('form/submit.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    """Recieve the article to be classified from an input form and use the
    model to classify.
    """
    db_data = query.get_data()
    query.close()

    # model_fp.get_requests()
    # response = json.dumps(model_fp.raw_data, sort_keys = True, indent = 4, separators = (',', ': '))
    # pred = str(model.predict([data])[0])
    return render_template('form/predict.html', response=response)


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
