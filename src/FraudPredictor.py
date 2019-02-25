import pandas as pd
import pickle
import requests
import json
from pandas.io.json import json_normalize
from DataLoader import format_data, impute_missing_data
import psycopg2


class FraudPredictor():
    '''
    Class for handling model building and new data classification
    '''

    def __init__(self, model_path):
        self.model_path = model_path # where input data lives
        self.random_seed = 1
        self.api_key = 'vYm9mTUuspeyAWH1v-acfoTlck-tCxwTw9YfCynC'
        self.url = 'https://hxobin8em5.execute-api.us-west-2.amazonaws.com/api/'
        self.sequence_number = 0

    def run_analysis(self):
        self.load_model()
        self.get_requests()
        self.load_dataframe()
        self.store_unique_id()
        self.format_api_data()
        self.predict()

        self.output = {'object_id': self.unique_id, \
        'prediction': self.prediction, 'probability_fraud': self.fraud_proba}
        print(self.output)

    def load_model(self):
        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)

    def get_requests(self):
        self.response = requests.post(self.url, json={'api_key': self.api_key,
                                    'sequence_number': self.sequence_number})
        self.raw_data = self.response.json()

    def store_unique_id(self):
        self.unique_id = self.df['object_id'].values[0]

    def load_dataframe(self):
        self.df = json_normalize(self.raw_data['data'])

    def format_api_data(self):
        self.df = format_data(self.df)
        train_df = pd.read_csv('../data/train_df.csv')
        train_df.append(self.df)
        train_df = impute_missing_data(train_df)
        self.df = train_df[-1:].drop(['fraud', 'Unnamed: 0'], axis=1)

    def predict(self):
        self.y_pred = self.model.predict(self.df)
        self.proba = self.model.predict_proba(self.df)
        self.fraud_proba = round(self.proba[0,1], 2)
        if self.y_pred == 1:
            self.prediction = 'Fraud'
        else:
            self.prediction = 'Not Fraud'

class DataSQL():
    def __init__(self):
        self.conn = psycopg2.connect(dbname = "fraud_detection", user = "ubuntu", host="/var/run/postgresql")
        self.cur = self.conn.cursor()

    def insert_data(self, data):
        self.cur.execute("INSERT INTO test_predictions VALUES (%(fraud_id)s, %(predictions)s, %(percentage)s)", data)
        self.conn.commit()

    def iter_insert(self, tup_ls):
        for tup in tup_ls:
            print(tup)
            self.insert_data(tup)

    def get_data(self):
        query = """
                SELECT fraud_id, prediction, percentage,
                FROM test_predictions,
                GROUP BY idx DESC LIMIT 2;"""
        self.cur.execute(query)
        rows = cur.fetchall()
        return rows

    def close(self):
        self.conn.close()
        self.cur.close()


if __name__=='__main__':
    home = '/Users/paulsandoval/Documents/galvanize-dsi/m3_cloud_computing/dsi-fraud-detection-case-study/'
    model_path = home + 'src/model.pkl'
    fp = FraudPredictor(model_path)
    fp.run_analysis()
    data = fp.output

    query = DataSQL()
    query.iter_insert(data)
    # db_data = query.get_data()
    query.close()
