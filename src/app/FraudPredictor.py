import pandas as pd 
import pickle 
import requests


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

    def load_model(self):
        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)

    def get_requests(self):
        self.response = requests.post(self.url, json={'api_key': self.api_key,
                                    'sequence_number': self.sequence_number})
        self.raw_data = self.response.json()


if __name__=='__main__':
    home = '/home/danny/Desktop/galvanize/w10/dsi-fraud-detection-case-study/'
    model_path = home + 'src/model.pkl'
    fp = FraudPredictor(model_path)
    fp.run_analysis()
