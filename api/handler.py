import pandas as pd
import pickle
from flask import Flask, request, Response
from fraudblocker.FraudBlocker import FraudBlocker

# model
model = pickle.load(open('/mnt/c/wsl/Comunidade_DS/Projetos/PA002 - Blocker Fraud Company/fraud_blocker/models/model.pkl', 'rb'))

# initialize API
app = Flask(__name__)

# create endpoint (URL), post method -> send data in order to receive data, the route() decorator tells Flask what URL will trigger the function
@app.route('/fraudblocker/predict', methods=['POST'])

# function to get the received data
def fraudblocker_predict():
    test_json = request.get_json()
    
    # check received data (json)
    if test_json: # there is data
        # test if 'test_json' is a dictionaire
        if isinstance(test_json, dict):
            test_raw = pd.DataFrame(test_json, index=[0])
            
        else:
            test_raw = pd.DataFrame(test_json, columns=test_json[0].keys()) # keys = json keys. They'll become the df columns
        
        # instantiate FraudBlocker class
        pipeline = FraudBlocker()
        
        # data cleaning
        df1 = pipeline.data_cleaning(test_raw)
        
        # feature engineering
        df2 = pipeline.feature_engineering(df1)
        test_return = df2.copy()
        
        # data preparation
        df3 = pipeline.data_preparation(df2)
        
        # prediction
        df_prediction = pipeline.get_predictions(model, test_return, df3)
        
        return df_prediction
    
    else: # there is no data
        return Response('{}', status=200, mimetype='application/json')
    
# check main function in the script
if __name__ == '__main__':
    app.run('0.0.0.0') # local host
