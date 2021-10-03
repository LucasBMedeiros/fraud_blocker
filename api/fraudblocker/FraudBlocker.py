import pickle 
import pandas as pd
import numpy as np
import inflection

# fraud detection definition class
class FraudBlocker(object):
    def __init__(self):
        
        # file path
        path = '/mnt/c/wsl/Comunidade_DS/Projetos/PA002 - Blocker Fraud Company/fraud_blocker/'
        
        # load scalers
        self.step = pickle.load(open(path + 'parameters/step.pkl', 'rb'))
        self.amount = pickle.load(open(path + 'parameters/amount.pkl', 'rb'))
        self.newbalance_dest = pickle.load(open(path + 'parameters/newbalance_dest.pkl', 'rb'))
        self.oldbalance_dest = pickle.load(open(path + 'parameters/oldbalance_dest.pkl', 'rb'))
        self.newbalance_orig = pickle.load(open(path + 'parameters/newbalance_orig.pkl', 'rb'))
        self.oldbalance_org = pickle.load(open(path + 'parameters/oldbalance_org.pkl', 'rb'))
        self.diff_dest = pickle.load(open(path + 'parameters/diff_dest.pkl', 'rb'))
        self.diff_orig = pickle.load(open(path + 'parameters/diff_orig.pkl', 'rb'))
       
    def data_cleaning(self, df1):
        old_cols = df1.columns
        snakecase = lambda x: inflection.underscore(x)
        new_cols = list(map(snakecase, old_cols))
        df1.columns = new_cols
        
        return df1       
        
    def feature_engineering(self, df2):
        # original balance difference feature
        df2['diff_orig'] = df2['oldbalance_org'] - df2['amount'] - df2['newbalance_orig']
        
        # destination balance difference feature
        df2['diff_dest'] = df2['oldbalance_dest'] + df2['amount'] - df2['newbalance_dest']
        
        # Categorical feature for merchant
        df2['merchant_dest'] = df2.apply(lambda x: 1 if 'M' in x['name_dest'] else 0, axis=1)
        
        # time feature for days, from 'step'
        df2['day'] = df2.apply(lambda x: (np.ceil(x['step'] / 24)).astype(int), axis=1)
        
        df2 = df2[['step', 'day', 'type', 'name_orig', 'amount', 'oldbalance_org',
                   'newbalance_orig', 'diff_orig', 'name_dest',
                   'merchant_dest', 'oldbalance_dest', 'newbalance_dest', 'diff_dest',
                   'is_fraud', 'is_flagged_fraud']]
        
        return df2
        
    def data_preparation(self, df5):
        # step (reasonably uniform distribution)
        df5['step'] = self.step.fit_transform(df5[['step']].values)

        # amount (contains significant outliers)
        df5['amount'] = self.amount.fit_transform(df5[['amount']].values)

        # oldbalance_org (contains significant outliers)
        df5['oldbalance_org'] = self.oldbalance_org.fit_transform(df5[['oldbalance_org']].values)

        # newbalance_orig (contains significant outliers)
        df5['newbalance_orig'] = self.newbalance_orig.fit_transform(df5[['newbalance_orig']].values)

        # diff_orig (contains significant outliers)
        df5['diff_orig'] = self.diff_orig.fit_transform(df5[['diff_orig']].values)

        # oldbalance_dest (contains significant outliers)
        df5['oldbalance_dest'] = self.oldbalance_dest.fit_transform(df5[['oldbalance_dest']].values)

        # newbalance_dest (contains significant outliers)
        df5['newbalance_dest'] = self.newbalance_dest.fit_transform(df5[['newbalance_dest']].values)

        # diff_dest (contains significant outliers)
        df5['diff_dest'] = self.diff_dest.fit_transform(df5[['diff_dest']].values)

        ### Transformation - Encoding - type features - apply one hot encoding
        # type feature - apply one hot encoding
        df5 = pd.get_dummies(df5, prefix=['type'], prefix_sep='_', columns=['type'])     
                
        ### Nature Transformation - apply nature transformation on day feature
        df5['day_sin'] = df5['day'].apply(lambda x: np.sin(x*(2.*np.pi/31)))
        df5['day_cos'] = df5['day'].apply(lambda x: np.cos(x*(2.*np.pi/31)))
        
        # selected features
        feat_list =  ['newbalance_orig',
                      'diff_orig',
                      'diff_dest',
                      'newbalance_dest',
                      'amount',
                      'oldbalance_org',
                      'oldbalance_dest',
                      'type_TRANSFER',
                      'day_cos',
                      'type_CASH_OUT',
                      'day_sin']

        return df5[feat_list]
        
        
    def get_predictions(self, model, original_data, test_data_prep):
        # prediction
        pred = model.predict(test_data_prep)
        
        # join prediction in the original dataset
        original_data['prediction'] = pred
        
        return original_data.to_json(orient='records')
























