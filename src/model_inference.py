import joblib
import pandas as pd

def predict_new_transactions(path='data/new_transactions.csv'):
    model = joblib.load('models/xgboost_fraud_model.pkl')
    new_data = pd.read_csv(path)
    predictions = model.predict(new_data)
    new_data['Fraud_Prediction'] = predictions
    return new_data

if __name__ == '__main__':
    df_result = predict_new_transactions()
    print(df_result.head())
