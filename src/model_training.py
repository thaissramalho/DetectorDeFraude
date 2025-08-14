import os
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from src.data_preparation import load_data, prepare_data

def train_model():
    df = load_data()
    X_train, X_test, y_train, y_test = prepare_data(df)

    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Cria a pasta 'models' se ainda não existir
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/xgboost_fraud_model.pkl')

if __name__ == '__main__':
    train_model()
