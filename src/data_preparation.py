import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def load_data(path='Data/transactions.csv'):
    df = pd.read_csv(path)
    return df

def prepare_data(df):
    X = df.drop('Class', axis=1)
    y = df['Class']
    smote = SMOTE()
    X_res, y_res = smote.fit_resample(X, y)
    return train_test_split(X_res, y_res, test_size=0.2, random_state=42)