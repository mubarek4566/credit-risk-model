import pandas as pd
import joblib
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

class RiskPrediction:
    def __init__(self, df):
        self.df = df
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.log_reg = None
        self.best_rf = None
    '''
    def preprocess_data(self):
        # Ensure TransactionStartTime is set as the index
        self.df['TransactionStartTime'] = pd.to_datetime(self.df['TransactionStartTime'])
        self.df.set_index('TransactionStartTime', inplace=True)

        # Encode Categorical Variables
        label_encoder = LabelEncoder()
        for col in self.df.select_dtypes(include=['object']).columns:
            self.df[col] = label_encoder.fit_transform(self.df[col])

        # Feature Scaling
        scaler = StandardScaler()
        numerical_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
        self.df[numerical_cols] = scaler.fit_transform(self.df[numerical_cols])

        # Convert Risk column: greater than 0 -> 1, else -> 0
        self.df['Risk'] = self.df['Risk'].apply(lambda x: 1 if x > 0 else 0)  '''

    def split_data(self):
        X = self.df.drop(['is_high_risk'], axis=1)
        y = self.df['is_high_risk']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print('X_train: ', self.X_train)
        print('y_train: ', self.y_train)
    
    def train_logistic_regression(self):
        self.log_reg = LogisticRegression(random_state=42, max_iter=1000)
        self.log_reg.fit(self.X_train, self.y_train)
        return self.log_reg
    
    def train_random_forest(self):
        self.rf = RandomForestClassifier(random_state=42)
        self.rf.fit(self.X_train, self.y_train)
        return self.rf
    
    def train_XGBoost(self):
        self.XGB = GradientBoostingClassifier(random_state=42)
        self.XGB.fit(self.X_train, self.y_train)
        return self.XGB
