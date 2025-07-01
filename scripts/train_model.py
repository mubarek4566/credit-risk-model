import joblib
import os
os.environ["SCIPY_ARRAY_API"] = "1"
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE  # <- NEW

import matplotlib.pyplot as plt
import seaborn as sns


class RiskPrediction:
    def __init__(self, df, sampling_strategy='smote'):
        self.df = df
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.log_reg = None
        self.rf = None
        self.XGB = None
        self.sampling_strategy = sampling_strategy.lower()

    def split_data(self):
        X = self.df.drop(['is_high_risk'], axis=1)
        y = self.df['is_high_risk']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Apply sampling to training set only
        if self.sampling_strategy == 'smote':
            smote = SMOTE(random_state=42)
            self.X_train, self.y_train = smote.fit_resample(X_train, y_train)
        elif self.sampling_strategy == 'oversample':
            from imblearn.over_sampling import RandomOverSampler
            ros = RandomOverSampler(random_state=42)
            self.X_train, self.y_train = ros.fit_resample(X_train, y_train)
        elif self.sampling_strategy == 'undersample':
            from imblearn.under_sampling import RandomUnderSampler
            rus = RandomUnderSampler(random_state=42)
            self.X_train, self.y_train = rus.fit_resample(X_train, y_train)
        else:
            self.X_train, self.y_train = X_train, y_train

        self.X_test, self.y_test = X_test, y_test

        print("✅ Data split and sampled:")
        print(f"Train size: {self.X_train.shape}, Test size: {self.X_test.shape}")
        print("Train target distribution:\n", self.y_train.value_counts())

    def train_logistic_regression(self):
        self.log_reg = LogisticRegression(random_state=42, max_iter=1000)
        self.log_reg.fit(self.X_train, self.y_train)
        self.log_reg_preds = self.log_reg.predict(self.X_test)
        return self.log_reg

    def train_random_forest(self):
        self.rf = RandomForestClassifier(random_state=42)
        self.rf.fit(self.X_train, self.y_train)
        self.rf_preds = self.rf.predict(self.X_test)
        return self.rf

    def train_XGBoost(self):
        self.XGB = GradientBoostingClassifier(random_state=42)
        self.XGB.fit(self.X_train, self.y_train)
        self.XGB_preds = self.XGB.predict(self.X_test)
        return self.XGB

    def LR_models_evaluate(self):
        print("Logistic Regression:")
        print("Accuracy:", accuracy_score(self.y_test, self.log_reg_preds))
        print("Precision:", precision_score(self.y_test, self.log_reg_preds))
        print("Recall:", recall_score(self.y_test, self.log_reg_preds))
        print("F1 Score:", f1_score(self.y_test, self.log_reg_preds))
        print("ROC-AUC:", roc_auc_score(self.y_test, self.log_reg.predict_proba(self.X_test)[:, 1]))

    def RF_models_evaluate(self):
        print("\nRandom Forest:")
        print("Accuracy:", accuracy_score(self.y_test, self.rf_preds))
        print("Precision:", precision_score(self.y_test, self.rf_preds))
        print("Recall:", recall_score(self.y_test, self.rf_preds))
        print("F1 Score:", f1_score(self.y_test, self.rf_preds))
        print("ROC-AUC:", roc_auc_score(self.y_test, self.rf.predict_proba(self.X_test)[:, 1]))

    def XGB_models_evaluate(self):
        print("\nXGB Boosting:")
        print("Accuracy:", accuracy_score(self.y_test, self.XGB_preds))
        print("Precision:", precision_score(self.y_test, self.XGB_preds))
        print("Recall:", recall_score(self.y_test, self.XGB_preds))
        print("F1 Score:", f1_score(self.y_test, self.XGB_preds))
        print("ROC-AUC:", roc_auc_score(self.y_test, self.XGB.predict_proba(self.X_test)[:, 1]))


    '''
    def evaluate_models(self):
        # Logistic Regression Predictions
        log_reg_preds = self.log_reg.predict(self.X_test)

        # Random Forest Predictions
        rf_preds = self.rf.predict(self.X_test)

        self.XGB_preds = self.XGB.predict(self.X_test)

        # Random Forest Predictions
        #rf_preds_g = self.best_rf.predict(self.X_test)

        # Random Forest Predictions
        #rf_preds_r = self.best_rf_r.predict(self.X_test)
        
        # Grid Search Random Forest Evaluation
        print("\nRandom Forest:")
        print("Accuracy:", accuracy_score(self.y_test, rf_preds_g))
        print("Precision:", precision_score(self.y_test, rf_preds_g))
        print("Recall:", recall_score(self.y_test, rf_preds_g))
        print("F1 Score:", f1_score(self.y_test, rf_preds_g))
        print("ROC-AUC:", roc_auc_score(self.y_test, self.best_rf.predict_proba(self.X_test)[:, 1]))

        # Random Search Random Forest Evaluation
        print("\nRandom Forest:")
        print("Accuracy:", accuracy_score(self.y_test, rf_preds_r))
        print("Precision:", precision_score(self.y_test, rf_preds_r))
        print("Recall:", recall_score(self.y_test, rf_preds_r))
        print("F1 Score:", f1_score(self.y_test, rf_preds_r))
        print("ROC-AUC:", roc_auc_score(self.y_test, self.best_rf_r.predict_proba(self.X_test)[:, 1])) 

        
        return log_reg_preds, rf_preds, rf_preds_g, rf_preds_r  
        '''