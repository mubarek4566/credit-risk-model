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
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import mlflow
import mlflow.sklearn



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

    def tune_random_forest(self):
        param_dist = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False]
        }

        random_search = RandomizedSearchCV(RandomForestClassifier(random_state=42),
                                        param_distributions=param_dist,
                                        n_iter=10,
                                        scoring='f1',
                                        cv=5,
                                        verbose=1,
                                        n_jobs=-1,
                                        random_state=42)
        random_search.fit(self.X_train, self.y_train)
        self.rf_r = random_search.best_estimator_
        print("Best Random Forest Params:", random_search.best_params_)

    def evaluate_tuned_random_forest(self):
        if not hasattr(self, 'rf_r'):
            raise ValueError("Random Forest model has not been tuned yet. Run tune_random_forest() first.")
        
        # Train the best model (already trained during RandomizedSearchCV)
        self.rf_r_preds = self.rf_r.predict(self.X_test)

        print("\n✅ Tuned Random Forest:")
        print("Accuracy:", accuracy_score(self.y_test, self.rf_r_preds))
        print("Precision:", precision_score(self.y_test, self.rf_r_preds))
        print("Recall:", recall_score(self.y_test, self.rf_r_preds))
        print("F1 Score:", f1_score(self.y_test, self.rf_r_preds))
        print("ROC-AUC:", roc_auc_score(self.y_test, self.rf_r.predict_proba(self.X_test)[:, 1]))

   
    def log_and_register_model(self, model, model_name: str, run_name: str):
        with mlflow.start_run(run_name=run_name):
            mlflow.sklearn.log_model(model, model_name)
            mlflow.log_params(model.get_params())
            
            preds = model.predict(self.X_test)
            mlflow.log_metric("accuracy", accuracy_score(self.y_test, preds))
            mlflow.log_metric("precision", precision_score(self.y_test, preds))
            mlflow.log_metric("recall", recall_score(self.y_test, preds))
            mlflow.log_metric("f1_score", f1_score(self.y_test, preds))
            mlflow.log_metric("roc_auc", roc_auc_score(self.y_test, model.predict_proba(self.X_test)[:, 1]))

            # Register the model to the model registry
            mlflow.sklearn.log_model(model, artifact_path=model_name, registered_model_name=model_name)
            print(f"{model_name} registered to MLflow Model Registry.")

   