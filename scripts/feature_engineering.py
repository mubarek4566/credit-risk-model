# src/aggregate.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer

class FeatureEngineering:
    def __init__(self, df):
        self.df = df

    def create_aggregate_features(self):
        """Creates aggregate features for each customer."""
        #self.df = df
        self.df['TotalTransactionAmount'] = self.df.groupby('CustomerId')['Amount'].transform('sum')
        self.df['AvgTransactionAmount'] = self.df.groupby('CustomerId')['Amount'].transform('mean')
        self.df['TransactionCount'] = self.df.groupby('CustomerId')['TransactionId'].transform('count')
        self.df['StdTransactionAmount'] = self.df.groupby('CustomerId')['Amount'].transform('std')
        return self.df

    def extract_datetime_features(self):
        """Extracts date and time features from TransactionStartTime."""
        #self.df = df 
        self.df['TransactionStartTime'] = pd.to_datetime(self.df['TransactionStartTime'])
        self.df['TransactionHour'] = self.df['TransactionStartTime'].dt.hour
        self.df['TransactionDay'] = self.df['TransactionStartTime'].dt.day
        self.df['TransactionMonth'] = self.df['TransactionStartTime'].dt.month
        self.df['TransactionYear'] = self.df['TransactionStartTime'].dt.year
        return self.df

    def encode_categorical_variables(self, method='onehot'):
        """Encodes categorical variables using One-Hot or Label Encoding."""
        #self.df = df
        categorical_columns = ['ProviderId', 'ProductId', 'ProductCategory', 'ChannelId']

        if method == 'onehot':
            encoder = OneHotEncoder(sparse_output=False, drop='first')  # Use sparse_output instead of sparse
            encoded = encoder.fit_transform(self.df[categorical_columns])
            encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_columns))
            self.df = pd.concat([self.df.reset_index(drop=True), encoded_df], axis=1).drop(columns=categorical_columns)

        elif method == 'label':
            for col in categorical_columns:
                le = LabelEncoder()
                self.df[col] = le.fit_transform(self.df[col])

        return self.df

    def handle_missing_values(self, method='imputation'):
        """Handles missing values by imputation or removal."""
        #self.df = df
        if method == 'imputation':
            # Separate numeric and non-numeric columns
            numeric_cols = self.df.select_dtypes(include=['number']).columns
            non_numeric_cols = self.df.select_dtypes(exclude=['number']).columns

            # Impute numeric columns with mean
            imputer = SimpleImputer(strategy='mean')
            self.df[numeric_cols] = imputer.fit_transform(self.df[numeric_cols])

            # Impute non-numeric columns with the most frequent value
            if non_numeric_cols.size > 0:
                imputer = SimpleImputer(strategy='most_frequent')
                self.df[non_numeric_cols] = imputer.fit_transform(self.df[non_numeric_cols])

        elif method == 'removal':
            self.df = self.df.dropna()

        return self.df

    def normalize_or_standardize(self, method='standardize'):
        """Normalizes or standardizes numerical features."""
        #self.df = df
        numerical_columns = self.df.select_dtypes(include=['float64', 'int64']).columns

        if method == 'normalize':
            scaler = MinMaxScaler()
            self.df[numerical_columns] = scaler.fit_transform(self.df[numerical_columns])

        elif method == 'standardize':
            scaler = StandardScaler()
            self.df[numerical_columns] = scaler.fit_transform(self.df[numerical_columns])

        return self.df

