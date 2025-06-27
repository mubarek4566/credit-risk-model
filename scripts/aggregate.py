
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class TnxAggregator(BaseEstimator, TransformerMixin):
    def __init__(self, datetime_col='TransactionStartTime'):
        self.datetime_col = datetime_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        # Convert to datetime
        df[self.datetime_col] = pd.to_datetime(df[self.datetime_col], errors='coerce')

        # Extract datetime features
        df['transaction_hour'] = df[self.datetime_col].dt.hour
        df['transaction_day'] = df[self.datetime_col].dt.day
        df['transaction_month'] = df[self.datetime_col].dt.month
        df['transaction_year'] = df[self.datetime_col].dt.year

        # Group by CustomerId and aggregate
        grouped = df.groupby('CustomerId').agg(
            total_amount=('Amount', 'sum'),
            avg_amount=('Amount', 'mean'),
            transaction_count=('TransactionId', 'count'),
            std_amount=('Amount', 'std'),
            total_value=('Value', 'sum'),
            distinct_product_categories=('ProductCategory', 'nunique'),
            fraud_count=('FraudResult', 'sum'),
            avg_transaction_hour=('transaction_hour', 'mean'),
            avg_transaction_day=('transaction_day', 'mean'),
            avg_transaction_month=('transaction_month', 'mean'),
            most_recent_year=('transaction_year', 'max')
        ).reset_index()

        grouped['fraud_ratio'] = grouped['fraud_count'] / grouped['transaction_count']

        return grouped.fillna(0)
