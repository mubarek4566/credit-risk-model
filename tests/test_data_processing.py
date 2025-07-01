import pandas as pd
import pytest
from feature_engineering import FeatureEngineering

def test_datetime_extraction():
    data = pd.DataFrame({
        'TransactionStartTime': ['2023-01-01 10:30:00', '2023-01-02 15:45:00']
    })
    
    fe = FeatureEngineering()
    result = fe.extract_datetime_features(data.copy())
    
    assert 'transaction_hour' in result.columns
    assert 'transaction_day' in result.columns
    assert result['transaction_hour'].tolist() == [10, 15]
    assert result['transaction_day'].tolist() == [1, 2]

def test_transaction_aggregation():
    data = pd.DataFrame({
        'CustomerId': [1, 1, 2],
        'Amount': [100, 150, 200],
        'Value': [120, 130, 210],
        'TransactionId': [11, 12, 13],
        'ProductCategory': ['A', 'B', 'A'],
        'FraudResult': [0, 1, 0]
    })

    fe = FeatureEngineering()
    result = fe.aggregate_features(data.copy())
    
    assert 'total_amount' in result.columns
    assert result.loc[result['CustomerId'] == 1, 'transaction_count'].values[0] == 2
    assert result.loc[result['CustomerId'] == 2, 'transaction_count'].values[0] == 1
