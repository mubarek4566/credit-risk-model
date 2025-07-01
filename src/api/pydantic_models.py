from pydantic import BaseModel
from typing import Optional

class CustomerFeatures(BaseModel):
    total_amount: float
    avg_amount: float
    transaction_count: int
    std_amount: float
    total_value: float
    distinct_product_categories: int
    fraud_count: int
    fraud_ratio: float
    transaction_hour: Optional[int] = None
    transaction_day: Optional[int] = None
    transaction_month: Optional[int] = None
    transaction_year: Optional[int] = None

class PredictionResponse(BaseModel):
    risk_probability: float
    is_high_risk: int
