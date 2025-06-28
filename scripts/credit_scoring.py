import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

class CreditScoring:
    def __init__(self, df):
        self.df = df

    def calculate_rfm_metrics(self):
        """
        Calculate RFM metrics and create a credit risk column based on disengaged customers.
        """
        # Convert TransactionStartTime to datetime
        self.df['TransactionStartTime'] = pd.to_datetime(self.df['TransactionStartTime'])
        
        # Define snapshot date (1 day after the last transaction in dataset)
        snapshot_date = self.df['TransactionStartTime'].max() + pd.Timedelta(days=1)
        
        # Calculate Recency for each transaction
        self.df['Recency'] = self.df['TransactionStartTime'].apply(lambda x: (snapshot_date - x).days)

        # Calculate Frequency for each transaction
        self.df['Frequency'] = self.df.groupby('CustomerId')['TransactionId'].transform('count')

        # Monetary remains the Amount for each transaction
        self.df['Monetary'] = self.df['Amount']

        # Size: Number of unique subscriptions per CustomerId
        self.df['No_Subscription'] = self.df.groupby('CustomerId')['SubscriptionId'].transform('nunique')

        # Number of accounts per CustomerId
        self.df['No_Account'] = self.df.groupby('CustomerId')['AccountId'].transform('nunique')
        
         # Calculate RFMS score for each transaction
        self.df['RFMS_Score'] = (
            self.df['Recency'] * 0.25 +
            self.df['Frequency'] * 0.25 +
            self.df['Monetary'] * 0.25 +
            self.df['No_Subscription'] * 0.25
        )

        # Sorting the transactions by RFMS score in descending order
        self.rfms_scores = self.df.sort_values(by='RFMS_Score', ascending=False)
        return self.rfms_scores
    
    def cluster_customers_rfm(self):
        """
        Cluster customers into 3 groups based on RFM metrics using K-Means.
        """
        # Extract RFM features
        #rfm_features = self.df[['CustomerId', 'Recency', 'Frequency', 'Monetary']].drop_duplicates()
        rfm_features = self.df.groupby('CustomerId').agg({
            'Recency': 'min',              # or last if appropriate
            'Frequency': 'sum',            # or mean, depending on definition
            'Monetary': 'sum'              # total transaction value
        }).reset_index()
        
        # Pre-processing: Log transform to handle skewness (especially for Monetary and Frequency)
        rfm_features['Recency_log'] = np.log1p(rfm_features['Recency'])
        rfm_features['Frequency_log'] = np.log1p(rfm_features['Frequency'])
        rfm_features['Monetary_log'] = np.log1p(rfm_features['Monetary'].abs())
        
        # Standardize features
        scaler = StandardScaler()
        rfm_scaled = scaler.fit_transform(rfm_features[['Recency_log', 'Frequency_log', 'Monetary_log']])
        rfm_features[['Recency_scaled', 'Frequency_scaled', 'Monetary_scaled']] = rfm_scaled
        
        # Perform K-Means clustering
        kmeans = KMeans(n_clusters=3, random_state=42)
        rfm_features['Cluster'] = kmeans.fit_predict(rfm_scaled)
        
        # Merge cluster labels back to original dataframe
        #self.df = self.df.merge(rfm_features[['CustomerId', 'Cluster']], on='CustomerId', how='left')
        self.df = self.df.merge(rfm_features[['CustomerId', 'Cluster']].drop_duplicates(), on='CustomerId', how='left')
        self.rfm_features = rfm_features
        
        return self.df, rfm_features


    def plot_rfm_clusters(self):
        """
        Plot RFM clusters from the latest rfm_features.
        Assumes self.rfm_features was saved in cluster_customers_rfm.
        """
        rfm_features = self.rfm_features  # <- make sure it's saved beforehand

        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        sns.scatterplot(data=rfm_features, x='Recency', y='Frequency', hue='Cluster', palette='viridis')
        plt.title('Recency vs Frequency')

        plt.subplot(1, 3, 2)
        sns.scatterplot(data=rfm_features, x='Frequency', y='Monetary', hue='Cluster', palette='viridis')
        plt.title('Frequency vs Monetary')

        plt.subplot(1, 3, 3)
        sns.scatterplot(data=rfm_features, x='Recency', y='Monetary', hue='Cluster', palette='viridis')
        plt.title('Recency vs Monetary')

        plt.tight_layout()
        plt.show()
