import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import missingno as msno

class Processing:
    def __init__(self, df):
        self.df = df
        self.numeric_cols = ['Amount','Value','PricingStrategy','FraudResult']
        self.categorical_cols = df.select_dtypes(include=['object', 'category'])

    def numeric_distributions(self):
        sns.set(style="whitegrid")

        for col in self.numeric_cols:
            plt.figure(figsize=(12, 5))

            # Histogram + KDE
            plt.subplot(1, 2, 1)
            sns.histplot(self.df[col], bins=30, kde=True, color='skyblue')
            plt.title(f'Distribution of {col}')
            plt.xlabel(col)
            plt.ylabel('Frequency')

            # Boxplot
            plt.subplot(1, 2, 2)
            sns.boxplot(x=self.df[col], color='salmon')
            plt.title(f'Boxplot of {col}')
            plt.xlabel(col)

            plt.tight_layout()
            plt.show()


    def categorical_distributions(self, column=None, max_plots=5):
        """
        Display distribution plots for categorical variables.
        """
        sns.set(style="whitegrid")

        if column:
            # Validate column
            if column not in self.df.columns:
                raise ValueError(f"Column '{column}' not found in DataFrame.")
            if column not in self.categorical_cols.columns:
                raise ValueError(f"Column '{column}' is not in categorical columns.")

            # Plot single column
            plt.figure(figsize=(8, 5))
            order = self.df[column].value_counts().index
            sns.countplot(data=self.df, x=column, order=order)  # Removed `palette` to avoid warning
            plt.title(f'Distribution of {column}')
            plt.xlabel(column)
            plt.ylabel("Count")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

        else:
            # Plot multiple columns (default behavior)
            cols = self.categorical_cols.columns[:max_plots]
            n = len(cols)
            ncols = 2
            nrows = (n + 1) // ncols

            fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5 * nrows))
            axes = axes.flatten()

            for i, col in enumerate(cols):
                order = self.df[col].value_counts().index
                sns.countplot(data=self.df, x=col, order=order, ax=axes[i])  # Removed `palette`
                axes[i].set_title(f'Distribution of {col}')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel("Count")
                axes[i].tick_params(axis='x', rotation=45)

            # Hide any unused subplots
            for j in range(i + 1, len(axes)):
                fig.delaxes(axes[j])

            plt.tight_layout()
            plt.show()

    def plot_correlation_matrix(self):
        # Use self.df to subset numeric columns and compute correlation
        corr_matrix = self.df[self.numeric_cols].corr(method='pearson')

        plt.figure(figsize=(12, 8))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, linewidths=0.5)
        plt.title("Correlation Matrix of Numerical Features", fontsize=14)
        plt.tight_layout()
        plt.show()

    def missing_values(self):
        print("\n--- Missing Values Summary ---\n")
        missing_count = self.df.isnull().sum()
        missing_percent = (missing_count / len(self.df)) * 100
        missing_df = pd.DataFrame({
            'Missing Count': missing_count,
            'Missing %': missing_percent
        }).sort_values(by='Missing %', ascending=False)

        print(missing_df[missing_df['Missing Count'] > 0])

        # Visualize missingness
        print("\nVisualizing Missingness (useful for pattern detection):")
        msno.matrix(self.df)
        plt.show()

        #msno.heatmap(self.df)
        #plt.show()

    def detect_outliers(self):
        print("\n--- Outlier Detection Using Boxplots ---\n")
        for col in self.numeric_cols:
            plt.figure(figsize=(8, 4))
            sns.boxplot(x=self.df[col], color='orange')
            plt.title(f'Outlier Detection for {col}', fontsize=12)
            plt.xlabel(col)
            plt.tight_layout()
            plt.show()