import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

class FeatureVisualizer:
    def __init__(self, df):
        self.df = df
        self.numeric_cols = df.select_dtypes(include=[np.number])
        self.categorical_cols = df.select_dtypes(include=['object', 'category'])

    def numeric_distributions(self):
        sns.set(style="whitegrid")

        for col in self.numeric_cols.columns:
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

    def categorical_distributions(self, max_plots=5):

        sns.set(style="whitegrid")
        cols = self.categorical_cols.columns[:max_plots]  # Limit number of columns to plot

        n = len(cols)
        ncols = 2
        nrows = (n + 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5 * nrows))
        axes = axes.flatten()

        for i, col in enumerate(cols):
            order = self.df[col].value_counts().index
            sns.countplot(data=self.df, x=col, order=order, palette='pastel', ax=axes[i])
            axes[i].set_title(f'Distribution of {col}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel("Count")
            axes[i].tick_params(axis='x', rotation=45)

        # Hide any unused subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()
