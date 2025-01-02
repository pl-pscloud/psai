import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp, chisquare

def compare_datasets(train_df, test_df):
    # Style and palette
    sns.set_style("whitegrid")
    palette = sns.color_palette('rocket')

    numeric_cols = train_df.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = train_df.select_dtypes(include=['object', 'category']).columns

    results = {}

    # Numeric comparisons
    for col in numeric_cols:
        train_values = train_df[col].dropna()
        test_values = test_df[col].dropna()

        stat, p_value = ks_2samp(train_values, test_values)
        results[col] = {
            'test': 'KS',
            'stat': stat,
            'p_value': p_value,
            'train_mean': train_values.mean(),
            'test_mean': test_values.mean(),
            'train_std': train_values.std(),
            'test_std': test_values.std()
        }

        plt.figure(figsize=(10,5))
        # Use palette colors and alpha for transparency
        sns.histplot(train_values, kde=True, color=palette[0], label='train', alpha=0.5, stat='density')
        sns.histplot(test_values, kde=True, color=palette[5], label='test', alpha=0.5, stat='density')
        plt.title(f'Distribution of {col}')
        plt.legend()
        plt.show()

    # Categorical comparisons
    for col in cat_cols:
        train_counts = train_df[col].value_counts()
        test_counts = test_df[col].value_counts()

        all_categories = sorted(list(set(train_counts.index).union(set(test_counts.index))))
        train_freqs = train_counts.reindex(all_categories, fill_value=0)
        test_freqs = test_counts.reindex(all_categories, fill_value=0)

        # Adjust expected frequencies to match the observed sum
        ratio = train_freqs.sum() / test_freqs.sum() if test_freqs.sum() != 0 else 1
        stat, p_value = chisquare(train_freqs, f_exp=(test_freqs * ratio), axis=None)
        results[col] = {
            'test': 'Chi-square',
            'stat': stat,
            'p_value': p_value,
            'train_proportions': (train_freqs / len(train_df)).to_dict(),
            'test_proportions': (test_freqs / len(test_df)).to_dict()
        }

        train_cat = pd.DataFrame({col: train_df[col].dropna(), 'dataset': 'train'})
        test_cat = pd.DataFrame({col: test_df[col].dropna(), 'dataset': 'test'})
        combined_cat = pd.concat([train_cat, test_cat], ignore_index=True)

        plt.figure(figsize=(10,5))
        sns.countplot(data=combined_cat, x=col, hue='dataset', palette=palette, alpha=0.7)
        plt.title(f'Category distribution for {col}')
        plt.xticks(rotation=90)
        plt.show()