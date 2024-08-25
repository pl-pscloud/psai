import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics as stat

import seaborn as sns
palette = ["#365486", "#7FC7D9", "#0F1035", "#7A316F", "#7A316F"]
sns.set_palette(palette=palette)

import warnings
warnings.filterwarnings('ignore')

def dist(df):
    for col in df.select_dtypes(include=[np.number]).columns:
        # Create a figure and axes. Then plot the data
        fig, ax = plt.subplots(figsize=(8,5))
        sns.distplot(df[col], ax=ax,rug=True, kde_kws={'shade':True})

        # Customize the labels and limits
        ax.set(xlabel=col, title="Var: "+col)

        median = stat.median(df[col])
        mean = stat.mean(df[col])

        # Add vertical lines for the median and mean
        ax.axvline(x=median, color='m', label='Median', linestyle='--', linewidth=1)
        ax.axvline(x=mean, color='b', label='Mean', linestyle='-', linewidth=1)

        # Show the legend and plot the data
        ax.legend()
        plt.show()
        plt.clf()
        
def reg(df, y_name):
    for col in df.columns:
        plt.figure(figsize=(8,5))
        sns.regplot(data=df, x=col, y=y_name)
        plt.show()
        plt.clf()

def box(df):
    for col in df.columns:
        plt.figure(figsize=(8,5))
        sns.boxplot(data=df, y=col)
        plt.show()
        plt.clf()

def pairCategory(df):
    for col in df.select_dtypes(include=[np.number]).columns:
        sns.pairplot(df, hue=col, size=2.5);
    