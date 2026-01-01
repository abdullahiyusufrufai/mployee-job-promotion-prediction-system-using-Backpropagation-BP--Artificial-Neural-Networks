#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 17:51:00 2024

@author: rufai
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore


# Load the dataset
path = '/home/rufai/Desktop/engrKhalid/train.csv'

data = pd.read_csv(path)
# Create the 'training_effectiveness' feature by combining 'no_of_trainings' and 'avg_training_score'
data['training_impact'] = data['no_of_trainings'] * data['avg_training_score']

# Save the DataFrame with the new feature to a CSV file
data.to_csv('updated_data.csv', index=False)

data = pd.read_csv('/home/rufai/Desktop/engrKhalid/updated_data.csv')

# Fill missing 'previous_year_rating' with the mean of the column
data['previous_year_rating'].fillna(data['previous_year_rating'].mean(), inplace=True)

# Replace missing 'education' with 'Bachelor\'s'
data['education'].fillna('Bachelor\'s', inplace=True)


## EXPLORATORY DATA ANALYSIS


# 1. Distribution of Target Variable (is_promoted)
plt.figure(figsize=(8, 6))
sns.countplot(data=data, x='is_promoted', palette='coolwarm')
plt.title('Distribution of Promotion (Target Variable)')
plt.xlabel('Is Promoted')
plt.ylabel('Count')
plt.show()

'''# 2. Correlation Heatmap for Numeric Features
plt.figure(figsize=(10, 8))
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()'''


# Ensure only numeric columns are used for correlation
numeric_columns = data.select_dtypes(include=['float64', 'int64'])
correlation_matrix = numeric_columns.corr()

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()


# 3. Age Distribution by Department
plt.figure(figsize=(12, 6))
sns.boxplot(data=data, x='department', y='age', palette='Set3')
plt.title('Age Distribution by Department')
plt.xticks(rotation=45)
plt.show()

# 4. Training Scores vs. Promotion (Violin Plot)
plt.figure(figsize=(10, 6))
sns.violinplot(data=data, x='is_promoted', y='avg_training_score', palette='muted')
plt.title('Training Scores by Promotion Status')
plt.xlabel('Is Promoted')
plt.ylabel('Average Training Score')
plt.show()


# 5. Count of Employees by Education Level
plt.figure(figsize=(8, 6))
sns.countplot(data=data, x='education', order=data['education'].value_counts().index, palette='viridis')
plt.title('Count of Employees by Education Level')
plt.xticks(rotation=45)
plt.xlabel('Education Level')
plt.ylabel('Count')
plt.show()

# 6. Number of Awards vs. Promotion
plt.figure(figsize=(8, 6))
sns.barplot(data=data, x='awards_won?', y='is_promoted', ci=None, palette='cool')
plt.title('Awards Won vs. Promotion')
plt.xlabel('Awards Won')
plt.ylabel('Average Promotion Rate')
plt.show()