#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 16:55:13 2024

@author: rufai
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore


# Load the dataset
train_path = '/home/rufai/Desktop/engrKhalid/train.csv'
test_path = '/home/rufai/Desktop/engrKhalid/test.csv'
train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)


# Create the 'training_effectiveness' feature by combining 'no_of_trainings' and 'avg_training_score'
train_data['training_impact'] = train_data['no_of_trainings'] * train_data['avg_training_score']
test_data['training_impact'] = test_data['no_of_trainings'] * test_data['avg_training_score']


save_dir = '/home/rufai/Desktop/engrKhalid/datd_summary_befor_cleaning'
data_summary = train_data.describe()
data_summary.to_csv(save_dir, index = False)


data_info = train_data.info()
print('data infomation data type: ', type(data_info))

# getting missing value
missing_values = train_data.isnull().sum()
#print(type(missing_values))


# Convert to a DataFrame for saving to CSV
missing_values_df = missing_values[missing_values > 0].reset_index()
missing_values_df.columns = ['Column', 'Missing Values']

# Save the missing values summary as a CSV file
missing_values_df.to_csv('/home/rufai/Desktop/engrKhalid/missing_values_summary.csv', index=False)

print("Missing values summary has been saved as 'missing_values_summary.csv'.")



# Compute mean, mode, and median for 'education' and 'previous_year_rating'

# For 'education' (categorical, only mode is meaningful)
education_level_mode = train_data['education'].mode()[0]

# For 'previous_year_rating' (numeric)
previous_year_rating_mean = train_data['previous_year_rating'].mean()
previous_year_rating_mode = train_data['previous_year_rating'].mode()[0]
previous_year_rating_median = train_data['previous_year_rating'].median()

# Display the results
print("Education:")
print(f"Mode: {education_level_mode}\n")

print("Previous Year Rating:")
print(f"Mean: {previous_year_rating_mean}")
print(f"Mode: {previous_year_rating_mode}")
print(f"Median: {previous_year_rating_median}")


previous_year_rating_min = train_data['previous_year_rating'].min()
previous_year_rating_max = train_data['previous_year_rating'].max()

print("previous year rating minimum: ", previous_year_rating_min)
print("previous year rating maximum: ", previous_year_rating_max)


## checking for outlier in the prevous year rating IQR

# Calculate the IQR
Q1 = train_data['previous_year_rating'].quantile(0.25)  # First quartile
Q3 = train_data['previous_year_rating'].quantile(0.75)  # Third quartile
IQR = Q3 - Q1  # Interquartile Range

# Define lower and upper bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identify outliers
outliers = train_data[(train_data['previous_year_rating'] < lower_bound) | 
                      (train_data['previous_year_rating'] > upper_bound)]

# Display results
print(f"Lower bound: {lower_bound}")
print(f"Upper bound: {upper_bound}")
print(f"Number of outliers: {len(outliers)}")



# Plot boxplot
plt.figure(figsize=(8, 6))
plt.boxplot(train_data['previous_year_rating'], vert=False, patch_artist=True)
plt.title('Boxplot of Previous Year Rating')
plt.xlabel('Previous Year Rating')
plt.show()



# Create a violin plot
plt.figure(figsize=(8, 6))
sns.violinplot(data=train_data, x='previous_year_rating', inner='quartile')
plt.title('Violin Plot of Previous Year Rating')
plt.xlabel('Previous Year Rating')
plt.show()

# Create a scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(train_data.index, train_data['previous_year_rating'], alpha=0.6, color='b')
plt.axhline(train_data['previous_year_rating'].quantile(0.25), color='r', linestyle='--', label='Q1 (25th Percentile)')
plt.axhline(train_data['previous_year_rating'].quantile(0.75), color='g', linestyle='--', label='Q3 (75th Percentile)')
plt.title('Scatter Plot of Previous Year Rating')
plt.xlabel('Index')
plt.ylabel('Previous Year Rating')
plt.legend()
plt.show()



## checking for outlier in the entire datasets

# Identify numeric columns
numeric_columns = train_data.select_dtypes(include=['float64', 'int64']).columns

# Check for outliers using the IQR method
outlier_summary = {}

for column in numeric_columns:
    Q1 = train_data[column].quantile(0.25)
    Q3 = train_data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = train_data[(train_data[column] < lower_bound) | 
                    (train_data[column] > upper_bound)]
    outlier_summary[column] = len(outliers)

# Convert outlier summary to a DataFrame
outlier_summary_df = pd.DataFrame(list(outlier_summary.items()), 
                                  columns=['Column', 'Outlier Count'])

# Plot boxplots for numeric columns
plt.figure(figsize=(12, 8))
train_data[numeric_columns].boxplot(rot=45, patch_artist=True)
plt.title('Boxplots of Numeric Columns')
plt.xlabel('Columns')
plt.ylabel('Values')
plt.show()

# Display outlier summary
outlier_summary_df

# Scatter plot for all numeric columns

plt.figure(figsize=(20, 15))

for i, column in enumerate(numeric_columns):
    plt.subplot(len(numeric_columns), 1, i + 1)
    plt.scatter(train_data.index, train_data[column], alpha=0.6, color='b')
    plt.axhline(train_data[column].quantile(0.25), color='r', linestyle='--', 
                label='Q1 (25th Percentile)')
    plt.axhline(train_data[column].quantile(0.75), color='g', linestyle='--', 
                label='Q3 (75th Percentile)')
    plt.title(f'Scatter Plot of {column}')
    plt.xlabel('Index')
    plt.ylabel(column)
    plt.legend()

plt.tight_layout()
plt.show()




# Compute z-scores for numeric columns
z_scores = train_data[numeric_columns].apply(zscore)

# Create a heatmap of z-scores
plt.figure(figsize=(12, 8))
sns.heatmap(z_scores, cmap='coolwarm', cbar=True, center=0)
plt.title('Heatmap of Z-Scores for Numeric Columns')
plt.xlabel('Columns')
plt.ylabel('Index')
plt.show()


# Iterate through all numeric columns and compute summary statistics
numeric_columns = train_data.select_dtypes(include=['float64', 'int64']).columns

# Dictionary to store results
summary_stats = {}

# Calculate statistics for each numeric column
for column in numeric_columns:
    stats = {
        'Mean': train_data[column].mean(),
        'Mode': train_data[column].mode()[0] if not train_data[column].mode().empty else None,
        'Median': train_data[column].median(),
        'Min': train_data[column].min(),
        'Max': train_data[column].max()
    }
    summary_stats[column] = stats

# Convert the results to a DataFrame for better visualization
summary_stats_df = pd.DataFrame(summary_stats).T

# Display the summary statistics
print("Summary Statistics for Numeric Columns:")
print(summary_stats_df)

# Save the summary statistics to a CSV file
summary_stats_df.to_csv('/home/rufai/Desktop/engrKhalid/numeric_columns_summary.csv', index=True)
print("Summary statistics have been saved as 'numeric_columns_summary.csv'.")


print(train_data)


# Fill missing 'previous_year_rating' with the mean of the column
train_data['previous_year_rating'].fillna(train_data['previous_year_rating'].mean(), inplace=True)

# Replace missing 'education' with 'Bachelor\'s'
train_data['education'].fillna('Bachelor\'s', inplace=True)


