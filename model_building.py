#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 18:30:50 2024

@author: rufai
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import os
import time

data = pd.read_csv('/home/rufai/Desktop/engrKhalid/updated_data.csv')

# Fill missing 'previous_year_rating' with the mean of the column
data['previous_year_rating'].fillna(
    data['previous_year_rating'].mean(), inplace=True)

# Replace missing 'education' with 'Bachelor\'s'
data['education'].fillna('Bachelor\'s', inplace=True)


# Step 1: Load and Preprocess Data
# Assuming 'train_data' is the dataset
X = data.drop(columns=['is_promoted'])  # Features
y = data['is_promoted']  # Target variable


# Encode categorical variables
X = pd.get_dummies(X, drop_first=True)

# Normalize numeric columns
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Convert the scaled array back to a DataFrame
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# Define the file path
file_path = os.path.join('/home/rufai/Desktop/engrKhalid', 'Scaled_data.csv')

# Save the scaled data to the specified directory
X_scaled_df.to_csv(file_path, index=False)



# Encode target variable (if not binary)
y_encoded = y.values  # Assuming binary classification (0 and 1)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, 
                                            test_size=0.2, random_state=42)

# Step 2: Build the Neural Network
model = Sequential()
# Input layer
model.add(Dense(units=64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(units=32, activation='relu'))  # Hidden layer
# Output layer for binary classification
model.add(Dense(units=1, activation='sigmoid'))

# Step 3: Compile the Model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy', metrics=['accuracy'])

# Step 4: Train the Model
history = model.fit(X_train, y_train, validation_split=0.2,
                    epochs=100, batch_size=32, verbose=1)








# Define a function to calculate training time
def measure_training_time(model, X_train, y_train, X_val, y_val, epochs):
    start_time = time.time()  # Start timing
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=32,
        verbose=1
    )
    end_time = time.time()  # End timing
    training_time = end_time - start_time
    return training_time, history

# Specify epochs to measure
epoch_values = [5, 20, 50, 100]
training_times = []

for epochs in epoch_values:
    print(f"Training for {epochs} epochs...")
    training_time, history = measure_training_time(model, X_train, y_train, X_test, y_test, epochs)
    print(f"Time for {epochs} epochs: {training_time:.2f} seconds\n")
    training_times.append(training_time)

# Display results
for epochs, t_time in zip(epoch_values, training_times):
    print(f"Epochs: {epochs}, Training Time: {t_time:.2f} seconds")




# Assume 'epoch_values' and 'training_times' contain the results from the previous code
# Example data (replace with actual results if available)
epoch_values = [5, 20, 50, 100]
training_times = [39.50, 160.02, 443.64, 1053.79]

# Plot the results
plt.figure(figsize=(8, 6))
plt.bar(epoch_values, training_times, color='skyblue', width=10, edgecolor='black')

# Add labels and title
plt.xlabel('Number of Epochs', fontsize=14)
plt.ylabel('Training Time (seconds)', fontsize=14)
plt.title('Training Time for Different Epoch Values', fontsize=16)
plt.xticks(epoch_values, fontsize=12)
plt.yticks(fontsize=12)





# Step 5: Evaluate the Model
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
print(f"Test Accuracy: {test_accuracy:.2f}")

# Plot training vs validation accuracy

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.show()





# Assuming you have your test data and model
# X_test: test features
# y_test: actual labels for the test set
# model: your trained Keras model

# Step 1: Make Predictions
y_pred_prob = model.predict(X_test)  # Predicted probabilities
y_pred = (y_pred_prob > 0.5).astype(int).flatten()  # Convert probabilities to binary predictions

# Step 2: Calculate Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Print Metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 3: Visualize the Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Promoted', 'Promoted'], yticklabels=['Not Promoted', 'Promoted'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Step 4: Visualize Accuracy vs. Validation Accuracy
history_dict = history.history  # Assuming 'history' is the object returned by model.fit
accuracy = history_dict['accuracy']
val_accuracy = history_dict['val_accuracy']

epochs = range(1, len(accuracy) + 1)

plt.figure(figsize=(8, 6))
plt.plot(epochs, accuracy, label='Training Accuracy', marker='o')
plt.plot(epochs, val_accuracy, label='Validation Accuracy', marker='o')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.show()

# Step 5: Visualize Loss vs. Validation Loss
loss = history_dict['loss']
val_loss = history_dict['val_loss']

plt.figure(figsize=(8, 6))
plt.plot(epochs, loss, label='Training Loss', marker='o')
plt.plot(epochs, val_loss, label='Validation Loss', marker='o')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()





# Step 6: Save the Model (optional)
model.save('bp_ann_model.h5')
