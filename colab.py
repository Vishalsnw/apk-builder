import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from datetime import datetime
import joblib
from google.colab import files
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.neural_network import MLPClassifier

# Define the folder containing CSV files
folder_path = '/content/drive/MyDrive/Kalyan_Satta'

# Fetch all CSV files dynamically
csv_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]

# Select the latest CSV file based on modification time
if not csv_files:
    raise FileNotFoundError("No CSV files found in the specified directory.")
file_path = max(csv_files, key=os.path.getmtime)  # Latest file

print(f"Using file: {file_path}")

# Function to add new results to the latest CSV file
def append_to_csv():
    print("\n=== Append New Results ===")
    date_input = input("Enter date (dd/mm/yyyy): ")
    market_input = input("Enter market (e.g., Kalyan): ")
    open_input = int(input("Enter Open value: "))
    jodi_input = int(input("Enter Jodi value: "))
    close_input = int(input("Enter Close value: "))

    try:
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
        latest_jodi = df['Jodi'].iloc[-1] if not df.empty else 0
        prev_jodi_distance = abs(jodi_input - latest_jodi)
    except:
        prev_jodi_distance = 0

    day_of_week = pd.to_datetime(date_input, format='%d/%m/%Y').strftime('%A')
    is_weekend = 1 if day_of_week in ['Saturday', 'Sunday'] else 0
    open_sum = sum(int(digit) for digit in str(open_input))
    close_sum = sum(int(digit) for digit in str(close_input))
    mirror_open = int(str(open_input)[::-1])
    mirror_close = int(str(close_input)[::-1])
    reverse_jodi = int(str(jodi_input)[::-1])
    is_holiday = 0

    new_row = pd.DataFrame([{
        'Date': date_input, 'Market': market_input, 'Open': open_input, 'Jodi': jodi_input,
        'Close': close_input, 'day_of_week': day_of_week, 'is_weekend': is_weekend,
        'open_sum': open_sum, 'close_sum': close_sum, 'mirror_open': mirror_open,
        'mirror_close': mirror_close, 'reverse_jodi': reverse_jodi, 'is_holiday': is_holiday,
        'prev_jodi_distance': prev_jodi_distance
    }])

    try:
        existing_data = pd.read_csv(file_path)
        updated_data = pd.concat([existing_data, new_row], ignore_index=True)
        updated_data.to_csv(file_path, index=False)
        print("New results successfully appended to the CSV!")
    except Exception as e:
        print(f"Error while appending to CSV: {e}")

# Function to train and test models
def train_and_test_models():
    print("\n=== Train and Test Models ===")

    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
    df.dropna(subset=['Date'], inplace=True)
    df = df[df['Market'].str.lower() == 'kalyan'].copy()
    df.dropna(inplace=True)

    cutoff_date = pd.to_datetime("31/03/2025", format='%d/%m/%Y')

    feature_cols = ['open_sum', 'close_sum', 'mirror_open', 'mirror_close', 'reverse_jodi',
                    'prev_jodi_distance', 'is_weekend', 'is_holiday']

    train_data = df[df['Date'] <= cutoff_date]
    test_data = df[df['Date'] > cutoff_date]

    if train_data.empty or test_data.empty:
        raise ValueError("Insufficient data for training/testing. Check your CSV file.")

    X_train = train_data[feature_cols]
    y_train = train_data['Jodi']
    X_test = test_data[feature_cols]
    y_test = test_data['Jodi']

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model_sklearn = MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu', solver='adam',
                                  max_iter=1000, random_state=42)
    model_sklearn.fit(X_train_scaled, y_train)

    joblib.dump(model_sklearn, 'kalyan_mlp_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    print("Scikit-learn model and scaler saved.")

    le = LabelEncoder()
    le.fit(y_train)
    y_train_encoded = le.transform(y_train)
    y_test_encoded = le.transform(y_test)
    joblib.dump(le, 'label_encoder.pkl')

    num_classes = len(np.unique(y_train_encoded))

    model_keras = Sequential([
        Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model_keras.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model_keras.fit(X_train_scaled, y_train_encoded, epochs=100, validation_data=(X_test_scaled, y_test_encoded),
                    callbacks=[early_stop], verbose=2)

    model_keras.save('model.h5')
    print("Keras model saved.")

# Function to predict next day's Jodi
def predict_next_day():
    print("\n=== Predict Next Day ===")

    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
    df.dropna(subset=['Date'], inplace=True)

    scaler = joblib.load('scaler.pkl')
    model_keras = tf.keras.models.load_model('model.h5')
    le = joblib.load('label_encoder.pkl')

    latest_row = df[df['Date'] == df['Date'].max()]
    if latest_row.empty:
        print("No data found for the latest date.")
    else:
        feature_cols = ['open_sum', 'close_sum', 'mirror_open', 'mirror_close',
                        'reverse_jodi', 'prev_jodi_distance', 'is_weekend', 'is_holiday']
        latest_features = latest_row[feature_cols]
        scaled_latest = scaler.transform(latest_features)
        probs_keras = model_keras.predict(scaled_latest)[0]

        top10_indices = np.argsort(-probs_keras)[:10]
        top10_jodis_keras = le.inverse_transform(top10_indices)

        next_day = latest_row['Date'].values[0] + pd.Timedelta(days=1)
        print(f"\n=== Keras Prediction for Next Day ({next_day.strftime('%d/%m/%Y')}) ===")
        for i, jodi in enumerate(top10_jodis_keras, 1):
            print(f"Sample {i}: {jodi}")

# Main Menu
def main():
    while True:
        choice = input("\n1. Append to CSV\n2. Train/Test Models\n3. Predict Next Day\n4. Exit\nChoose: ")
        if choice == '1': append_to_csv()
        elif choice == '2': train_and_test_models()
        elif choice == '3': predict_next_day()
        elif choice == '4': break

# Run program
if __name__ == "__main__":
    main()