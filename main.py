import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.popup import Popup
import joblib

# Constants for file paths
FOLDER_PATH = "/storage/emulated/0/KalyanApp"
CSV_PATH = os.path.join(FOLDER_PATH, "data.csv")
SCALER_PATH = os.path.join(FOLDER_PATH, "scaler.pkl")
MODEL_PATH = os.path.join(FOLDER_PATH, "model.h5")
ENCODER_PATH = os.path.join(FOLDER_PATH, "label_encoder.pkl")


class KalyanApp(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'

        # UI Elements
        self.date_input = TextInput(hint_text='Enter date (dd/mm/yyyy)', multiline=False)
        self.market_input = TextInput(hint_text='Enter market (e.g., Kalyan)', multiline=False)
        self.open_input = TextInput(hint_text='Enter Open value', multiline=False, input_filter="int")
        self.jodi_input = TextInput(hint_text='Enter Jodi value', multiline=False, input_filter="int")
        self.close_input = TextInput(hint_text='Enter Close value', multiline=False, input_filter="int")

        self.add_widget(self.date_input)
        self.add_widget(self.market_input)
        self.add_widget(self.open_input)
        self.add_widget(self.jodi_input)
        self.add_widget(self.close_input)

        self.append_btn = Button(text="Append to CSV", on_press=self.append_to_csv)
        self.train_btn = Button(text="Train & Test Model", on_press=self.train_model)
        self.predict_btn = Button(text="Predict Next Day", on_press=self.predict_jodi)

        self.add_widget(self.append_btn)
        self.add_widget(self.train_btn)
        self.add_widget(self.predict_btn)

    def popup(self, msg):
        popup = Popup(title='Notification', content=Label(text=msg),
                      size_hint=(None, None), size=(400, 200))
        popup.open()

    def append_to_csv(self, instance):
        try:
            date_input = self.date_input.text
            market_input = self.market_input.text
            open_val = int(self.open_input.text)
            jodi_val = int(self.jodi_input.text)
            close_val = int(self.close_input.text)

            if os.path.exists(CSV_PATH):
                df = pd.read_csv(CSV_PATH)
            else:
                df = pd.DataFrame()

            prev_jodi_distance = abs(jodi_val - df['Jodi'].iloc[-1]) if not df.empty else 0

            day_of_week = datetime.strptime(date_input, "%d/%m/%Y").strftime('%A')
            is_weekend = 1 if day_of_week in ['Saturday', 'Sunday'] else 0
            open_sum = sum(int(d) for d in str(open_val))
            close_sum = sum(int(d) for d in str(close_val))
            mirror_open = int(str(open_val)[::-1])
            mirror_close = int(str(close_val)[::-1])
            reverse_jodi = int(str(jodi_val)[::-1])

            new_row = {
                'Date': date_input, 'Market': market_input, 'Open': open_val,
                'Jodi': jodi_val, 'Close': close_val, 'day_of_week': day_of_week,
                'is_weekend': is_weekend, 'open_sum': open_sum,
                'close_sum': close_sum, 'mirror_open': mirror_open,
                'mirror_close': mirror_close, 'reverse_jodi': reverse_jodi,
                'is_holiday': 0, 'prev_jodi_distance': prev_jodi_distance
            }

            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            df.to_csv(CSV_PATH, index=False)
            self.popup("Row appended successfully!")

        except Exception as e:
            self.popup(f"Error: {e}")

    def train_model(self, instance):
        try:
            df = pd.read_csv(CSV_PATH)
            df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
            df.dropna(subset=['Date'], inplace=True)
            df = df[df['Market'].str.lower() == 'kalyan']
            df.dropna(inplace=True)

            cutoff_date = pd.to_datetime("31/03/2025", format='%d/%m/%Y')
            features = ['open_sum', 'close_sum', 'mirror_open', 'mirror_close', 'reverse_jodi',
                        'prev_jodi_distance', 'is_weekend', 'is_holiday']

            X_train = df[df['Date'] <= cutoff_date][features]
            y_train = df[df['Date'] <= cutoff_date]['Jodi']

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            joblib.dump(scaler, SCALER_PATH)

            le = LabelEncoder()
            y_train_encoded = le.fit_transform(y_train)
            joblib.dump(le, ENCODER_PATH)

            model = Sequential([
                Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
                Dense(64, activation='relu'),
                Dense(len(np.unique(y_train_encoded)), activation='softmax')
            ])
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            model.fit(X_train_scaled, y_train_encoded, epochs=100, verbose=0)
            model.save(MODEL_PATH)

            self.popup("Model trained and saved!")

        except Exception as e:
            self.popup(f"Training failed: {e}")

    def predict_jodi(self, instance):
        try:
            df = pd.read_csv(CSV_PATH)
            df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
            latest = df[df['Date'] == df['Date'].max()]
            features = ['open_sum', 'close_sum', 'mirror_open', 'mirror_close',
                        'reverse_jodi', 'prev_jodi_distance', 'is_weekend', 'is_holiday']

            scaler = joblib.load(SCALER_PATH)
            model = load_model(MODEL_PATH)
            le = joblib.load(ENCODER_PATH)

            X_latest = scaler.transform(latest[features])
            preds = model.predict(X_latest)[0]
            top10 = le.inverse_transform(np.argsort(-preds)[:10])

            message = "\n".join([f"Top {i+1}: {val}" for i, val in enumerate(top10)])
            self.popup("Predictions:\n" + message)

        except Exception as e:
            self.popup(f"Prediction failed: {e}")


class KalyanMLApp(App):
    def build(self):
        return KalyanApp()


if __name__ == '__main__':
    KalyanMLApp().run()
