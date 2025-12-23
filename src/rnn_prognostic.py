import tensorflow as tf
from tensorflow.keras import layers, Model
import pandas as pd
import numpy as np
import os

# Ensure this class is NOT indented inside any function
class PINN_LSTM(Model):
    def __init__(self, window_size=60, features=2):
        super(PINN_LSTM, self).__init__()
        self.lstm = layers.LSTM(64, input_shape=(window_size, features), return_sequences=False)
        self.dense = layers.Dense(32, activation='relu')
        self.soh_head = layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        x = self.lstm(inputs)
        x = self.dense(x)
        return self.soh_head(x)

# Custom loss must also be at top level
def physics_informed_loss(y_true, y_pred):
    data_loss = tf.reduce_mean(tf.square(y_true - y_pred))
    diff = y_pred[1:] - y_pred[:-1]
    physics_residual = tf.reduce_mean(tf.maximum(0.0, diff))
    return data_loss + 0.1 * physics_residual

def train_model():
    # 1. Load your generated data
    df = pd.read_csv('data/battery_history.csv')
    
    # 2. Data Preprocessing (Sliding Window of 60 minutes)
    X, y = [], []
    window = 60
    for i in range(len(df) - window):
        X.append(df[['current', 'temp']].iloc[i:i+window].values)
        y.append(df['soh'].iloc[i+window])
    
    X, y = np.array(X), np.array(y)

    # 3. Define Architecture (LSTM + Dense)
    inputs = layers.Input(shape=(window, 2))
    x = layers.LSTM(64, return_sequences=False)(inputs)
    x = layers.Dense(32, activation='relu')(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss=physics_informed_loss)

    # 4. Train and Save
    print(" Training Physics-Informed RNN...")
    model.fit(X, y, epochs=10, batch_size=32, verbose=1)
    
    os.makedirs('models', exist_ok=True)
    model.save('models/fatigue_model.h5')
    print(" Model saved to models/fatigue_model.h5")

if __name__ == "__main__":
    train_model()