import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


db_path = '/Users/ananth/Downloads/FPA_FOD_20170508.sqlite'
conn = sqlite3.connect(db_path)

query = "SELECT LATITUDE, LONGITUDE, DISCOVERY_DOY, DISCOVERY_TIME, FIRE_SIZE FROM Fires"
data = pd.read_sql_query(query, conn)

print(data.head())

conn.close()

data = data.dropna()


data['FIRE_SIZE_CLASS'] = (data['FIRE_SIZE'] > 0).astype(int)

X = data[['LATITUDE', 'LONGITUDE', 'DISCOVERY_DOY', 'DISCOVERY_TIME']]
y = data['FIRE_SIZE_CLASS']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)

history = model.fit(X_train, y_train, epochs=50, validation_split=0.2, batch_size=32, callbacks=[early_stopping])

model.save('wildfire_detection_model.h5')

y_pred = (model.predict(X_test) > 0.5).astype(int)

print('Classification Report:')
print(classification_report(y_test, y_pred))
print('Accuracy:', accuracy_score(y_test, y_pred))

# Data Visualisation
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.tight_layout()
plt.show()
