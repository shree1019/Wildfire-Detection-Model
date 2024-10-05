import pandas as pd
from tensorflow.keras.models import load_model

model = load_model('wildfire_detection_model.h5')

# Example input data
new_data = {
    'LATITUDE': [38.0],
    'LONGITUDE': [-120.0],
    'DISCOVERY_DOY': [200],
    'DISCOVERY_TIME': [1430]
}

# Convert to DataFrame
X_new = pd.DataFrame(new_data)

# Predict using the loaded model
predictions = (model.predict(X_new) > 0.5).astype(int)

# Print predictions
if predictions[0] == 1:
    print("Prediction: Wildfire detected.")
else:
    print("Prediction: No wildfire detected.")

