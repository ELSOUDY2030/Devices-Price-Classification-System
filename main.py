from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
import joblib
import numpy as np


# Define a model for incoming device specifications
class DeviceSpecs(BaseModel):
    battery_power: int
    blue: int
    clock_speed: float
    dual_sim: int
    fc: int
    four_g: int
    int_memory: int
    m_dep: float
    mobile_wt: int
    n_cores: int
    pc: int
    px_height: int
    px_width: int
    ram: int
    sc_h: int
    sc_w: int
    talk_time: int
    three_g: int
    touch_screen: int
    wifi: int

# Load the trained machine learning model
model = joblib.load('random_forest_model.pkl')

# Create a FastAPI app
app = FastAPI()

# Define a route for the predict endpoint
@app.post("/predict")
async def predict_price(device_specs: DeviceSpecs):

    # Convert device specifications to a list of values
    input_data = [device_specs.dict()[feature] for feature in device_specs.__fields__]

    # Reshape the input data to have 1 row and multiple columns (-1 indicates that the number of columns is inferred)
    input_data_reshaped = np.array(input_data).reshape(1, -1)

    # Predict the price using the loaded model
    predicted_price = model.predict(input_data_reshaped)[0]

    # Convert the predicted price to a Python native type
    predicted_price = int(predicted_price)

    # Return the predicted price as a JSON response
    return jsonable_encoder({"predicted_price": predicted_price})

