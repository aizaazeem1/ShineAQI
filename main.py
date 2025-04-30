from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

model = joblib.load("rf_model.pkl")

class AQIInput(BaseModel):
    co: float
    no: float
    no2: float
    o3: float
    so2: float
    pm2_5: float
    pm10: float
    nh3: float
    temp: float
    dwpt: float
    rhum: float
    prcp: float
    snow: float
    wdir: float
    wspd: float
    wpgt: float
    pres: float
    tsun: float
    coco: float
    hour: int
    day: int
    weekday: int
    month: int
    aqi_pm25: float
    aqi_pm10: float
    aqi_change_1h: float
    aqi_rolling_mean_6h: float

app = FastAPI()

@app.post("/predict")
def predict_aqi(data: AQIInput):
    df = pd.DataFrame([data.dict()])
    prediction = model.predict(df)[0]
    return {"predicted_aqi": float(prediction)}
