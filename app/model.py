import time
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import joblib
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

X = np.array([
    [1000, 2, 1, 20],
    [1500, 3, 2, 15],
    [1800, 3, 2, 10],
    [2200, 4, 3, 5],
    [2500, 4, 3, 2]
])

y = np.array([
    2000000,
    3000000,
    3800000,
    5000000,
    6000000
])

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LinearRegression()),
])

pipeline.fit(X, y)

joblib.dump(pipeline, "models/house_price_pipeline.pkl")

