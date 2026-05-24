import pandas as pd
import logging
import joblib


# Настройка логгера
logger = logging.getLogger(__name__)

logger.info('Importing pretrained entities...')

# Import pretrained entities
model_xgb = joblib.load("trained_entities/model.pkl")
onehot_encoder = joblib.load("trained_entities/onehot_encoder.pkl")
catboost_encoder = joblib.load("trained_entities/catboost_encoder.pkl")
feature_columns = joblib.load("trained_entities/feature_columns.pkl")

# Define optimal threshold
model_th = 0.97
logger.info('Pretrained entities imported successfully...')

# Make prediction
def make_prediction(dt, source_info="kafka"):

    y_proba = model_xgb.predict_proba(dt)[:, 1]

    # Make submission dataframe
    submission = pd.DataFrame({
        'score': y_proba,
        'fraud_flag': (y_proba > model_th) * 1,
    })
    logger.info(f'Prediction complete for file: {source_info}')

    # Return proba for positive class
    return submission, y_proba
