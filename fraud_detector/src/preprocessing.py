import logging
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from category_encoders import CatBoostEncoder


logger = logging.getLogger(__name__)

# Перевод колонки с датой и временем в Unix
def transaction_time_unix(df):
    
    df["transaction_time"] = pd.to_datetime(df["transaction_time"], utc=True)
    df["transaction_time_unix"] = df["transaction_time"].astype("int64") // 10**9
    df = df.drop(columns=["transaction_time"])
    
    return df

# Склеивание имени и фамилии
def full_name(df):
    
    df['full_name'] = df['name_1'].fillna("").astype(str) + ' ' + df['name_2'].fillna("").astype(str)
    df = df.drop(columns=["name_1", "name_2"])

    return df

# Кодирование переменных "идентификатор товара" и "пол" через OneHotEncoder
def one_hot_encoding_transform(df, encoder: OneHotEncoder):

    one_hot_cols = ['cat_id', 'gender']

    transformed = encoder.transform(df[one_hot_cols])
    transformed_cols = encoder.get_feature_names_out(one_hot_cols)
    transformed_df = pd.DataFrame(transformed, columns=transformed_cols, index=df.index)
    df = df.drop(columns=one_hot_cols)
    df = pd.concat([df, transformed_df], axis=1)

    return df

# Кодирование текстовых переменных через CatBoostEncoder
def catboost_encoder_transform(df, encoder: CatBoostEncoder):

    cat_cols = ["merch", "full_name", "street", "one_city", "us_state", "jobs"]

    transformed = encoder.transform(df[cat_cols])
    df = df.drop(columns=cat_cols)
    df = pd.concat([df, transformed], axis=1)

    return df

def preprocess_transform(df, onehot_encoder, catboost_encoder, feature_columns):
    
    df = df.copy()

    df = full_name(df)
    df = transaction_time_unix(df)

    df = one_hot_encoding_transform(df, onehot_encoder)
    df = catboost_encoder_transform(df, catboost_encoder)

    df = df[feature_columns]

    return df