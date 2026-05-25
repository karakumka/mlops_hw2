import json
import logging
import os
import sys

import pandas as pd
from confluent_kafka import Consumer
from confluent_kafka import Producer
from prometheus_client import start_http_server, Summary, Counter, Histogram, Gauge

sys.path.append(os.path.abspath('./src'))  # noqa: PTH100
from preprocessing import preprocess_transform
from scorer import onehot_encoder, catboost_encoder, feature_columns, make_prediction


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/service.log'),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Set kafka configuration file
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
TRANSACTIONS_TOPIC = os.getenv("KAFKA_TRANSACTIONS_TOPIC", "transactions")
SCORING_TOPIC = os.getenv("KAFKA_SCORING_TOPIC", "scoring")

# Определяем метрики
PROCESSING_TIME = Summary('transaction_processing_seconds', 'Время обработки транзакции')
TRANSACTION_COUNT = Counter('transactions_total', 'Общее количество обработанных транзакций')

# Создаем более детальную гистограмму для распределения скоров
# Используем линейные бакеты с шагом 0.02 от 0 до 1 (50 бакетов)
FRAUD_SCORE = Histogram('fraud_score', 'Распределение скоров мошенничества',
                       buckets=[i/50.0 for i in range(51)])  # [0.0, 0.02, 0.04, ..., 0.98, 1.0]

FRAUD_RATIO = Gauge('fraud_ratio', 'Соотношение мошеннических транзакций к общему числу')


class ProcessingService:
    def __init__(self):
        self.consumer_config = {
            'bootstrap.servers': KAFKA_BOOTSTRAP_SERVERS,
            'group.id': 'ml-scorer',
            'auto.offset.reset': 'earliest',
        }
        self.producer_config = {
            'bootstrap.servers': KAFKA_BOOTSTRAP_SERVERS,
        }
        self.consumer = Consumer(self.consumer_config)
        self.consumer.subscribe([TRANSACTIONS_TOPIC])
        self.producer = Producer(self.producer_config)
        
        # Счетчики для метрик
        self.total_transactions = 0
        self.fraud_transactions = 0
        
        # Запуск HTTP-сервера для Prometheus
        start_http_server(8000)
        logger.info("Prometheus метрики доступны на порту 8000")

    @PROCESSING_TIME.time()
    def process_message(self, msg):
        try:
            # Десериализация JSON
            data = json.loads(msg.value().decode('utf-8'))

            # Извлекаем ID и данные
            transaction_id = data['transaction_id']
            transaction_dict = data['data']
            input_df = pd.DataFrame([transaction_dict])
            
            state = str(input_df["us_state"].iloc[0])
            merch = str(input_df["merch"].iloc[0])
            cat_id = str(input_df["cat_id"].iloc[0])

            # Препроцессинг и предсказание
            processed_df = preprocess_transform(input_df, onehot_encoder, catboost_encoder, feature_columns)
            submission, y_proba = make_prediction(processed_df, "kafka_stream")
            score = float(y_proba[0])

            # Обновляем метрики
            TRANSACTION_COUNT.inc()
            FRAUD_SCORE.observe(y_proba[0])
            
            self.total_transactions += 1
            is_fraud = int(submission['fraud_flag'].iloc[0])
            if is_fraud == 1:
                self.fraud_transactions += 1

            # Обновляем соотношение мошеннических транзакций
            if self.total_transactions > 0:
                FRAUD_RATIO.set(self.fraud_transactions / self.total_transactions)

            output_data = {
                'transaction_id': transaction_id,
                'score': score,
                'fraud_flag': is_fraud,
                "cat_id": cat_id,
                "us_state": state,
                "merch": merch,
            }

            # Отправка результата в топик scoring
            self.producer.produce(
                'scoring',
                value=json.dumps(output_data),
            )
            self.producer.flush()
            return True
        except Exception as e:
            logger.exception(f"Error processing message: {e}")
            return False

    def process_messages(self):
        while True:
            msg = self.consumer.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                logger.error(f"Kafka error: {msg.error()}")
                continue
            
            self.process_message(msg)


if __name__ == "__main__":
    logger.info('Starting Kafka ML scoring service...')
    service = ProcessingService()
    try:
        service.process_messages()
    except KeyboardInterrupt:
        logger.info('Service stopped by user')