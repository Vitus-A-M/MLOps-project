MAX_DATE = "2024-01-31"
MIN_DATE = "2024-01-01"


ARTIFACTS_DIR = "./artifacts"
RAW_DATA_PATH = "./artifacts/raw_data.csv"
GOLD_DATA_PATH = "./artifacts/train_data_gold.csv"
SCALER_PATH = "./artifacts/scaler.pkl"


MODEL_NAME = "lead_model"
ARTIFACT_PATH = "model"
DATA_VERSION = "00000"


COLUMNS_TO_DROP_INITIAL = [
    "is_active", "marketing_consent", "first_booking", 
    "existing_customer", "last_seen"
]

COLUMNS_TO_DROP_EDA = [
    "domain", "country", "visited_learn_more_before_booking", "visited_faq"
]