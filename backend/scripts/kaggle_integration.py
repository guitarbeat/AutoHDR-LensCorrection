import os
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
from dotenv import load_dotenv

# Load environment variables (KAGGLE_USERNAME, KAGGLE_KEY)
load_dotenv()

COMPETITION_NAME = "automatic-lens-correction"
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

def init_kaggle_api():
    """Initializes and authenticates the Kaggle API."""
    api = KaggleApi()
    api.authenticate()
    return api

def download_competition_file(file_name: str, force: bool = False):
    """
    Selectively downloads a single file from the competition.
    Prevents downloading the entire massive dataset locally.
    """
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    target_path = os.path.join(DATA_DIR, file_name)
    if os.path.exists(target_path) and not force:
        print(f"File {file_name} already exists. Skipping download.")
        return target_path

    api = init_kaggle_api()
    print(f"Downloading {file_name} from {COMPETITION_NAME}...")
    api.competition_download_file(COMPETITION_NAME, file_name, path=DATA_DIR)
    
    # Kaggle downloads are often zipped
    zip_path = target_path + ".zip"
    if os.path.exists(zip_path):
        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)
        os.remove(zip_path) # Clean up zip
        
    return target_path

def load_partial_csv(file_name: str, nrows: int = 1000, usecols: list = None) -> pd.DataFrame:
    """
    Loads only a subset of a CSV file into memory to adhere to the
    Cloud-First strategy and prevent local OOM errors.
    """
    file_path = download_competition_file(file_name)
    print(f"Loading {nrows} rows from {file_name}...")
    
    # Handle both direct CSV and zipped CSVs seamlessly in pandas
    try:
        df = pd.read_csv(file_path, nrows=nrows, usecols=usecols)
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

if __name__ == "__main__":
    # Example usage: Just download the sample submission or a small metadata file
    # dataset.csv is a placeholder name for the competition's training labels
    print("Testing Kaggle selective download...")
    # df = load_partial_csv("train.csv", nrows=10, usecols=["id", "k1", "k2"]) # Example
    print("Kaggle integration ready.")
