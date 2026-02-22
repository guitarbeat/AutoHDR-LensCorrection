import os
import logging
from typing import List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

COMPETITION_NAME = "automatic-lens-correction"

def setup_kaggle_auth():
    """
    Ensures Kaggle authentication is set up.
    Expects KAGGLE_USERNAME and KAGGLE_KEY environment variables to be set,
    or a kaggle.json file in ~/.kaggle/kaggle.json.
    """
    if "KAGGLE_USERNAME" in os.environ and "KAGGLE_KEY" in os.environ:
        logger.info("Using Kaggle API credentials from environment variables.")
    else:
        kaggle_json_path = os.path.expanduser("~/.kaggle/kaggle.json")
        if os.path.exists(kaggle_json_path):
            logger.info("Using Kaggle API credentials from ~/.kaggle/kaggle.json.")
        else:
            logger.warning(
                "Kaggle API credentials not found! Please set KAGGLE_USERNAME "
                "and KAGGLE_KEY environment variables, or place kaggle.json "
                "in ~/.kaggle/kaggle.json"
            )

def download_competition_file(filename: str, download_path: str = "./data") -> bool:
    """
    Downloads a specific file from the Kaggle competition to avoid downloading
    the entire massive dataset.
    """
    setup_kaggle_auth()
    
    os.makedirs(download_path, exist_ok=True)
    logger.info(f"Downloading {filename} from competition {COMPETITION_NAME}...")
    
    try:
        # Note: We import kaggle here so it doesn't immediately fail on import 
        # if credentials aren't set up yet during app initialization.
        import kaggle
        kaggle.api.authenticate()
        
        kaggle.api.competition_download_file(
            competition=COMPETITION_NAME,
            file_name=filename,
            path=download_path
        )
        logger.info(f"Successfully downloaded {filename} to {download_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to download {filename}: {e}")
        return False

def list_competition_files() -> List[str]:
    """
    Lists all available files in the competition.
    """
    setup_kaggle_auth()
    try:
        import kaggle
        kaggle.api.authenticate()
        files = kaggle.api.competition_list_files(competition=COMPETITION_NAME)
        file_names = [f.name for f in files]
        logger.info(f"Found {len(file_names)} files in competition {COMPETITION_NAME}.")
        return file_names
    except Exception as e:
        logger.error(f"Failed to list files: {e}")
        return []

if __name__ == "__main__":
    # Example usage:
    # files = list_competition_files()
    # if files:
    #     print("Available files:", files)
    #     # Example: download_competition_file("sample_submission.csv")
    pass
