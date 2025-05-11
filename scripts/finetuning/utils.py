# finetuning/utils.py

import logging
import os
from dotenv import load_dotenv

# --- Logging Configuration ---
def setup_logging():
    """Configures logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger("finetuning_script") # Main logger name

logger = setup_logging() # Initialize logger once

def load_hf_token() -> str:
    """Loads Hugging Face token from .env file."""
    load_dotenv()
    token = os.getenv("HF_TOKEN")
    if not token:
        logger.error("Hugging Face token (HF_TOKEN) not found in .env file.")
        raise ValueError("Hugging Face token not found in .env.")
    logger.info("Hugging Face token loaded successfully.")
    return token