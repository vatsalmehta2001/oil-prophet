"""
Configuration setup for Reddit API access.

This module helps set up the Reddit API credentials for the sentiment analysis module.
"""

import os
import json
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def create_reddit_config(
    client_id: str = "YNnpGUsSVTH-94FqVOK6ug",
    client_secret: str = "mXHF3zW4w3FdzO_Q66YEmB6De65dNQ",
    user_agent: str = "OilProphet/1.0 (by /u/your_reddit_username)",
    config_file: str = "reddit_config.json"
) -> None:
    """
    Create a Reddit API configuration file with provided credentials.
    
    Args:
        client_id: Reddit API client ID
        client_secret: Reddit API client secret
        user_agent: Reddit API user agent
        config_file: Path to save the config file
    """
    config = {
        "client_id": client_id,
        "client_secret": client_secret,
        "user_agent": user_agent
    }
    
    # Create the config file
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=4)
    
    logger.info(f"Created Reddit API configuration file: {config_file}")
    logger.info("You can update this file with your own credentials if needed.")

def check_reddit_config(config_file: str = "reddit_config.json") -> bool:
    """
    Check if Reddit API configuration file exists.
    
    Args:
        config_file: Path to the config file
        
    Returns:
        True if the file exists and has valid credentials, False otherwise
    """
    if not os.path.exists(config_file):
        logger.warning(f"Reddit API configuration file not found: {config_file}")
        logger.info("Creating a new configuration file with default credentials")
        create_reddit_config(config_file=config_file)
        return True
    
    # Check if the file has valid credentials
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        required_keys = ['client_id', 'client_secret', 'user_agent']
        if all(key in config for key in required_keys):
            logger.info(f"Reddit API configuration file found: {config_file}")
            return True
        else:
            logger.warning(f"Reddit API configuration file is missing required keys")
            return False
    
    except json.JSONDecodeError:
        logger.error(f"Reddit API configuration file is not valid JSON")
        return False

if __name__ == "__main__":
    # Create a Reddit API configuration file with provided credentials
    create_reddit_config()