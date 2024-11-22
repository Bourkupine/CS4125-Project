from logging import getLogger

# Initialize the logger
logger = getLogger(__name__)

def log_classification(email, classification):

    #Log the classification of an email.

    logger.info(f"Email: '{email}' classified as {classification}")

def log_metrics_update(stats):

    #Log metrics updates.

    logger.info(f"Updated stats: {stats}")

def log_ui_update(email, classification):

    #Log UI updates for email classification.

    logger.info(f"Dashboard updated: '{email}' -> {classification}")
