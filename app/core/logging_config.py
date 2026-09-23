import logging
import os
import sys


DEFAULT_LOG_FORMAT = (
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)


def configure_logging() -> None:
    """
    Configure application-wide logging.

    LOG_LEVEL can be configured through the environment.
    Supported values include DEBUG, INFO, WARNING, ERROR, CRITICAL.
    """

    log_level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_name, logging.INFO)

    logging.basicConfig(
        level=log_level,
        format=DEFAULT_LOG_FORMAT,
        stream=sys.stdout,
        force=True,
    )

    # Keep Uvicorn logs aligned with the application log level.
    logging.getLogger("uvicorn").setLevel(log_level)
    logging.getLogger("uvicorn.error").setLevel(log_level)
    logging.getLogger("uvicorn.access").setLevel(log_level)

    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)