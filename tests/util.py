import logging
from pathlib import Path


def setup_cli_logging():
    tests_log_file = Path(__file__).resolve().parent / "log/tests.log"

    logging.basicConfig(
        format='%(message)s',
        level=logging.INFO, handlers=[
            logging.FileHandler(
                tests_log_file
            ),
            logging.StreamHandler()
        ]
    )
