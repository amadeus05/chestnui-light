import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ANSI_YELLOW = "\033[93m"
ANSI_RESET = "\033[0m"

BASE_OUTPUT_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]
BARRIER_OUTPUT_COLUMNS = ["barrier_stop_pct", "barrier_take_pct"]
