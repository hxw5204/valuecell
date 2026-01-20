"""Constants for the Quant Screener pipeline."""

DEFAULT_TOP_K: int = 300
DEFAULT_TOP_N: int = 30
DEFAULT_UNIVERSE_SIZE: int = 4800

METADATA_MAX_CONCURRENCY: int = 2
METADATA_MIN_INTERVAL_S: float = 0.5
FINANCIAL_MAX_CONCURRENCY: int = 2
FINANCIAL_MIN_INTERVAL_S: float = 0.5

DATA_DIR_NAME: str = "data"
RUNS_DIR_NAME: str = "runs"
EVIDENCE_DIR_NAME: str = "evidence"
LOGIC_GRAPH_DIR_NAME: str = "logic_graph"
REPORTS_DIR_NAME: str = "reports"
