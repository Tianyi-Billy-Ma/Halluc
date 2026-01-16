import logging
from logging import DEBUG, INFO

from llmhalluc.utils.sys_utils import is_rank_zero


class Rank0Filter(logging.Filter):
    """Logging filter that only allows messages from rank 0.

    In distributed training, this prevents duplicate log messages
    from being printed by all processes.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        return is_rank_zero()


def setup_logging(verbose: bool, rank_zero_only: bool = True) -> None:
    """Setup logging with optional rank-0 filtering.

    Args:
        verbose: If True, use DEBUG level; otherwise INFO.
        rank_zero_only: If True, only log on rank 0 in distributed training.
    """
    level = DEBUG if verbose else INFO
    logging_format = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=level, format=logging_format)

    if rank_zero_only:
        # Add filter to root logger to suppress non-rank-0 logs
        root_logger = logging.getLogger()
        root_logger.addFilter(Rank0Filter())
