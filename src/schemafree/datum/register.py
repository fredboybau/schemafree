import logging
from typing import Optional

import gin

_CONFIGURED = False


def get_logger(name: str = "schemafree") -> logging.Logger:
    global _CONFIGURED
    logger = logging.getLogger(name)
    if not _CONFIGURED:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
        root = logging.getLogger("schemafree")
        root.addHandler(handler)
        root.setLevel(logging.INFO)
        root.propagate = False
        _CONFIGURED = True
    return logger


def bind_plan(plan_path: str, overrides: Optional[list[str]] = None) -> None:
    gin.parse_config_file(plan_path)
    if overrides:
        gin.parse_config(overrides)
