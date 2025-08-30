import logging.config
import os


def setup_logging(level: str = None, use_json: bool = None):
    """Setup standardized logging configuration using dict config."""
    level = level or os.getenv("LOG_LEVEL", "INFO")
    use_json = use_json if use_json is not None else os.getenv("LOG_JSON", "").lower() == "true"
    
    formatters = {
        "standard": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        }
    }
    
    if use_json:
        formatters["json"] = {
            "()": "shared.json_formatter.JSONFormatter",
            "fmt_keys": {
                "level": "levelname",
                "logger": "name", 
                "module": "module",
                "function": "funcName",
                "line": "lineno"
            }
        }
    
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": formatters,
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": level,
                "formatter": "json" if use_json else "standard",
                "stream": "ext://sys.stdout"
            }
        },
        "root": {
            "level": level,
            "handlers": ["console"]
        }
    }
    
    logging.config.dictConfig(config)