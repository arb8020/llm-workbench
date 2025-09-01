#!/usr/bin/env python3
"""
Test script to demonstrate the new log level control functionality in bifrost deploy.

This script shows various logging scenarios that users might want:
1. Suppress bifrost logs while showing errors
2. Show detailed bifrost logs but suppress third-party libraries
3. Show all logs for debugging
"""

import logging
from shared.logging_config import setup_logging, parse_logger_levels

def demo_logging_scenario(title, level=None, logger_levels=None):
    """Demonstrate a logging scenario."""
    print(f"\n{'='*50}")
    print(f"SCENARIO: {title}")
    print(f"{'='*50}")
    
    # Setup logging for this scenario
    setup_logging(level=level, logger_levels=logger_levels or {})
    
    # Create various loggers
    bifrost_logger = logging.getLogger('bifrost')
    bifrost_deploy_logger = logging.getLogger('bifrost.deploy')
    bifrost_client_logger = logging.getLogger('bifrost.client')
    paramiko_logger = logging.getLogger('paramiko')
    broker_logger = logging.getLogger('broker')
    
    # Test messages
    bifrost_logger.info("Starting bifrost operation")
    bifrost_deploy_logger.debug("Deploying code to remote instance")
    bifrost_client_logger.info("Connecting to SSH server")
    paramiko_logger.info("SSH connection established")
    paramiko_logger.debug("SSH key authentication successful")
    broker_logger.warning("GPU instance utilization is high")
    bifrost_logger.error("Failed to deploy - connection timeout")

if __name__ == "__main__":
    print("Bifrost Log Level Control Demo")
    print("This demonstrates how users can control logging for different parts of the system")
    
    # Scenario 1: Default logging (everything at INFO level)
    demo_logging_scenario(
        "Default logging - everything at INFO level",
        level="INFO"
    )
    
    # Scenario 2: Quiet bifrost logs (suppress bifrost, show errors only)
    demo_logging_scenario(
        "Quiet mode - suppress bifrost logs (--quiet flag equivalent)",
        level="INFO",
        logger_levels={
            "bifrost": "ERROR",
            "bifrost.deploy": "ERROR", 
            "bifrost.client": "ERROR"
        }
    )
    
    # Scenario 3: Debug bifrost, suppress third-party
    demo_logging_scenario(
        "Debug bifrost, suppress third-party libraries",
        level="INFO",
        logger_levels={
            "bifrost": "DEBUG",
            "paramiko": "WARNING"
        }
    )
    
    # Scenario 4: Suppress everything except errors
    demo_logging_scenario(
        "Only show errors from all components",
        level="INFO",
        logger_levels={
            "bifrost": "ERROR",
            "paramiko": "ERROR", 
            "broker": "ERROR"
        }
    )
    
    print(f"\n{'='*50}")
    print("CLI Usage Examples:")
    print("1. Default: bifrost deploy user@host 'python script.py'")
    print("2. Quiet:   bifrost --quiet deploy user@host 'python script.py'")
    print("3. Debug:   bifrost --log-level DEBUG deploy user@host 'python script.py'")
    print("4. Custom:  bifrost --logger-level bifrost:DEBUG --logger-level paramiko:ERROR deploy user@host 'python script.py'")
    print(f"{'='*50}")