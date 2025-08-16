import argparse
import asyncio
import logging
import importlib
import sys
from typing import Type

from .controller import Controller
from .worker_base import WorkerBase

def setup_logging(level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def import_worker_class(worker_spec: str) -> Type[WorkerBase]:
    """Import worker class from module:class specification."""
    try:
        module_name, class_name = worker_spec.rsplit(':', 1)
        module = importlib.import_module(module_name)
        worker_class = getattr(module, class_name)
        
        if not issubclass(worker_class, WorkerBase):
            raise ValueError(f"{class_name} is not a subclass of WorkerBase")
        
        return worker_class
    except Exception as e:
        raise ValueError(f"Failed to import worker {worker_spec}: {e}")

def run_controller():
    """Run the leyline controller."""
    parser = argparse.ArgumentParser(description="Run Leyline Controller")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="HTTP port to bind to")
    parser.add_argument("--ws-port", type=int, help="WebSocket port (default: HTTP port + 1)")
    parser.add_argument("--log-level", default="INFO", help="Log level")
    
    args = parser.parse_args()
    setup_logging(args.log_level)
    
    controller = Controller(host=args.host, port=args.port, ws_port=args.ws_port)
    controller.run()

def run_worker():
    """Run a leyline worker."""
    parser = argparse.ArgumentParser(description="Run Leyline Worker")
    parser.add_argument("worker", help="Worker class in format module:class (e.g., examples.echo_worker:EchoWorker)")
    parser.add_argument("--controller", default="ws://localhost:8001", help="Controller WebSocket URL")
    parser.add_argument("--ping-interval", type=float, default=20.0, help="WebSocket ping interval")
    parser.add_argument("--ping-timeout", type=float, default=10.0, help="WebSocket ping timeout")
    parser.add_argument("--max-retries", type=int, default=-1, help="Max connection retries (-1 for infinite)")
    parser.add_argument("--log-level", default="INFO", help="Log level")
    
    args = parser.parse_args()
    setup_logging(args.log_level)
    
    try:
        worker_class = import_worker_class(args.worker)
        worker = worker_class(
            controller_url=args.controller,
            ping_interval=args.ping_interval,
            ping_timeout=args.ping_timeout
        )
        
        asyncio.run(worker.run(max_retries=args.max_retries))
        
    except KeyboardInterrupt:
        print("\nShutting down worker...")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Leyline - WebSocket tunneling framework")
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Controller subcommand
    controller_parser = subparsers.add_parser("controller", help="Run the controller")
    controller_parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    controller_parser.add_argument("--port", type=int, default=8000, help="HTTP port to bind to")
    controller_parser.add_argument("--ws-port", type=int, help="WebSocket port (default: HTTP port + 1)")
    controller_parser.add_argument("--log-level", default="INFO", help="Log level")
    
    # Worker subcommand
    worker_parser = subparsers.add_parser("worker", help="Run a worker")
    worker_parser.add_argument("worker", help="Worker class in format module:class")
    worker_parser.add_argument("--controller", default="ws://localhost:8001", help="Controller WebSocket URL")
    worker_parser.add_argument("--ping-interval", type=float, default=20.0, help="WebSocket ping interval")
    worker_parser.add_argument("--ping-timeout", type=float, default=10.0, help="WebSocket ping timeout")
    worker_parser.add_argument("--max-retries", type=int, default=-1, help="Max connection retries")
    worker_parser.add_argument("--log-level", default="INFO", help="Log level")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    setup_logging(args.log_level)
    
    try:
        if args.command == "controller":
            controller = Controller(host=args.host, port=args.port, ws_port=args.ws_port)
            controller.run()
        elif args.command == "worker":
            worker_class = import_worker_class(args.worker)
            worker = worker_class(
                controller_url=args.controller,
                ping_interval=args.ping_interval,
                ping_timeout=args.ping_timeout
            )
            asyncio.run(worker.run(max_retries=args.max_retries))
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()