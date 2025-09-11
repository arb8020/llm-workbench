#!/usr/bin/env python3
"""
Monitor Basic Prompt Variation GSM8K Experiment

Connects to remote workers and tails their log files in real-time.

Usage:
    python monitor_experiment.py results/emotional_pilot_20250911_123456/
    python monitor_experiment.py results/emotional_pilot_20250911_123456/ --follow-only worker_1
"""

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import threading
import queue
import select

from bifrost.client import BifrostClient

def load_experiment_config(experiment_dir: Path) -> Dict[str, Any]:
    """Load experiment configuration."""
    config_path = experiment_dir / "experiment_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"No experiment config found at {config_path}")
    
    with open(config_path, 'r') as f:
        return json.load(f)

class LogStreamer:
    """Streams logs from a remote worker."""
    
    def __init__(self, worker_info: Dict[str, Any], experiment_name: str, output_queue: queue.Queue):
        self.worker_id = worker_info["worker_id"] 
        self.ssh_connection = worker_info["ssh_connection"]
        self.experiment_name = experiment_name
        self.output_queue = output_queue
        self.bifrost_client = BifrostClient(self.ssh_connection)
        self.running = False
        
    def start_streaming(self):
        """Start streaming logs in a separate thread."""
        self.running = True
        thread = threading.Thread(target=self._stream_logs, daemon=True)
        thread.start()
        return thread
    
    def _stream_logs(self):
        """Stream logs from remote worker."""
        log_file = f"~/experiment_logs/{self.experiment_name}_{self.worker_id}.log"
        
        try:
            # Check if log file exists
            check_cmd = f"test -f {log_file} && echo 'exists' || echo 'missing'"
            result = self.bifrost_client.exec(check_cmd)
            
            if "missing" in result:
                self.output_queue.put((self.worker_id, "⚠️", f"Log file not found: {log_file}"))
                self.output_queue.put((self.worker_id, "📋", "Worker may not have started yet..."))
                
                # Wait for log file to appear
                while self.running:
                    result = self.bifrost_client.exec(check_cmd)
                    if "exists" in result:
                        self.output_queue.put((self.worker_id, "✅", "Log file found, starting stream..."))
                        break
                    time.sleep(2)
                
                if not self.running:
                    return
            
            # Start tailing the log  
            self.output_queue.put((self.worker_id, "📡", f"Streaming logs from {self.worker_id}..."))
            
            # Use improved polling with tail -f approach
            # Start tail -f in background and poll its output
            tail_cmd = f"tail -f {log_file} 2>/dev/null"
            
            while self.running:
                try:
                    # Use tail -n 0 -f to get only new lines (more efficient than line counting)
                    # Get recent lines with timeout to avoid blocking
                    recent_cmd = f"timeout 3 tail -n 0 -f {log_file} 2>/dev/null | head -20"
                    result = self.bifrost_client.exec(recent_cmd)
                    
                    if result.strip():
                        lines = result.strip().split('\n')
                        for line in lines:
                            if not line.strip():
                                continue
                                
                            # Parse timestamp and level from line if present  
                            if " - INFO - " in line:
                                level = "📋"
                            elif " - ERROR - " in line:
                                level = "❌"
                            elif " - WARNING - " in line:
                                level = "⚠️"
                            elif "✅" in line:
                                level = "✅"
                            elif "🔄" in line:
                                level = "🔄"
                            else:
                                level = "📄"
                            
                            self.output_queue.put((self.worker_id, level, line.strip()))
                    
                    # Poll every 3 seconds (reduced frequency since we use timeout)
                    time.sleep(3)
                    
                except Exception as poll_error:
                    self.output_queue.put((self.worker_id, "⚠️", f"Polling error: {poll_error}"))
                    time.sleep(5)  # Wait longer on error
        
        except Exception as e:
            self.output_queue.put((self.worker_id, "❌", f"Streaming error: {e}"))
    
    def stop(self):
        """Stop streaming logs."""
        self.running = False

def format_log_line(worker_id: str, level: str, message: str) -> str:
    """Format a log line for display."""
    timestamp = time.strftime("%H:%M:%S")
    return f"[{timestamp}] [{worker_id:>8}] {level} {message}"

async def monitor_experiment(experiment_dir: Path, follow_only: Optional[str] = None):
    """Monitor experiment progress by streaming worker logs."""
    
    print(f"📂 Monitoring experiment: {experiment_dir.name}")
    
    # Load experiment config
    try:
        config = load_experiment_config(experiment_dir)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return
    
    experiment_name = config["experiment_name"]
    workers_info = config["workers_info"]
    
    if follow_only:
        workers_info = [w for w in workers_info if w["worker_id"] == follow_only]
        if not workers_info:
            print(f"❌ Worker '{follow_only}' not found")
            return
    
    print(f"🎯 Experiment: {experiment_name}")
    print(f"📊 Monitoring {len(workers_info)} worker(s): {[w['worker_id'] for w in workers_info]}")
    print(f"🔧 Variants: {config['variants']}")
    print(f"📋 Samples: {config['samples']}")
    print("=" * 80)
    
    # Create output queue for all streamers
    output_queue = queue.Queue()
    
    # Start log streamers
    streamers = []
    for worker_info in workers_info:
        streamer = LogStreamer(worker_info, experiment_name, output_queue)
        streamers.append(streamer)
        streamer.start_streaming()
    
    print("📡 Starting log streams...")
    print("   Press Ctrl+C to stop monitoring")
    print("=" * 80)
    
    try:
        # Main monitoring loop
        while True:
            try:
                # Get log line with timeout
                worker_id, level, message = output_queue.get(timeout=1.0)
                formatted_line = format_log_line(worker_id, level, message)
                print(formatted_line)
                
            except queue.Empty:
                # No logs in the last second, continue
                continue
                
    except KeyboardInterrupt:
        print("\n📋 Stopping monitoring...")
        
        # Stop all streamers
        for streamer in streamers:
            streamer.stop()
        
        print("✅ Monitoring stopped")
    
    except Exception as e:
        print(f"❌ Monitoring error: {e}")
        
        # Stop all streamers
        for streamer in streamers:
            streamer.stop()

def show_experiment_status(experiment_dir: Path):
    """Show current experiment status without streaming."""
    
    try:
        config = load_experiment_config(experiment_dir)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return
    
    experiment_name = config["experiment_name"] 
    workers_info = config["workers_info"]
    
    print(f"📊 Experiment Status: {experiment_name}")
    print("=" * 50)
    
    for worker_info in workers_info:
        worker_id = worker_info["worker_id"]
        ssh_connection = worker_info["ssh_connection"]
        
        try:
            client = BifrostClient(ssh_connection)
            
            # Check if tmux session is running
            tmux_session = f"{experiment_name}_{worker_id}"
            tmux_check = client.exec(f"tmux list-sessions | grep {tmux_session} || echo 'not_running'")
            
            if "not_running" in tmux_check:
                status = "❌ Not running"
            else:
                status = "✅ Running"
            
            # Check log file size
            log_file = f"~/experiment_logs/{experiment_name}_{worker_id}.log"
            log_check = client.exec(f"wc -l {log_file} 2>/dev/null || echo '0'")
            log_lines = log_check.split()[0] if log_check.split() else "0"
            
            print(f"   {worker_id}: {status} | Log lines: {log_lines}")
            
        except Exception as e:
            print(f"   {worker_id}: ❌ Connection error: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Monitor Basic Prompt Variation GSM8K Experiment")
    parser.add_argument("experiment_dir", type=Path, help="Path to experiment results directory")
    parser.add_argument("--follow-only", type=str, help="Only follow logs from specific worker")
    parser.add_argument("--status-only", action="store_true", help="Show status without streaming logs")
    
    args = parser.parse_args()
    
    if not args.experiment_dir.exists():
        print(f"❌ Experiment directory not found: {args.experiment_dir}")
        sys.exit(1)
    
    if args.status_only:
        show_experiment_status(args.experiment_dir)
    else:
        asyncio.run(monitor_experiment(args.experiment_dir, args.follow_only))