#!/usr/bin/env python3
"""
Collect Results from Distributed GSM8K Experiment

Collects experiment results from remote workers back to the local machine using BifrostClient.

Usage:
    python collect_results.py results/experiment_name_timestamp/
    python collect_results.py results/experiment_name_timestamp/ --worker-subset worker_1,worker_2
    python collect_results.py results/experiment_name_timestamp/ --dry-run
    python collect_results.py results/experiment_name_timestamp/ --verify-only
"""

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

from shared.logging_config import setup_logging
from bifrost.client import BifrostClient

logger = logging.getLogger(__name__)


def check_local_disk_space(local_dir: Path, required_bytes: int, safety_factor: float = 1.5) -> tuple[bool, str]:
    """Check if local directory has enough disk space.
    
    Args:
        local_dir: Local directory to check
        required_bytes: Estimated bytes needed
        safety_factor: Multiply required by this factor for safety buffer
        
    Returns:
        (has_space, message)
    """
    try:
        # Get available disk space
        statvfs = shutil.disk_usage(local_dir.parent if local_dir.parent.exists() else local_dir)
        available_bytes = statvfs.free
        
        # Calculate required space with safety buffer
        required_with_buffer = int(required_bytes * safety_factor)
        
        if available_bytes >= required_with_buffer:
            return True, f"✅ Sufficient disk space: {available_bytes:,} bytes available, {required_with_buffer:,} needed"
        else:
            shortage_mb = (required_with_buffer - available_bytes) / (1024 * 1024)
            return False, f"❌ Insufficient disk space: Need {shortage_mb:.1f} MB more free space"
            
    except Exception as e:
        return False, f"❌ Could not check disk space: {e}"


def load_experiment_config(experiment_dir: Path) -> Dict[str, Any]:
    """Load experiment configuration from local directory."""
    config_path = experiment_dir / "experiment_config.json"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Experiment config not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    logger.info(f"Loaded experiment: {config['experiment_name']}")
    logger.info(f"Workers: {len(config['workers_info'])}")
    logger.info(f"Variants: {config['variants']}")
    logger.info(f"Samples: {config['samples']}")
    
    return config


def collect_worker_results(worker_info: Dict[str, Any], local_experiment_dir: Path, 
                          config: Dict[str, Any], dry_run: bool = False) -> Dict[str, Any]:
    """Collect results from a single worker."""
    worker_id = worker_info["worker_id"]
    ssh_connection = worker_info["ssh_connection"]
    
    logger.info(f"🔄 [{worker_id}] Connecting to {ssh_connection}")
    
    result_stats = {
        "worker_id": worker_id,
        "success": False,
        "files_transferred": 0,
        "bytes_transferred": 0,
        "error": None
    }
    
    try:
        # Connect to remote worker
        bifrost_client = BifrostClient(ssh_connection)
        
        # The output_dir path from config represents where results should be locally,
        # but on remote workers, results are saved to the same relative path structure
        # Let's check what exists on the remote machine
        
        # First, check if there are any results in the workspace
        logger.info(f"🔍 [{worker_id}] Checking for results on remote machine...")
        
        # Results should be in the bifrost workspace, mirroring local structure
        local_output_dir = config.get("output_dir", str(local_experiment_dir.absolute()))
        logger.info(f"🔍 [{worker_id}] Local output dir from config: {local_output_dir}")
        
        if local_output_dir.startswith('/Users/'):
            # Convert local ~/llm-workbench/... path to ~/.bifrost/workspace/... path
            parts = Path(local_output_dir).parts
            llm_workbench_index = next(i for i, part in enumerate(parts) if part == 'llm-workbench')
            relative_path = Path(*parts[llm_workbench_index+1:])  # everything after llm-workbench
            remote_results_pattern = f"/root/.bifrost/workspace/{relative_path}"
        else:
            # Fallback to experiment name
            remote_results_pattern = f"/root/.bifrost/workspace/examples/mats_neel/basic_prompt_variation_gsm8k/results/{local_experiment_dir.name}"
        
        logger.info(f"🎯 [{worker_id}] Searching remote path: {remote_results_pattern}")
        
        # Safety check: make sure we're not searching too broadly
        if remote_results_pattern.endswith("/.bifrost/workspace") or remote_results_pattern == "/root/.bifrost/workspace":
            result_stats["error"] = f"Path calculation error - would search entire workspace: {remote_results_pattern}"
            logger.error(f"❌ [{worker_id}] {result_stats['error']}")
            return result_stats
        
        # Check what exists remotely
        check_cmd = f"ls -la {remote_results_pattern} 2>/dev/null || echo 'NO_RESULTS'"
        check_result = bifrost_client.exec(check_cmd)
        
        if "NO_RESULTS" in check_result or not check_result.strip():
            logger.warning(f"⚠️ [{worker_id}] No results found at {remote_results_pattern}")
            # Try alternate locations
            alt_patterns = [
                "/root/.bifrost/workspace/examples/mats_neel/basic_prompt_variation_gsm8k/results",
                f"/root/.bifrost/workspace/results/{local_experiment_dir.name}",
                "/root/.bifrost/workspace/results",
                "/root/.bifrost/workspace"
            ]
            
            found_results = False
            for pattern in alt_patterns:
                check_cmd = f"find {pattern} -name '*.json' -o -name '*.jsonl' 2>/dev/null | head -5"
                alt_result = bifrost_client.exec(check_cmd)
                if alt_result.strip():
                    logger.info(f"🔍 [{worker_id}] Found files at {pattern}:")
                    for line in alt_result.strip().split('\n'):
                        logger.info(f"     {line}")
                    found_results = True
                    break
            
            if not found_results:
                result_stats["error"] = f"No results found on remote machine"
                return result_stats
            
            # Use the found location
            remote_results_pattern = pattern
        
        logger.info(f"📥 [{worker_id}] Found results, preparing download from {remote_results_pattern}")
        
        if dry_run:
            logger.info(f"🔍 [{worker_id}] DRY RUN: Would download {remote_results_pattern} to {local_experiment_dir}")
            list_cmd = f"find {remote_results_pattern} -type f | wc -l"
            file_count = bifrost_client.exec(list_cmd).strip()
            logger.info(f"🔍 [{worker_id}] DRY RUN: Estimated {file_count} files to transfer")
            result_stats["success"] = True
            result_stats["files_transferred"] = int(file_count) if file_count.isdigit() else 0
            return result_stats
        
        # Download all results using per-file SFTP (more robust than single connection)
        logger.info(f"⬇️ [{worker_id}] Downloading results...")
        
        # Get list of files to download
        find_cmd = f"find {remote_results_pattern} -type f"
        file_list_output = bifrost_client.exec(find_cmd).strip()
        
        if not file_list_output:
            result_stats["error"] = "No files found to download"
            return result_stats
        
        file_paths = [f.strip() for f in file_list_output.split('\n') if f.strip()]
        total_files = len(file_paths)
        logger.info(f"   📁 Found {total_files} files to download")
        
        # Estimate total size needed
        logger.info(f"   📏 Estimating download size...")
        estimated_bytes = 0
        ssh_client = bifrost_client._get_ssh_client()
        sftp = ssh_client.open_sftp()
        try:
            # Sample a few files to estimate average size
            sample_files = file_paths[:min(5, len(file_paths))]
            sample_total = 0
            for sample_file in sample_files:
                try:
                    sample_total += sftp.stat(sample_file).st_size
                except:
                    pass  # Skip files we can't stat
            
            if sample_total > 0:
                avg_size = sample_total / len(sample_files)
                estimated_bytes = int(avg_size * total_files)
            else:
                # Fallback estimate: assume 10KB per file
                estimated_bytes = total_files * 10 * 1024
                
        finally:
            sftp.close()
        
        logger.info(f"   📊 Estimated download size: {estimated_bytes:,} bytes ({estimated_bytes / (1024*1024):.1f} MB)")
        
        # Check local disk space before starting downloads
        has_space, space_message = check_local_disk_space(local_experiment_dir, estimated_bytes)
        logger.info(f"   💾 {space_message}")
        
        if not has_space:
            result_stats["error"] = f"Insufficient local disk space: {space_message}"
            return result_stats
        
        # Download files one by one with individual SFTP connections
        downloaded_files = 0
        total_bytes = 0
        failed_files = []
        
        for i, remote_file_path in enumerate(file_paths, 1):
            try:
                # Calculate relative path for local structure
                if remote_file_path.startswith(remote_results_pattern):
                    relative_path = remote_file_path[len(remote_results_pattern):].lstrip('/')
                    local_file_path = local_experiment_dir / relative_path
                else:
                    # Fallback: use just the filename
                    local_file_path = local_experiment_dir / Path(remote_file_path).name
                
                # Ensure local directory exists
                local_file_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Download single file with individual SFTP connection (Morph-style)
                ssh_client = bifrost_client._get_ssh_client()
                sftp = ssh_client.open_sftp()
                try:
                    file_size = sftp.stat(remote_file_path).st_size
                    sftp.get(remote_file_path, str(local_file_path))
                    downloaded_files += 1
                    total_bytes += file_size
                    logger.info(f"   ✅ [{i}/{total_files}] Downloaded {relative_path} ({file_size:,} bytes)")
                finally:
                    sftp.close()
                    
            except Exception as e:
                failed_files.append((remote_file_path, str(e)))
                logger.warning(f"   ❌ [{i}/{total_files}] Failed {Path(remote_file_path).name}: {e}")
        
        # Summary
        result_stats["success"] = downloaded_files > 0
        result_stats["files_transferred"] = downloaded_files
        result_stats["bytes_transferred"] = total_bytes
        
        if failed_files:
            logger.warning(f"   ⚠️ {len(failed_files)} files failed to download")
        
        logger.info(f"   📊 Downloaded {downloaded_files}/{total_files} files ({total_bytes:,} bytes)")
        
        logger.info(f"✅ [{worker_id}] Download complete!")
        logger.info(f"   📁 Files: {result_stats['files_transferred']}")
        logger.info(f"   📊 Size: {result_stats['bytes_transferred']:,} bytes")
        
    except Exception as e:
        logger.error(f"❌ [{worker_id}] Failed to collect results: {e}")
        result_stats["error"] = str(e)
        
    finally:
        # Close connection
        if 'bifrost_client' in locals():
            bifrost_client.close()
    
    return result_stats


def verify_collected_results(experiment_dir: Path, config: Dict[str, Any]) -> Dict[str, Any]:
    """Verify that all expected results were collected."""
    logger.info("🔍 Verifying collected results...")
    
    verification = {
        "total_expected": len(config["variants"]) * config["samples"],
        "total_found": 0,
        "missing_results": [],
        "extra_results": [],
        "variants_found": set(),
        "samples_found": set()
    }
    
    # Check each variant directory
    for variant in config["variants"]:
        variant_dir = experiment_dir / variant
        if not variant_dir.exists():
            logger.warning(f"⚠️ Missing variant directory: {variant}")
            continue
            
        verification["variants_found"].add(variant)
        
        # Check sample directories within variant
        for i in range(config["samples"]):
            sample_id = f"gsm8k_{i+1:04d}"
            sample_dir = variant_dir / sample_id
            
            if sample_dir.exists():
                # Check for required files
                required_files = ["trajectory.jsonl", "agent_state.json", "sample.json"]
                missing_files = []
                
                for req_file in required_files:
                    if not (sample_dir / req_file).exists():
                        missing_files.append(req_file)
                
                if not missing_files:
                    verification["total_found"] += 1
                    verification["samples_found"].add(sample_id)
                else:
                    verification["missing_results"].append(f"{variant}/{sample_id}: missing {missing_files}")
            else:
                verification["missing_results"].append(f"{variant}/{sample_id}: directory not found")
    
    # Report verification results
    logger.info(f"📊 Verification complete:")
    logger.info(f"   ✅ Found: {verification['total_found']}/{verification['total_expected']} results")
    logger.info(f"   📁 Variants: {len(verification['variants_found'])}/{len(config['variants'])}")
    logger.info(f"   📋 Samples: {len(verification['samples_found'])}/{config['samples']}")
    
    if verification["missing_results"]:
        logger.warning(f"⚠️ Missing results ({len(verification['missing_results'])}):")
        for missing in verification["missing_results"][:5]:  # Show first 5
            logger.warning(f"     {missing}")
        if len(verification["missing_results"]) > 5:
            logger.warning(f"     ... and {len(verification['missing_results']) - 5} more")
    
    return verification


def check_collection_readiness(config: Dict[str, Any]) -> Dict[str, Any]:
    """Quick check if experiment is ready for collection."""
    logger.info("🔍 Checking collection readiness...")
    
    readiness = {
        "ready_for_collection": False,
        "workers_with_results": 0,
        "workers_completed": 0,
        "total_workers": len(config["workers_info"]),
        "worker_details": []
    }
    
    for worker_info in config["workers_info"]:
        worker_id = worker_info["worker_id"]
        ssh_connection = worker_info["ssh_connection"]
        
        worker_status = {
            "worker_id": worker_id,
            "has_results": False,
            "experiment_completed": False,
            "result_count": 0,
            "error": None
        }
        
        try:
            bifrost_client = BifrostClient(ssh_connection)
            
            # Check if experiment tmux session is still running
            experiment_name = config["experiment_name"]
            tmux_session = f"{experiment_name}_{worker_id}"
            tmux_check = bifrost_client.exec(f"tmux list-sessions | grep {tmux_session} || echo 'not_running'")
            
            if "not_running" in tmux_check:
                worker_status["experiment_completed"] = True
                readiness["workers_completed"] += 1
            
            # Check if results directory exists and has files
            local_output_dir = config["output_dir"]
            if local_output_dir.startswith('/Users/'):
                # Convert local path to bifrost workspace path
                parts = Path(local_output_dir).parts
                llm_workbench_index = next(i for i, part in enumerate(parts) if part == 'llm-workbench')
                relative_path = Path(*parts[llm_workbench_index+1:])
                remote_results_pattern = f"/root/.bifrost/workspace/{relative_path}"
            else:
                # Fallback
                remote_results_pattern = f"/root/.bifrost/workspace/examples/mats_neel/basic_prompt_variation_gsm8k/results/{local_output_dir}"
            check_cmd = f"find {remote_results_pattern} -name '*.json' -o -name '*.jsonl' 2>/dev/null | wc -l"
            result_count = bifrost_client.exec(check_cmd).strip()
            
            if result_count.isdigit() and int(result_count) > 0:
                worker_status["has_results"] = True
                worker_status["result_count"] = int(result_count)
                readiness["workers_with_results"] += 1
            
        except Exception as e:
            worker_status["error"] = str(e)
        finally:
            if 'bifrost_client' in locals():
                bifrost_client.close()
        
        readiness["worker_details"].append(worker_status)
    
    # Determine overall readiness
    readiness["ready_for_collection"] = (
        readiness["workers_with_results"] > 0 and 
        readiness["workers_completed"] == readiness["total_workers"]
    )
    
    return readiness


def main():
    parser = argparse.ArgumentParser(description="Collect results from distributed GSM8K experiment")
    parser.add_argument("experiment_dir", type=str, help="Path to experiment directory")
    parser.add_argument("--worker-subset", type=str, help="Comma-separated list of worker IDs to collect from")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be collected without downloading")
    parser.add_argument("--verify-only", action="store_true", help="Only verify existing local results")
    parser.add_argument("--status", action="store_true", help="Quick check if results are ready for collection")
    parser.add_argument("--explore", action="store_true", help="Explore remote filesystem to find where results are actually saved")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level=logging.DEBUG if args.verbose else logging.INFO)
    
    experiment_dir = Path(args.experiment_dir)
    
    if not experiment_dir.exists():
        logger.error(f"❌ Experiment directory not found: {experiment_dir}")
        sys.exit(1)
    
    logger.info(f"🚀 Collecting results for experiment: {experiment_dir.name}")
    
    try:
        # Load experiment configuration
        config = load_experiment_config(experiment_dir)
        
        if args.explore:
            # Explore remote filesystem to find where results actually are
            logger.info("🔍 Exploring remote filesystem...")
            
            for worker_info in config["workers_info"]:
                worker_id = worker_info["worker_id"]
                ssh_connection = worker_info["ssh_connection"]
                logger.info(f"🔄 [{worker_id}] Connecting to {ssh_connection}")
                
                try:
                    bifrost_client = BifrostClient(ssh_connection)
                    
                    # Check common locations
                    locations_to_check = [
                        "/root/",
                        "/root/.bifrost/",
                        "/root/.bifrost/workspace/",
                        "/root/.bifrost/workspace/examples/",
                        "/root/.bifrost/workspace/examples/mats_neel/",
                        "/root/.bifrost/workspace/examples/mats_neel/basic_prompt_variation_gsm8k/",
                        f"/root/.bifrost/workspace/examples/mats_neel/basic_prompt_variation_gsm8k/results/",
                        f"/root/experiment_results/",
                        f"~/.bifrost/workspace/results/",
                    ]
                    
                    for location in locations_to_check:
                        logger.info(f"🔍 [{worker_id}] Checking {location}")
                        try:
                            result = bifrost_client.exec(f"ls -la {location}")
                            if result.strip() and "No such file" not in result:
                                logger.info(f"✅ [{worker_id}] Found: {location}")
                                # Show first few lines
                                lines = result.strip().split('\n')[:10]
                                for line in lines:
                                    logger.info(f"     {line}")
                            else:
                                logger.info(f"❌ [{worker_id}] Not found: {location}")
                        except Exception as e:
                            logger.info(f"❌ [{worker_id}] Error checking {location}: {e}")
                    
                    # Also search for any JSON/JSONL files
                    logger.info(f"🔍 [{worker_id}] Searching for result files...")
                    try:
                        result = bifrost_client.exec(f"find /root -name '*.json' -o -name '*.jsonl' | head -20")
                        if result.strip():
                            logger.info(f"📁 [{worker_id}] Found result files:")
                            for line in result.strip().split('\n'):
                                logger.info(f"     {line}")
                    except Exception as e:
                        logger.info(f"❌ [{worker_id}] Error searching for files: {e}")
                        
                except Exception as e:
                    logger.error(f"❌ [{worker_id}] Failed to connect: {e}")
                finally:
                    if 'bifrost_client' in locals():
                        bifrost_client.close()
            
            return
        
        if args.status:
            # Quick readiness check
            readiness = check_collection_readiness(config)
            
            if readiness["ready_for_collection"]:
                logger.info("🎉 READY FOR COLLECTION!")
                logger.info(f"   ✅ All {readiness['total_workers']} workers completed")
                logger.info(f"   📁 {readiness['workers_with_results']} workers have results")
                
                total_files = sum(w["result_count"] for w in readiness["worker_details"] if w["has_results"])
                logger.info(f"   📊 Total files available: {total_files}")
                logger.info(f"💡 Run: python collect_results.py {experiment_dir}")
                sys.exit(0)
            else:
                logger.warning("⏳ NOT READY FOR COLLECTION")
                logger.info(f"   📊 Status: {readiness['workers_completed']}/{readiness['total_workers']} workers completed")
                logger.info(f"   📁 Results: {readiness['workers_with_results']}/{readiness['total_workers']} workers have results")
                
                for worker in readiness["worker_details"]:
                    status = "✅" if worker["experiment_completed"] else "🔄"
                    results = f"📁{worker['result_count']}" if worker["has_results"] else "❌"
                    error = f" | Error: {worker['error']}" if worker["error"] else ""
                    logger.info(f"   {worker['worker_id']}: {status} completed | {results} files{error}")
                
                if readiness["workers_completed"] < readiness["total_workers"]:
                    logger.info("💡 Wait for workers to complete, then try again")
                else:
                    logger.info("💡 Check worker logs for errors")
                sys.exit(1)
        
        if args.verify_only:
            # Only verify existing results
            verification = verify_collected_results(experiment_dir, config)
            if verification["total_found"] == verification["total_expected"]:
                logger.info("🎉 All results verified successfully!")
                sys.exit(0)
            else:
                logger.error("❌ Verification failed - missing results")
                sys.exit(1)
        
        # Filter workers if subset specified
        workers_to_collect = config["workers_info"]
        if args.worker_subset:
            subset_ids = [w.strip() for w in args.worker_subset.split(",")]
            workers_to_collect = [w for w in workers_to_collect if w["worker_id"] in subset_ids]
            logger.info(f"📋 Collecting from subset: {subset_ids}")
        
        if not workers_to_collect:
            logger.error("❌ No workers to collect from")
            sys.exit(1)
        
        # Collect results from each worker
        logger.info(f"📥 Starting collection from {len(workers_to_collect)} workers...")
        
        all_results = []
        successful_collections = 0
        total_files = 0
        total_bytes = 0
        
        for worker_info in workers_to_collect:
            result_stats = collect_worker_results(
                worker_info=worker_info,
                local_experiment_dir=experiment_dir,
                config=config,
                dry_run=args.dry_run
            )
            
            all_results.append(result_stats)
            
            if result_stats["success"]:
                successful_collections += 1
                total_files += result_stats["files_transferred"]
                total_bytes += result_stats["bytes_transferred"]
            else:
                logger.error(f"❌ {result_stats['worker_id']}: {result_stats['error']}")
        
        # Summary
        logger.info("🎉 Collection complete!")
        logger.info(f"   ✅ Successful: {successful_collections}/{len(workers_to_collect)} workers")
        logger.info(f"   📁 Total files: {total_files:,}")
        logger.info(f"   📊 Total size: {total_bytes:,} bytes")
        
        if not args.dry_run and successful_collections > 0:
            # Verify collected results
            verification = verify_collected_results(experiment_dir, config)
            
            if verification["total_found"] == verification["total_expected"]:
                logger.info("🎉 All expected results collected successfully!")
                logger.info(f"📊 Ready for analysis: python analyze_results.py {experiment_dir}")
            else:
                logger.warning("⚠️ Some results may be missing - check worker logs")
        
        # Exit with error if any collections failed
        if successful_collections < len(workers_to_collect):
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"❌ Collection failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()