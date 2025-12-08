#!/usr/bin/env python3
"""
InferaGrid Resource Locker
Allows users to reserve a portion of their compute resources for InferaGrid marketplace
with a 20% safety buffer for local machine usage.
"""

import json
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# Import our modules
try:
    from system_profiler import (
        get_system_info, get_cpu_info, get_memory_info,
        get_gpu_info_nvidia, get_gpu_info_macos, get_gpu_info_apple_silicon,
        get_disk_info, get_network_info, Colors, get_percentage_bar
    )
except ImportError:
    print("Error: system_profiler.py must be in the same directory")
    sys.exit(1)

try:
    from categories import (
        classify_node, CategoryDefinition, InferenceSpeed, ComputeTier,
        GPUCapability, WorkloadType, print_all_categories
    )
except ImportError:
    print("Error: categories.py must be in the same directory")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

SAFETY_BUFFER_PERCENT = 20  # Always keep 20% for local machine
CONFIG_DIR = Path.home() / ".inferagrid"
LOCK_FILE = CONFIG_DIR / "resource_lock.json"
NODE_ID_FILE = CONFIG_DIR / "node_id"


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def get_or_create_node_id() -> str:
    """Get existing node ID or create a new one"""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    
    if NODE_ID_FILE.exists():
        return NODE_ID_FILE.read_text().strip()
    
    node_id = f"node_{uuid.uuid4().hex[:12]}"
    NODE_ID_FILE.write_text(node_id)
    return node_id


def clear_screen():
    """Clear terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_header():
    """Print application header"""
    print(f"\n{Colors.BOLD}{Colors.HEADER}")
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║            🔒 INFERAGRID RESOURCE LOCKER 🔒                      ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print(f"{Colors.ENDC}")


def gather_system_resources() -> Dict[str, Any]:
    """Gather all system resources"""
    print(f"\n{Colors.CYAN}Scanning system resources...{Colors.ENDC}\n")
    
    system_info = get_system_info()
    cpu_info = get_cpu_info()
    mem_info = get_memory_info()
    disk_info = get_disk_info()
    net_info = get_network_info()
    
    # Try different GPU detection methods
    gpu_info = get_gpu_info_nvidia()
    if not gpu_info:
        gpu_info = get_gpu_info_macos()
    
    # Check for Apple Silicon
    apple_silicon = get_gpu_info_apple_silicon()
    if apple_silicon and not gpu_info:
        gpu_info = [apple_silicon]
    
    return {
        "system": system_info,
        "cpu": cpu_info,
        "memory": mem_info,
        "disk": disk_info,
        "network": net_info,
        "gpu": gpu_info or []
    }


def calculate_available_resources(resources: Dict) -> Dict[str, Any]:
    """Calculate what's available to offer (before user lock)"""
    cpu_info = resources["cpu"]
    mem_info = resources["memory"]
    disk_info = resources["disk"]
    gpu_info = resources["gpu"]
    
    # CPU: available cores based on current usage
    total_cores = cpu_info.get("total_cores", 0)
    cpu_available_percent = cpu_info.get("available_percent", 100)
    available_cores = total_cores * (cpu_available_percent / 100)
    
    # Memory: available GB
    available_mem_bytes = mem_info.get("available", 0)
    available_mem_gb = available_mem_bytes / (1024**3)
    
    # Disk: available GB
    available_disk_bytes = disk_info.get("available", 0)
    available_disk_gb = available_disk_bytes / (1024**3)
    
    # GPU info
    gpu_name = None
    vram_mb = None
    vram_available_mb = None
    
    if gpu_info:
        gpu = gpu_info[0]
        gpu_name = gpu.get("name", "Unknown GPU")
        vram_mb = gpu.get("vram_mb") or gpu.get("memory_total_mb")
        vram_available_mb = gpu.get("memory_free_mb") or vram_mb
    
    return {
        "cpu_cores_total": total_cores,
        "cpu_cores_available": round(available_cores, 2),
        "memory_total_gb": round(mem_info.get("total", 0) / (1024**3), 2),
        "memory_available_gb": round(available_mem_gb, 2),
        "storage_total_gb": round(disk_info.get("total", 0) / (1024**3), 2),
        "storage_available_gb": round(available_disk_gb, 2),
        "gpu_name": gpu_name,
        "gpu_vram_total_mb": vram_mb,
        "gpu_vram_available_mb": vram_available_mb,
    }


def display_available_resources(available: Dict):
    """Display available resources in a nice format"""
    print(f"{Colors.BOLD}{Colors.CYAN}┌─ AVAILABLE RESOURCES ────────────────────────────────────────────┐{Colors.ENDC}")
    print(f"  {Colors.BOLD}Resources You Can Offer:{Colors.ENDC}")
    print(f"  ├─ CPU: {available['cpu_cores_available']:.1f} cores (of {available['cpu_cores_total']} total)")
    print(f"  ├─ Memory: {available['memory_available_gb']:.2f} GB (of {available['memory_total_gb']:.2f} GB total)")
    print(f"  ├─ Storage: {available['storage_available_gb']:.2f} GB (of {available['storage_total_gb']:.2f} GB total)")
    
    if available['gpu_name']:
        print(f"  └─ GPU: {available['gpu_name']}")
        if available['gpu_vram_total_mb']:
            vram_gb = available['gpu_vram_total_mb'] / 1024
            print(f"       └─ VRAM: {vram_gb:.1f} GB")
    else:
        print(f"  └─ GPU: {Colors.YELLOW}None (CPU-only){Colors.ENDC}")
    print()


def calculate_with_buffer_and_lock(available: Dict, lock_percent: int) -> Dict[str, Any]:
    """
    Calculate final locked resources with 20% safety buffer
    
    Formula:
    - User keeps: 100% - lock_percent
    - Safety buffer: 20% of total
    - InferaGrid gets: lock_percent - safety_buffer (if positive)
    
    Example with 50% lock:
    - Available: 5.3 cores
    - Safety buffer (20%): 1.06 cores reserved for machine
    - User requested lock: 50% = 2.65 cores
    - After buffer: max(0, 2.65 - 1.06) = 1.59 cores for InferaGrid
    - User keeps: 5.3 - 1.59 = 3.71 cores
    """
    
    result = {
        "lock_percent_requested": lock_percent,
        "safety_buffer_percent": SAFETY_BUFFER_PERCENT,
    }
    
    # CPU
    cpu_available = available['cpu_cores_available']
    cpu_buffer = cpu_available * (SAFETY_BUFFER_PERCENT / 100)
    cpu_lock_requested = cpu_available * (lock_percent / 100)
    cpu_for_inferagrid = max(0, cpu_lock_requested)
    cpu_user_keeps = cpu_available - cpu_for_inferagrid
    # Ensure user always has at least the buffer
    if cpu_user_keeps < cpu_buffer:
        cpu_for_inferagrid = cpu_available - cpu_buffer
        cpu_user_keeps = cpu_buffer
    
    result["cpu"] = {
        "available": round(cpu_available, 2),
        "buffer_reserved": round(cpu_buffer, 2),
        "locked_for_inferagrid": round(cpu_for_inferagrid, 2),
        "user_keeps": round(cpu_user_keeps, 2),
    }
    
    # Memory
    mem_available = available['memory_available_gb']
    mem_buffer = mem_available * (SAFETY_BUFFER_PERCENT / 100)
    mem_lock_requested = mem_available * (lock_percent / 100)
    mem_for_inferagrid = max(0, mem_lock_requested)
    mem_user_keeps = mem_available - mem_for_inferagrid
    if mem_user_keeps < mem_buffer:
        mem_for_inferagrid = mem_available - mem_buffer
        mem_user_keeps = mem_buffer
    
    result["memory"] = {
        "available_gb": round(mem_available, 2),
        "buffer_reserved_gb": round(mem_buffer, 2),
        "locked_for_inferagrid_gb": round(mem_for_inferagrid, 2),
        "user_keeps_gb": round(mem_user_keeps, 2),
    }
    
    # Storage
    storage_available = available['storage_available_gb']
    storage_buffer = storage_available * (SAFETY_BUFFER_PERCENT / 100)
    storage_lock_requested = storage_available * (lock_percent / 100)
    storage_for_inferagrid = min(max(0, storage_lock_requested), 100)  # Cap at 100GB
    storage_user_keeps = storage_available - storage_for_inferagrid
    if storage_user_keeps < storage_buffer:
        storage_for_inferagrid = max(0, storage_available - storage_buffer)
        storage_user_keeps = storage_buffer
    
    result["storage"] = {
        "available_gb": round(storage_available, 2),
        "buffer_reserved_gb": round(storage_buffer, 2),
        "locked_for_inferagrid_gb": round(storage_for_inferagrid, 2),
        "user_keeps_gb": round(storage_user_keeps, 2),
    }
    
    # GPU
    if available['gpu_name']:
        vram_mb = available['gpu_vram_available_mb'] or available['gpu_vram_total_mb'] or 0
        vram_gb = vram_mb / 1024
        vram_buffer = vram_gb * (SAFETY_BUFFER_PERCENT / 100)
        vram_lock_requested = vram_gb * (lock_percent / 100)
        vram_for_inferagrid = max(0, vram_lock_requested)
        vram_user_keeps = vram_gb - vram_for_inferagrid
        if vram_user_keeps < vram_buffer:
            vram_for_inferagrid = vram_gb - vram_buffer
            vram_user_keeps = vram_buffer
        
        result["gpu"] = {
            "name": available['gpu_name'],
            "vram_available_gb": round(vram_gb, 2),
            "vram_buffer_reserved_gb": round(vram_buffer, 2),
            "vram_locked_for_inferagrid_gb": round(vram_for_inferagrid, 2),
            "vram_user_keeps_gb": round(vram_user_keeps, 2),
        }
    else:
        result["gpu"] = None
    
    return result


def display_lock_calculation(lock_result: Dict, category: CategoryDefinition):
    """Display the lock calculation results"""
    lock_percent = lock_result["lock_percent_requested"]
    buffer_percent = lock_result["safety_buffer_percent"]
    
    print(f"\n{Colors.BOLD}{Colors.GREEN}╔══════════════════════════════════════════════════════════════════╗")
    print(f"║              🔒 RESOURCE LOCK CALCULATION 🔒                     ║")
    print(f"╚══════════════════════════════════════════════════════════════════╝{Colors.ENDC}")
    
    print(f"\n  {Colors.BOLD}Lock Configuration:{Colors.ENDC}")
    print(f"  ├─ Requested Lock: {lock_percent}%")
    print(f"  └─ Safety Buffer:  {buffer_percent}% (always reserved for your machine)")
    
    # CPU breakdown
    cpu = lock_result["cpu"]
    print(f"\n  {Colors.BOLD}CPU Allocation:{Colors.ENDC}")
    print(f"  ├─ Available:            {cpu['available']:.2f} cores")
    print(f"  ├─ Safety Buffer (20%):  {cpu['buffer_reserved']:.2f} cores")
    print(f"  ├─ {Colors.GREEN}→ InferaGrid Gets:    {cpu['locked_for_inferagrid']:.2f} cores{Colors.ENDC}")
    print(f"  └─ {Colors.CYAN}→ You Keep:           {cpu['user_keeps']:.2f} cores{Colors.ENDC}")
    
    # Memory breakdown
    mem = lock_result["memory"]
    print(f"\n  {Colors.BOLD}Memory Allocation:{Colors.ENDC}")
    print(f"  ├─ Available:            {mem['available_gb']:.2f} GB")
    print(f"  ├─ Safety Buffer (20%):  {mem['buffer_reserved_gb']:.2f} GB")
    print(f"  ├─ {Colors.GREEN}→ InferaGrid Gets:    {mem['locked_for_inferagrid_gb']:.2f} GB{Colors.ENDC}")
    print(f"  └─ {Colors.CYAN}→ You Keep:           {mem['user_keeps_gb']:.2f} GB{Colors.ENDC}")
    
    # Storage breakdown
    storage = lock_result["storage"]
    print(f"\n  {Colors.BOLD}Storage Allocation:{Colors.ENDC}")
    print(f"  ├─ Available:            {storage['available_gb']:.2f} GB")
    print(f"  ├─ Safety Buffer (20%):  {storage['buffer_reserved_gb']:.2f} GB")
    print(f"  ├─ {Colors.GREEN}→ InferaGrid Gets:    {storage['locked_for_inferagrid_gb']:.2f} GB{Colors.ENDC}")
    print(f"  └─ {Colors.CYAN}→ You Keep:           {storage['user_keeps_gb']:.2f} GB{Colors.ENDC}")
    
    # GPU breakdown
    if lock_result["gpu"]:
        gpu = lock_result["gpu"]
        print(f"\n  {Colors.BOLD}GPU Allocation ({gpu['name']}):{Colors.ENDC}")
        print(f"  ├─ VRAM Available:       {gpu['vram_available_gb']:.2f} GB")
        print(f"  ├─ Safety Buffer (20%):  {gpu['vram_buffer_reserved_gb']:.2f} GB")
        print(f"  ├─ {Colors.GREEN}→ InferaGrid Gets:    {gpu['vram_locked_for_inferagrid_gb']:.2f} GB{Colors.ENDC}")
        print(f"  └─ {Colors.CYAN}→ You Keep:           {gpu['vram_user_keeps_gb']:.2f} GB{Colors.ENDC}")
    
    # Category classification
    print(f"\n{Colors.BOLD}{Colors.YELLOW}╔══════════════════════════════════════════════════════════════════╗")
    print(f"║              📊 NODE CLASSIFICATION 📊                            ║")
    print(f"╚══════════════════════════════════════════════════════════════════╝{Colors.ENDC}")
    
    print(f"\n  {Colors.BOLD}Category:{Colors.ENDC} {category.display_name}")
    print(f"  {Colors.BOLD}Description:{Colors.ENDC} {category.description}")
    
    print(f"\n  {Colors.BOLD}Classification Details:{Colors.ENDC}")
    print(f"  ├─ Inference Speed:  {category.inference_speed.value.replace('_', ' ').title()}")
    print(f"  ├─ Compute Tier:     {category.compute_tier.value.title()}")
    print(f"  ├─ GPU Capability:   {category.gpu_capability.value.replace('_', ' ').title()}")
    print(f"  └─ Price Multiplier: {category.base_rate_multiplier:.1f}x base rate")
    
    print(f"\n  {Colors.BOLD}Supported Workloads:{Colors.ENDC}")
    for i, workload in enumerate(category.supported_workloads):
        prefix = "└─" if i == len(category.supported_workloads) - 1 else "├─"
        print(f"  {prefix} {workload.value.replace('_', ' ').title()}")
    
    print(f"\n  {Colors.BOLD}Recommended Tasks:{Colors.ENDC}")
    for i, task in enumerate(category.recommended_tasks):
        prefix = "└─" if i == len(category.recommended_tasks) - 1 else "├─"
        print(f"  {prefix} {task}")
    
    print(f"\n  {Colors.BOLD}Limits:{Colors.ENDC}")
    print(f"  ├─ Max Concurrent Jobs: {category.max_concurrent_jobs}")
    print(f"  ├─ Max Context Length:  {category.max_context_length:,} tokens")
    print(f"  └─ Max Batch Size:      {category.max_batch_size}")
    
    # Estimated earnings
    base_rate = 0.01 * category.base_rate_multiplier
    cpu_rate = cpu['locked_for_inferagrid'] * base_rate
    hourly_rate = cpu_rate + (0.01 if lock_result["gpu"] else 0)
    
    print(f"\n  {Colors.BOLD}Estimated Earnings (if fully utilized):{Colors.ENDC}")
    print(f"  ├─ Hourly:  ${hourly_rate:.3f}")
    print(f"  ├─ Daily:   ${hourly_rate * 24:.2f}")
    print(f"  └─ Monthly: ${hourly_rate * 24 * 30:.2f}")


def save_lock_config(
    node_id: str,
    available: Dict,
    lock_result: Dict,
    category: CategoryDefinition,
    resources: Dict
) -> Dict:
    """Save the lock configuration to file"""
    
    config = {
        "version": "1.0.0",
        "node_id": node_id,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        
        "system_info": resources["system"],
        
        "lock_config": {
            "lock_percent": lock_result["lock_percent_requested"],
            "safety_buffer_percent": lock_result["safety_buffer_percent"],
            "status": "active",
        },
        
        "resources_available": available,
        "resources_locked": {
            "cpu_cores": lock_result["cpu"]["locked_for_inferagrid"],
            "memory_gb": lock_result["memory"]["locked_for_inferagrid_gb"],
            "storage_gb": lock_result["storage"]["locked_for_inferagrid_gb"],
            "gpu": lock_result["gpu"],
        },
        "resources_user_keeps": {
            "cpu_cores": lock_result["cpu"]["user_keeps"],
            "memory_gb": lock_result["memory"]["user_keeps_gb"],
            "storage_gb": lock_result["storage"]["user_keeps_gb"],
        },
        
        "category": {
            "display_name": category.display_name,
            "inference_speed": category.inference_speed.value,
            "compute_tier": category.compute_tier.value,
            "gpu_capability": category.gpu_capability.value,
            "supported_workloads": [w.value for w in category.supported_workloads],
            "recommended_tasks": category.recommended_tasks,
            "base_rate_multiplier": category.base_rate_multiplier,
            "max_concurrent_jobs": category.max_concurrent_jobs,
            "max_context_length": category.max_context_length,
            "max_batch_size": category.max_batch_size,
        },
        
        "api_config": {
            "enabled": True,
            "port": 8420,
            "host": "0.0.0.0",
            "endpoints": {
                "health": "/health",
                "info": "/node/info",
                "resources": "/node/resources",
                "jobs": "/jobs",
            }
        }
    }
    
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOCK_FILE, "w") as f:
        json.dump(config, f, indent=2)
    
    return config


def load_lock_config() -> Optional[Dict]:
    """Load existing lock configuration"""
    if LOCK_FILE.exists():
        with open(LOCK_FILE, "r") as f:
            return json.load(f)
    return None


def get_user_lock_percentage() -> int:
    """Interactive prompt to get lock percentage from user"""
    print(f"\n{Colors.BOLD}How much of your available resources do you want to lock for InferaGrid?{Colors.ENDC}")
    print(f"{Colors.YELLOW}Note: A 20% safety buffer is always reserved for your machine.{Colors.ENDC}\n")
    
    print("  Suggested options:")
    print("  ├─ 30% - Light contribution (good for active work machine)")
    print("  ├─ 50% - Balanced (recommended for most users)")
    print("  ├─ 70% - Heavy contribution (when machine is mostly idle)")
    print("  └─ Custom - Enter your own percentage (10-90)")
    
    while True:
        try:
            user_input = input(f"\n{Colors.CYAN}Enter percentage to lock (10-90): {Colors.ENDC}").strip()
            
            if user_input.endswith('%'):
                user_input = user_input[:-1]
            
            lock_percent = int(user_input)
            
            if 10 <= lock_percent <= 90:
                return lock_percent
            else:
                print(f"{Colors.RED}Please enter a value between 10 and 90.{Colors.ENDC}")
        
        except ValueError:
            print(f"{Colors.RED}Invalid input. Please enter a number.{Colors.ENDC}")
        except KeyboardInterrupt:
            print(f"\n{Colors.YELLOW}Cancelled.{Colors.ENDC}")
            sys.exit(0)


def confirm_lock(lock_result: Dict) -> bool:
    """Ask user to confirm the lock"""
    print(f"\n{Colors.BOLD}{Colors.YELLOW}═══════════════════════════════════════════════════════════════════{Colors.ENDC}")
    print(f"{Colors.BOLD}Summary - Resources to be locked for InferaGrid:{Colors.ENDC}")
    print(f"  • CPU:     {lock_result['cpu']['locked_for_inferagrid']:.2f} cores")
    print(f"  • Memory:  {lock_result['memory']['locked_for_inferagrid_gb']:.2f} GB")
    print(f"  • Storage: {lock_result['storage']['locked_for_inferagrid_gb']:.2f} GB")
    if lock_result['gpu']:
        print(f"  • GPU:     {lock_result['gpu']['name']} ({lock_result['gpu']['vram_locked_for_inferagrid_gb']:.2f} GB VRAM)")
    
    while True:
        confirm = input(f"\n{Colors.CYAN}Confirm this lock? (yes/no): {Colors.ENDC}").strip().lower()
        if confirm in ['yes', 'y']:
            return True
        elif confirm in ['no', 'n']:
            return False
        else:
            print("Please enter 'yes' or 'no'")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="InferaGrid Resource Locker")
    parser.add_argument("--categories", action="store_true", help="Show all category definitions")
    parser.add_argument("--status", action="store_true", help="Show current lock status")
    parser.add_argument("--unlock", action="store_true", help="Remove current lock")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()
    
    # Show categories
    if args.categories:
        print_all_categories()
        return
    
    # Show status
    if args.status:
        config = load_lock_config()
        if config:
            if args.json:
                print(json.dumps(config, indent=2))
            else:
                print(f"\n{Colors.GREEN}Lock Status: ACTIVE{Colors.ENDC}")
                print(f"Node ID: {config['node_id']}")
                print(f"Lock Percent: {config['lock_config']['lock_percent']}%")
                print(f"Category: {config['category']['display_name']}")
                print(f"\nLocked Resources:")
                print(f"  CPU: {config['resources_locked']['cpu_cores']:.2f} cores")
                print(f"  Memory: {config['resources_locked']['memory_gb']:.2f} GB")
                print(f"  Storage: {config['resources_locked']['storage_gb']:.2f} GB")
                print(f"\nConfig file: {LOCK_FILE}")
        else:
            print(f"\n{Colors.YELLOW}No active lock. Run without arguments to create one.{Colors.ENDC}")
        return
    
    # Unlock
    if args.unlock:
        if LOCK_FILE.exists():
            LOCK_FILE.unlink()
            print(f"{Colors.GREEN}Lock removed successfully.{Colors.ENDC}")
        else:
            print(f"{Colors.YELLOW}No active lock to remove.{Colors.ENDC}")
        return
    
    # Main flow - interactive lock creation
    clear_screen()
    print_header()
    
    # Get node ID
    node_id = get_or_create_node_id()
    print(f"  Node ID: {Colors.CYAN}{node_id}{Colors.ENDC}")
    
    # Check for existing lock
    existing_config = load_lock_config()
    if existing_config:
        print(f"\n{Colors.YELLOW}⚠️  Existing lock found!{Colors.ENDC}")
        print(f"   Current lock: {existing_config['lock_config']['lock_percent']}%")
        print(f"   Category: {existing_config['category']['display_name']}")
        
        override = input(f"\n{Colors.CYAN}Override existing lock? (yes/no): {Colors.ENDC}").strip().lower()
        if override not in ['yes', 'y']:
            print("Keeping existing lock.")
            return
    
    # Gather resources
    resources = gather_system_resources()
    available = calculate_available_resources(resources)
    
    # Display available resources
    display_available_resources(available)
    
    # Get user input for lock percentage
    lock_percent = get_user_lock_percentage()
    
    # Calculate with buffer
    lock_result = calculate_with_buffer_and_lock(available, lock_percent)
    
    # Classify the node
    category = classify_node(
        cpu_cores=lock_result["cpu"]["locked_for_inferagrid"],
        memory_gb=lock_result["memory"]["locked_for_inferagrid_gb"],
        storage_gb=lock_result["storage"]["locked_for_inferagrid_gb"],
        gpu_name=available["gpu_name"],
        vram_mb=available["gpu_vram_available_mb"]
    )
    
    # Display results
    display_lock_calculation(lock_result, category)
    
    # Confirm
    if confirm_lock(lock_result):
        config = save_lock_config(node_id, available, lock_result, category, resources)
        
        print(f"\n{Colors.GREEN}✓ Lock saved successfully!{Colors.ENDC}")
        print(f"  Config file: {LOCK_FILE}")
        print(f"\n{Colors.BOLD}Next Steps:{Colors.ENDC}")
        print(f"  1. Run the API server: python api_server.py")
        print(f"  2. Your node will be available at: http://localhost:8420")
        print(f"  3. Register with InferaGrid Exchange to start earning!")
        
        if args.json:
            print(f"\n{Colors.CYAN}Lock Configuration:{Colors.ENDC}")
            print(json.dumps(config, indent=2))
    else:
        print(f"\n{Colors.YELLOW}Lock cancelled.{Colors.ENDC}")


if __name__ == "__main__":
    main()
