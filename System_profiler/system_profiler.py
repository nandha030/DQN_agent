#!/usr/bin/env python3
"""
InferaGrid System Profiler
Detects CPU, GPU, Memory and shows available resources for the marketplace.
Works on macOS (Intel/Apple Silicon), Linux, and Windows.
"""

import platform
import subprocess
import json
import os
from datetime import datetime
from typing import Optional, Dict, Any

# Optional imports with fallbacks
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# For NVIDIA GPUs
try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False


class Colors:
    """Terminal colors for pretty output"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def get_size(bytes_val: int, suffix: str = "B") -> str:
    """Convert bytes to human readable format"""
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]:
        if bytes_val < factor:
            return f"{bytes_val:.2f} {unit}{suffix}"
        bytes_val /= factor
    return f"{bytes_val:.2f} P{suffix}"


def get_percentage_bar(percentage: float, width: int = 30) -> str:
    """Create a visual percentage bar"""
    filled = int(width * percentage / 100)
    bar = '█' * filled + '░' * (width - filled)
    
    if percentage < 50:
        color = Colors.GREEN
    elif percentage < 80:
        color = Colors.YELLOW
    else:
        color = Colors.RED
    
    return f"{color}{bar}{Colors.ENDC} {percentage:.1f}%"


def get_system_info() -> Dict[str, Any]:
    """Get basic system information"""
    uname = platform.uname()
    return {
        "system": uname.system,
        "node_name": uname.node,
        "release": uname.release,
        "version": uname.version,
        "machine": uname.machine,
        "processor": uname.processor or platform.processor(),
        "python_version": platform.python_version(),
    }


def get_cpu_info() -> Dict[str, Any]:
    """Get CPU information and usage"""
    cpu_info = {
        "physical_cores": None,
        "total_cores": None,
        "max_frequency_mhz": None,
        "current_frequency_mhz": None,
        "usage_percent": None,
        "per_core_usage": [],
        "available_percent": None,
    }
    
    if PSUTIL_AVAILABLE:
        cpu_info["physical_cores"] = psutil.cpu_count(logical=False)
        cpu_info["total_cores"] = psutil.cpu_count(logical=True)
        
        # CPU frequency
        try:
            cpu_freq = psutil.cpu_freq()
            if cpu_freq:
                cpu_info["max_frequency_mhz"] = cpu_freq.max
                cpu_info["current_frequency_mhz"] = cpu_freq.current
        except Exception:
            pass
        
        # CPU usage
        cpu_info["usage_percent"] = psutil.cpu_percent(interval=1)
        cpu_info["per_core_usage"] = psutil.cpu_percent(interval=0.5, percpu=True)
        cpu_info["available_percent"] = 100 - cpu_info["usage_percent"]
    
    # macOS specific - get CPU brand
    if platform.system() == "Darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True
            )
            cpu_info["brand"] = result.stdout.strip()
        except Exception:
            pass
    
    return cpu_info


def get_memory_info() -> Dict[str, Any]:
    """Get RAM information"""
    mem_info = {
        "total": None,
        "available": None,
        "used": None,
        "usage_percent": None,
        "available_percent": None,
    }
    
    if PSUTIL_AVAILABLE:
        mem = psutil.virtual_memory()
        mem_info["total"] = mem.total
        mem_info["available"] = mem.available
        mem_info["used"] = mem.used
        mem_info["usage_percent"] = mem.percent
        mem_info["available_percent"] = 100 - mem.percent
        mem_info["total_human"] = get_size(mem.total)
        mem_info["available_human"] = get_size(mem.available)
        mem_info["used_human"] = get_size(mem.used)
    
    return mem_info


def get_gpu_info_nvidia() -> Optional[list]:
    """Get NVIDIA GPU information using GPUtil"""
    if not GPUTIL_AVAILABLE:
        return None
    
    try:
        gpus = GPUtil.getGPUs()
        if not gpus:
            return None
        
        gpu_list = []
        for gpu in gpus:
            gpu_list.append({
                "id": gpu.id,
                "name": gpu.name,
                "driver": gpu.driver,
                "memory_total_mb": gpu.memoryTotal,
                "memory_used_mb": gpu.memoryUsed,
                "memory_free_mb": gpu.memoryFree,
                "memory_usage_percent": gpu.memoryUtil * 100,
                "memory_available_percent": (1 - gpu.memoryUtil) * 100,
                "gpu_load_percent": gpu.load * 100,
                "gpu_available_percent": (1 - gpu.load) * 100,
                "temperature_c": gpu.temperature,
                "type": "NVIDIA"
            })
        return gpu_list
    except Exception:
        return None


def get_gpu_info_macos() -> Optional[list]:
    """Get GPU information on macOS using system_profiler"""
    if platform.system() != "Darwin":
        return None
    
    try:
        result = subprocess.run(
            ["system_profiler", "SPDisplaysDataType", "-json"],
            capture_output=True, text=True
        )
        data = json.loads(result.stdout)
        
        gpu_list = []
        displays = data.get("SPDisplaysDataType", [])
        
        for display in displays:
            gpu_info = {
                "name": display.get("sppci_model", "Unknown"),
                "vendor": display.get("sppci_vendor", "Unknown"),
                "vram": display.get("sppci_vram", "Unknown"),
                "device_type": display.get("sppci_device_type", "Unknown"),
                "metal_support": display.get("sppci_metal", "Unknown"),
                "type": "Integrated" if "Intel" in display.get("sppci_model", "") else "Discrete"
            }
            
            # Parse VRAM if available
            vram_str = display.get("sppci_vram", "")
            if "MB" in vram_str:
                try:
                    gpu_info["vram_mb"] = int(vram_str.replace(" MB", "").replace(",", ""))
                except ValueError:
                    pass
            elif "GB" in vram_str:
                try:
                    gpu_info["vram_mb"] = int(float(vram_str.replace(" GB", "").replace(",", "")) * 1024)
                except ValueError:
                    pass
            
            gpu_list.append(gpu_info)
        
        return gpu_list if gpu_list else None
    except Exception as e:
        return None


def get_gpu_info_apple_silicon() -> Optional[Dict]:
    """Detect Apple Silicon GPU (M1/M2/M3)"""
    if platform.system() != "Darwin":
        return None
    
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True
        )
        cpu_brand = result.stdout.strip()
        
        if "Apple" in cpu_brand:
            # It's Apple Silicon - GPU is integrated
            return {
                "name": f"{cpu_brand} GPU",
                "type": "Apple Silicon (Unified Memory)",
                "note": "GPU shares memory with CPU (Unified Memory Architecture)",
                "metal_support": "Yes",
                "neural_engine": "Yes"
            }
    except Exception:
        pass
    
    return None


def get_disk_info() -> Dict[str, Any]:
    """Get disk information"""
    disk_info = {
        "partitions": [],
        "total": None,
        "available": None,
    }
    
    if PSUTIL_AVAILABLE:
        try:
            # Get main disk (root partition)
            disk = psutil.disk_usage('/')
            disk_info["total"] = disk.total
            disk_info["used"] = disk.used
            disk_info["available"] = disk.free
            disk_info["usage_percent"] = disk.percent
            disk_info["available_percent"] = 100 - disk.percent
            disk_info["total_human"] = get_size(disk.total)
            disk_info["available_human"] = get_size(disk.free)
        except Exception:
            pass
    
    return disk_info


def get_network_info() -> Dict[str, Any]:
    """Get network interface information"""
    net_info = {
        "interfaces": [],
        "bandwidth_estimate": "Unknown"
    }
    
    if PSUTIL_AVAILABLE:
        try:
            interfaces = psutil.net_if_addrs()
            for interface_name, addresses in interfaces.items():
                if interface_name.startswith(('lo', 'veth', 'docker', 'br-')):
                    continue
                for addr in addresses:
                    if addr.family.name == 'AF_INET':
                        net_info["interfaces"].append({
                            "name": interface_name,
                            "ip": addr.address
                        })
        except Exception:
            pass
    
    return net_info


def calculate_marketplace_offering(cpu_info: Dict, mem_info: Dict, gpu_info: list, disk_info: Dict) -> Dict:
    """Calculate what resources can be offered to the marketplace"""
    
    # Default: offer 70% of available resources (keep 30% for system)
    OFFER_RATIO = 0.7
    
    offering = {
        "cpu": {
            "cores_available": None,
            "recommended_offer": None,
            "estimated_compute_units": None,
        },
        "memory": {
            "available_gb": None,
            "recommended_offer_gb": None,
        },
        "gpu": {
            "available": False,
            "type": None,
            "vram_mb": None,
            "compute_capability": None,
        },
        "storage": {
            "available_gb": None,
            "recommended_offer_gb": None,
        },
        "overall_tier": "CPU",  # CPU, GPU_CONSUMER, GPU_PROFESSIONAL, GPU_DATACENTER
        "estimated_hourly_rate_usd": 0.0,
    }
    
    # CPU offering
    if cpu_info.get("total_cores") and cpu_info.get("available_percent"):
        available_cores = cpu_info["total_cores"] * (cpu_info["available_percent"] / 100)
        offering["cpu"]["cores_available"] = round(available_cores, 1)
        offering["cpu"]["recommended_offer"] = round(available_cores * OFFER_RATIO, 1)
        # Rough compute units (1 modern core = ~10 compute units)
        offering["cpu"]["estimated_compute_units"] = round(offering["cpu"]["recommended_offer"] * 10)
    
    # Memory offering
    if mem_info.get("available"):
        available_gb = mem_info["available"] / (1024**3)
        offering["memory"]["available_gb"] = round(available_gb, 2)
        offering["memory"]["recommended_offer_gb"] = round(available_gb * OFFER_RATIO, 2)
    
    # Storage offering
    if disk_info.get("available"):
        available_gb = disk_info["available"] / (1024**3)
        offering["storage"]["available_gb"] = round(available_gb, 2)
        offering["storage"]["recommended_offer_gb"] = round(min(available_gb * 0.5, 100), 2)  # Max 100GB or 50%
    
    # GPU offering
    if gpu_info:
        gpu = gpu_info[0]  # Primary GPU
        offering["gpu"]["available"] = True
        offering["gpu"]["type"] = gpu.get("name", "Unknown")
        
        if gpu.get("memory_free_mb"):
            offering["gpu"]["vram_mb"] = gpu["memory_free_mb"]
            offering["gpu"]["vram_available_gb"] = round(gpu["memory_free_mb"] / 1024, 2)
        elif gpu.get("vram_mb"):
            offering["gpu"]["vram_mb"] = gpu["vram_mb"]
            offering["gpu"]["vram_available_gb"] = round(gpu["vram_mb"] / 1024, 2)
        
        # Determine GPU tier
        gpu_name = gpu.get("name", "").upper()
        if any(x in gpu_name for x in ["A100", "H100", "A6000", "RTX 6000"]):
            offering["overall_tier"] = "GPU_DATACENTER"
            offering["estimated_hourly_rate_usd"] = 2.50
        elif any(x in gpu_name for x in ["RTX 4090", "RTX 4080", "RTX 3090"]):
            offering["overall_tier"] = "GPU_PROFESSIONAL"
            offering["estimated_hourly_rate_usd"] = 0.80
        elif any(x in gpu_name for x in ["RTX", "GTX", "RADEON"]):
            offering["overall_tier"] = "GPU_CONSUMER"
            offering["estimated_hourly_rate_usd"] = 0.30
        elif "INTEL" in gpu_name:
            offering["overall_tier"] = "CPU"
            offering["estimated_hourly_rate_usd"] = 0.05
    
    # CPU-only pricing
    if offering["overall_tier"] == "CPU":
        cores = offering["cpu"].get("recommended_offer", 0)
        offering["estimated_hourly_rate_usd"] = round(cores * 0.01, 3)  # $0.01 per core/hour
    
    return offering


def print_report(system_info: Dict, cpu_info: Dict, mem_info: Dict, 
                 gpu_info: list, disk_info: Dict, offering: Dict):
    """Print a beautiful report to terminal"""
    
    print(f"\n{Colors.BOLD}{Colors.HEADER}")
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║              🖥️  INFERAGRID SYSTEM PROFILER  🖥️                  ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print(f"{Colors.ENDC}")
    
    # System Info
    print(f"{Colors.BOLD}{Colors.CYAN}┌─ SYSTEM INFORMATION ─────────────────────────────────────────────┐{Colors.ENDC}")
    print(f"  OS: {system_info['system']} {system_info['release']}")
    print(f"  Machine: {system_info['machine']}")
    print(f"  Hostname: {system_info['node_name']}")
    print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # CPU Info
    print(f"{Colors.BOLD}{Colors.CYAN}┌─ CPU ────────────────────────────────────────────────────────────┐{Colors.ENDC}")
    if cpu_info.get("brand"):
        print(f"  Model: {cpu_info['brand']}")
    print(f"  Cores: {cpu_info.get('physical_cores', 'N/A')} physical, {cpu_info.get('total_cores', 'N/A')} logical")
    if cpu_info.get("current_frequency_mhz"):
        print(f"  Frequency: {cpu_info['current_frequency_mhz']:.0f} MHz")
    if cpu_info.get("usage_percent") is not None:
        print(f"  Usage:     {get_percentage_bar(cpu_info['usage_percent'])}")
        print(f"  {Colors.GREEN}Available:  {cpu_info['available_percent']:.1f}%{Colors.ENDC}")
    print()
    
    # Memory Info
    print(f"{Colors.BOLD}{Colors.CYAN}┌─ MEMORY ─────────────────────────────────────────────────────────┐{Colors.ENDC}")
    if mem_info.get("total_human"):
        print(f"  Total: {mem_info['total_human']}")
        print(f"  Used:  {mem_info['used_human']} | Available: {mem_info['available_human']}")
        print(f"  Usage: {get_percentage_bar(mem_info['usage_percent'])}")
        print(f"  {Colors.GREEN}Available: {mem_info['available_percent']:.1f}%{Colors.ENDC}")
    print()
    
    # GPU Info
    print(f"{Colors.BOLD}{Colors.CYAN}┌─ GPU ────────────────────────────────────────────────────────────┐{Colors.ENDC}")
    if gpu_info:
        for i, gpu in enumerate(gpu_info):
            print(f"  [{i}] {gpu.get('name', 'Unknown GPU')}")
            print(f"      Type: {gpu.get('type', 'Unknown')}")
            
            if gpu.get("vram_mb"):
                print(f"      VRAM: {gpu['vram_mb']} MB ({gpu['vram_mb']/1024:.1f} GB)")
            elif gpu.get("vram"):
                print(f"      VRAM: {gpu['vram']}")
            
            if gpu.get("memory_usage_percent") is not None:
                print(f"      VRAM Usage: {get_percentage_bar(gpu['memory_usage_percent'])}")
                print(f"      {Colors.GREEN}VRAM Available: {gpu['memory_available_percent']:.1f}%{Colors.ENDC}")
            
            if gpu.get("gpu_load_percent") is not None:
                print(f"      GPU Load:   {get_percentage_bar(gpu['gpu_load_percent'])}")
                print(f"      {Colors.GREEN}GPU Available:  {gpu['gpu_available_percent']:.1f}%{Colors.ENDC}")
            
            if gpu.get("metal_support"):
                print(f"      Metal: {gpu['metal_support']}")
            
            if gpu.get("temperature_c"):
                print(f"      Temperature: {gpu['temperature_c']}°C")
    else:
        print(f"  {Colors.YELLOW}No dedicated GPU detected{Colors.ENDC}")
        print(f"  Note: Intel integrated graphics may be present")
    print()
    
    # Disk Info
    print(f"{Colors.BOLD}{Colors.CYAN}┌─ STORAGE ────────────────────────────────────────────────────────┐{Colors.ENDC}")
    if disk_info.get("total_human"):
        print(f"  Total: {disk_info['total_human']}")
        print(f"  Available: {disk_info['available_human']}")
        print(f"  Usage: {get_percentage_bar(disk_info['usage_percent'])}")
    print()
    
    # Marketplace Offering
    print(f"{Colors.BOLD}{Colors.GREEN}╔══════════════════════════════════════════════════════════════════╗")
    print(f"║            💰 MARKETPLACE OFFERING SUMMARY 💰                    ║")
    print(f"╚══════════════════════════════════════════════════════════════════╝{Colors.ENDC}")
    print()
    print(f"  {Colors.BOLD}Tier:{Colors.ENDC} {offering['overall_tier']}")
    print()
    print(f"  {Colors.BOLD}Resources You Can Offer:{Colors.ENDC}")
    print(f"  ├─ CPU: {offering['cpu'].get('recommended_offer', 'N/A')} cores ({offering['cpu'].get('estimated_compute_units', 'N/A')} compute units)")
    print(f"  ├─ Memory: {offering['memory'].get('recommended_offer_gb', 'N/A')} GB")
    print(f"  ├─ Storage: {offering['storage'].get('recommended_offer_gb', 'N/A')} GB")
    
    if offering['gpu']['available']:
        print(f"  └─ GPU: {offering['gpu']['type']}")
        if offering['gpu'].get('vram_available_gb'):
            print(f"       └─ VRAM: {offering['gpu']['vram_available_gb']} GB available")
    else:
        print(f"  └─ GPU: {Colors.YELLOW}None (CPU-only node){Colors.ENDC}")
    
    print()
    print(f"  {Colors.BOLD}Estimated Earnings:{Colors.ENDC}")
    hourly = offering['estimated_hourly_rate_usd']
    print(f"  ├─ Hourly:  ${hourly:.3f}")
    print(f"  ├─ Daily:   ${hourly * 24:.2f}")
    print(f"  └─ Monthly: ${hourly * 24 * 30:.2f}")
    print()
    
    # JSON Export hint
    print(f"{Colors.CYAN}─────────────────────────────────────────────────────────────────────{Colors.ENDC}")
    print(f"  Run with --json flag to export machine-readable data")
    print()


def export_json(system_info: Dict, cpu_info: Dict, mem_info: Dict, 
                gpu_info: list, disk_info: Dict, net_info: Dict, offering: Dict) -> Dict:
    """Export all data as JSON"""
    return {
        "timestamp": datetime.now().isoformat(),
        "system": system_info,
        "cpu": cpu_info,
        "memory": mem_info,
        "gpu": gpu_info or [],
        "disk": disk_info,
        "network": net_info,
        "marketplace_offering": offering,
        "inferagrid_version": "0.1.0"
    }


def main():
    import sys
    
    # Check dependencies
    if not PSUTIL_AVAILABLE:
        print(f"{Colors.YELLOW}Warning: psutil not installed. Install with: pip install psutil{Colors.ENDC}")
        print("Some features will be limited.\n")
    
    # Gather all info
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
    
    # Calculate marketplace offering
    offering = calculate_marketplace_offering(cpu_info, mem_info, gpu_info, disk_info)
    
    # Output
    if "--json" in sys.argv:
        data = export_json(system_info, cpu_info, mem_info, gpu_info, disk_info, net_info, offering)
        print(json.dumps(data, indent=2, default=str))
    else:
        print_report(system_info, cpu_info, mem_info, gpu_info, disk_info, offering)


if __name__ == "__main__":
    main()
