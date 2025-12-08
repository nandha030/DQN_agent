#!/usr/bin/env python3
"""
InferaGrid Provider API Server
Exposes locked compute resources to the InferaGrid Exchange network.
"""

import json
import os
import sys
import asyncio
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager

try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
    import uvicorn
except ImportError:
    print("Error: FastAPI and uvicorn required. Install with:")
    print("  pip install fastapi uvicorn[standard]")
    sys.exit(1)

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Warning: psutil not available. Real-time metrics will be limited.")


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

CONFIG_DIR = Path.home() / ".inferagrid"
LOCK_FILE = CONFIG_DIR / "resource_lock.json"
NODE_ID_FILE = CONFIG_DIR / "node_id"

# Server settings
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8420

# Heartbeat settings
HEARTBEAT_INTERVAL = 30  # seconds
EXCHANGE_URL = os.getenv("INFERAGRID_EXCHANGE_URL", "https://exchange.inferagrid.io")


# ═══════════════════════════════════════════════════════════════════════════════
# PYDANTIC MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class ResourceLimits(BaseModel):
    cpu_cores: float
    memory_gb: float
    storage_gb: float
    gpu_vram_gb: Optional[float] = None


class NodeCategory(BaseModel):
    display_name: str
    inference_speed: str
    compute_tier: str
    gpu_capability: str
    supported_workloads: List[str]
    recommended_tasks: List[str]
    base_rate_multiplier: float
    max_concurrent_jobs: int
    max_context_length: int
    max_batch_size: int


class NodeInfo(BaseModel):
    node_id: str
    version: str
    status: str
    uptime_seconds: float
    created_at: str
    updated_at: str
    
    system: Dict[str, Any]
    category: NodeCategory
    
    resources_locked: ResourceLimits
    resources_available: ResourceLimits
    resources_user_reserved: ResourceLimits
    
    lock_config: Dict[str, Any]
    api_config: Dict[str, Any]


class RealTimeMetrics(BaseModel):
    timestamp: str
    cpu_usage_percent: float
    memory_usage_percent: float
    available_for_jobs: bool
    current_jobs: int
    queue_depth: int
    
    cpu_available_cores: float
    memory_available_gb: float
    gpu_available: bool
    gpu_usage_percent: Optional[float] = None
    gpu_memory_used_gb: Optional[float] = None


class JobRequest(BaseModel):
    job_id: str
    job_type: str  # "inference", "embedding", "batch", "data_processing"
    priority: int = Field(default=5, ge=1, le=10)
    
    # Resource requirements
    required_cpu_cores: Optional[float] = None
    required_memory_gb: Optional[float] = None
    required_gpu_vram_gb: Optional[float] = None
    
    # Job payload
    model_id: Optional[str] = None
    input_data: Optional[Dict[str, Any]] = None
    config: Optional[Dict[str, Any]] = None
    
    # Timeout
    timeout_seconds: int = 300
    
    # Callback
    callback_url: Optional[str] = None


class JobResponse(BaseModel):
    job_id: str
    status: str  # "queued", "running", "completed", "failed", "rejected"
    message: str
    
    estimated_start_time: Optional[str] = None
    estimated_completion_time: Optional[str] = None
    
    result: Optional[Dict[str, Any]] = None


class HealthCheck(BaseModel):
    status: str
    node_id: str
    uptime_seconds: float
    timestamp: str
    
    lock_active: bool
    api_ready: bool
    accepting_jobs: bool
    
    current_load: Dict[str, float]


# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL STATE
# ═══════════════════════════════════════════════════════════════════════════════

class NodeState:
    def __init__(self):
        self.start_time = time.time()
        self.config: Optional[Dict] = None
        self.jobs_queue: List[Dict] = []
        self.active_jobs: Dict[str, Dict] = {}
        self.completed_jobs: List[Dict] = []
        self.accepting_jobs = True
        
    def load_config(self):
        if LOCK_FILE.exists():
            with open(LOCK_FILE) as f:
                self.config = json.load(f)
            return True
        return False
    
    @property
    def uptime(self) -> float:
        return time.time() - self.start_time
    
    @property
    def node_id(self) -> str:
        if self.config:
            return self.config.get("node_id", "unknown")
        if NODE_ID_FILE.exists():
            return NODE_ID_FILE.read_text().strip()
        return "unknown"
    
    def get_current_load(self) -> Dict[str, float]:
        """Get current system load"""
        load = {
            "cpu_percent": 0.0,
            "memory_percent": 0.0,
            "active_jobs": len(self.active_jobs),
            "queued_jobs": len(self.jobs_queue),
        }
        
        if PSUTIL_AVAILABLE:
            load["cpu_percent"] = psutil.cpu_percent(interval=0.1)
            load["memory_percent"] = psutil.virtual_memory().percent
        
        return load
    
    def can_accept_job(self, job: JobRequest) -> tuple[bool, str]:
        """Check if node can accept a job"""
        if not self.accepting_jobs:
            return False, "Node is not accepting jobs"
        
        if not self.config:
            return False, "No resource lock configured"
        
        locked = self.config.get("resources_locked", {})
        
        # Check resource requirements
        if job.required_cpu_cores:
            if job.required_cpu_cores > locked.get("cpu_cores", 0):
                return False, f"Insufficient CPU cores (need {job.required_cpu_cores}, have {locked.get('cpu_cores', 0)})"
        
        if job.required_memory_gb:
            if job.required_memory_gb > locked.get("memory_gb", 0):
                return False, f"Insufficient memory (need {job.required_memory_gb}GB, have {locked.get('memory_gb', 0)}GB)"
        
        if job.required_gpu_vram_gb:
            gpu = locked.get("gpu")
            if not gpu:
                return False, "GPU required but not available"
            if job.required_gpu_vram_gb > gpu.get("vram_locked_for_inferagrid_gb", 0):
                return False, f"Insufficient GPU VRAM"
        
        # Check max concurrent jobs
        max_jobs = self.config.get("category", {}).get("max_concurrent_jobs", 1)
        if len(self.active_jobs) >= max_jobs:
            return False, f"Max concurrent jobs reached ({max_jobs})"
        
        return True, "OK"


state = NodeState()


# ═══════════════════════════════════════════════════════════════════════════════
# API APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup
    print(f"\n{'='*60}")
    print("🚀 InferaGrid Provider API Starting...")
    
    if state.load_config():
        print(f"✓ Lock config loaded: {LOCK_FILE}")
        print(f"  Node ID: {state.node_id}")
        print(f"  Category: {state.config['category']['display_name']}")
        print(f"  Locked CPU: {state.config['resources_locked']['cpu_cores']:.2f} cores")
        print(f"  Locked Memory: {state.config['resources_locked']['memory_gb']:.2f} GB")
    else:
        print(f"⚠️  No lock config found at {LOCK_FILE}")
        print("   Run 'python resource_locker.py' first to configure resources")
    
    print(f"\n📡 API available at: http://localhost:{DEFAULT_PORT}")
    print(f"   Health: http://localhost:{DEFAULT_PORT}/health")
    print(f"   Node Info: http://localhost:{DEFAULT_PORT}/node/info")
    print(f"   Metrics: http://localhost:{DEFAULT_PORT}/node/metrics")
    print(f"{'='*60}\n")
    
    # Start background heartbeat task
    heartbeat_task = asyncio.create_task(heartbeat_loop())
    
    yield
    
    # Shutdown
    heartbeat_task.cancel()
    print("\n👋 InferaGrid Provider API shutting down...")


app = FastAPI(
    title="InferaGrid Provider API",
    description="API for InferaGrid compute provider nodes",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════════════════════════════
# BACKGROUND TASKS
# ═══════════════════════════════════════════════════════════════════════════════

async def heartbeat_loop():
    """Send periodic heartbeats to the exchange"""
    while True:
        try:
            await asyncio.sleep(HEARTBEAT_INTERVAL)
            # In production, this would send heartbeat to EXCHANGE_URL
            # For now, just log
            if state.config:
                load = state.get_current_load()
                print(f"💓 Heartbeat | CPU: {load['cpu_percent']:.1f}% | "
                      f"Mem: {load['memory_percent']:.1f}% | "
                      f"Jobs: {load['active_jobs']}/{load['queued_jobs']}")
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"Heartbeat error: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "service": "InferaGrid Provider API",
        "version": "1.0.0",
        "node_id": state.node_id,
        "status": "online" if state.config else "unconfigured",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthCheck, tags=["Health"])
async def health_check():
    """Health check endpoint for monitoring and exchange"""
    load = state.get_current_load()
    
    return HealthCheck(
        status="healthy" if state.config else "unconfigured",
        node_id=state.node_id,
        uptime_seconds=state.uptime,
        timestamp=datetime.utcnow().isoformat(),
        lock_active=state.config is not None,
        api_ready=True,
        accepting_jobs=state.accepting_jobs and state.config is not None,
        current_load=load,
    )


@app.get("/node/info", response_model=NodeInfo, tags=["Node"])
async def get_node_info():
    """Get complete node information"""
    if not state.config:
        raise HTTPException(status_code=503, detail="Node not configured. Run resource_locker.py first.")
    
    config = state.config
    
    return NodeInfo(
        node_id=config["node_id"],
        version=config["version"],
        status="active" if state.accepting_jobs else "paused",
        uptime_seconds=state.uptime,
        created_at=config["created_at"],
        updated_at=config["updated_at"],
        
        system=config["system_info"],
        
        category=NodeCategory(
            display_name=config["category"]["display_name"],
            inference_speed=config["category"]["inference_speed"],
            compute_tier=config["category"]["compute_tier"],
            gpu_capability=config["category"]["gpu_capability"],
            supported_workloads=config["category"]["supported_workloads"],
            recommended_tasks=config["category"]["recommended_tasks"],
            base_rate_multiplier=config["category"]["base_rate_multiplier"],
            max_concurrent_jobs=config["category"]["max_concurrent_jobs"],
            max_context_length=config["category"]["max_context_length"],
            max_batch_size=config["category"]["max_batch_size"],
        ),
        
        resources_locked=ResourceLimits(
            cpu_cores=config["resources_locked"]["cpu_cores"],
            memory_gb=config["resources_locked"]["memory_gb"],
            storage_gb=config["resources_locked"]["storage_gb"],
            gpu_vram_gb=config["resources_locked"]["gpu"]["vram_locked_for_inferagrid_gb"] if config["resources_locked"]["gpu"] else None,
        ),
        
        resources_available=ResourceLimits(
            cpu_cores=config["resources_available"]["cpu_cores_available"],
            memory_gb=config["resources_available"]["memory_available_gb"],
            storage_gb=config["resources_available"]["storage_available_gb"],
            gpu_vram_gb=config["resources_available"]["gpu_vram_available_mb"] / 1024 if config["resources_available"]["gpu_vram_available_mb"] else None,
        ),
        
        resources_user_reserved=ResourceLimits(
            cpu_cores=config["resources_user_keeps"]["cpu_cores"],
            memory_gb=config["resources_user_keeps"]["memory_gb"],
            storage_gb=config["resources_user_keeps"]["storage_gb"],
        ),
        
        lock_config=config["lock_config"],
        api_config=config["api_config"],
    )


@app.get("/node/metrics", response_model=RealTimeMetrics, tags=["Node"])
async def get_real_time_metrics():
    """Get real-time system metrics"""
    if not state.config:
        raise HTTPException(status_code=503, detail="Node not configured")
    
    locked = state.config["resources_locked"]
    load = state.get_current_load()
    
    # Calculate available resources based on current usage
    if PSUTIL_AVAILABLE:
        cpu_avail = locked["cpu_cores"] * (1 - load["cpu_percent"] / 100)
        mem = psutil.virtual_memory()
        mem_avail = min(locked["memory_gb"], (mem.available / 1024**3))
    else:
        cpu_avail = locked["cpu_cores"]
        mem_avail = locked["memory_gb"]
    
    return RealTimeMetrics(
        timestamp=datetime.utcnow().isoformat(),
        cpu_usage_percent=load["cpu_percent"],
        memory_usage_percent=load["memory_percent"],
        available_for_jobs=state.accepting_jobs and load["active_jobs"] < state.config["category"]["max_concurrent_jobs"],
        current_jobs=load["active_jobs"],
        queue_depth=load["queued_jobs"],
        cpu_available_cores=round(cpu_avail, 2),
        memory_available_gb=round(mem_avail, 2),
        gpu_available=locked.get("gpu") is not None,
    )


@app.get("/node/category", tags=["Node"])
async def get_node_category():
    """Get node category classification details"""
    if not state.config:
        raise HTTPException(status_code=503, detail="Node not configured")
    
    return state.config["category"]


@app.post("/node/pause", tags=["Node"])
async def pause_node():
    """Pause accepting new jobs"""
    state.accepting_jobs = False
    return {"status": "paused", "message": "Node will not accept new jobs"}


@app.post("/node/resume", tags=["Node"])
async def resume_node():
    """Resume accepting jobs"""
    state.accepting_jobs = True
    return {"status": "active", "message": "Node is now accepting jobs"}


# ═══════════════════════════════════════════════════════════════════════════════
# JOB ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/jobs/submit", response_model=JobResponse, tags=["Jobs"])
async def submit_job(job: JobRequest, background_tasks: BackgroundTasks):
    """Submit a job to this node"""
    # Check if we can accept the job
    can_accept, reason = state.can_accept_job(job)
    
    if not can_accept:
        return JobResponse(
            job_id=job.job_id,
            status="rejected",
            message=reason,
        )
    
    # Queue the job
    job_entry = {
        "job_id": job.job_id,
        "request": job.dict(),
        "status": "queued",
        "queued_at": datetime.utcnow().isoformat(),
        "started_at": None,
        "completed_at": None,
        "result": None,
    }
    
    state.jobs_queue.append(job_entry)
    
    # In production, this would trigger actual job processing
    # background_tasks.add_task(process_job, job_entry)
    
    return JobResponse(
        job_id=job.job_id,
        status="queued",
        message="Job queued successfully",
        estimated_start_time=datetime.utcnow().isoformat(),
    )


@app.get("/jobs/{job_id}", response_model=JobResponse, tags=["Jobs"])
async def get_job_status(job_id: str):
    """Get status of a specific job"""
    # Check active jobs
    if job_id in state.active_jobs:
        job = state.active_jobs[job_id]
        return JobResponse(
            job_id=job_id,
            status="running",
            message="Job is currently processing",
        )
    
    # Check queue
    for job in state.jobs_queue:
        if job["job_id"] == job_id:
            return JobResponse(
                job_id=job_id,
                status="queued",
                message=f"Position in queue: {state.jobs_queue.index(job) + 1}",
            )
    
    # Check completed
    for job in state.completed_jobs:
        if job["job_id"] == job_id:
            return JobResponse(
                job_id=job_id,
                status=job["status"],
                message="Job completed",
                result=job.get("result"),
            )
    
    raise HTTPException(status_code=404, detail="Job not found")


@app.get("/jobs", tags=["Jobs"])
async def list_jobs(status: Optional[str] = None, limit: int = 50):
    """List all jobs"""
    jobs = []
    
    # Active jobs
    if not status or status == "running":
        for job_id, job in state.active_jobs.items():
            jobs.append({"job_id": job_id, "status": "running", **job})
    
    # Queued jobs
    if not status or status == "queued":
        for job in state.jobs_queue:
            jobs.append(job)
    
    # Completed jobs
    if not status or status in ["completed", "failed"]:
        for job in state.completed_jobs[-limit:]:
            jobs.append(job)
    
    return {
        "total": len(jobs),
        "jobs": jobs[:limit],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# EXCHANGE INTEGRATION ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/exchange/registration-payload", tags=["Exchange"])
async def get_registration_payload():
    """Get payload for registering with InferaGrid Exchange"""
    if not state.config:
        raise HTTPException(status_code=503, detail="Node not configured")
    
    config = state.config
    
    return {
        "node_id": config["node_id"],
        "api_endpoint": f"http://YOUR_PUBLIC_IP:{DEFAULT_PORT}",  # User needs to configure
        
        "resources": {
            "cpu_cores": config["resources_locked"]["cpu_cores"],
            "memory_gb": config["resources_locked"]["memory_gb"],
            "storage_gb": config["resources_locked"]["storage_gb"],
            "gpu": config["resources_locked"]["gpu"],
        },
        
        "category": config["category"],
        
        "capabilities": {
            "supported_workloads": config["category"]["supported_workloads"],
            "max_concurrent_jobs": config["category"]["max_concurrent_jobs"],
            "max_context_length": config["category"]["max_context_length"],
        },
        
        "system": {
            "os": config["system_info"]["system"],
            "machine": config["system_info"]["machine"],
        },
        
        "pricing": {
            "base_rate_multiplier": config["category"]["base_rate_multiplier"],
            "accepts_igx_token": True,
            "accepts_usd": True,
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Run the API server"""
    import argparse
    
    parser = argparse.ArgumentParser(description="InferaGrid Provider API Server")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Host to bind to")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    args = parser.parse_args()
    
    uvicorn.run(
        "api_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
