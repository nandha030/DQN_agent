#!/usr/bin/env python3
"""
Dheera Web API Server
FastAPI backend with WebSocket support for real-time chat
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import json
import asyncio
from datetime import datetime
import tempfile
import shutil

from dheera import Dheera
from api.llm_router import LLMRouter, LLMConfig, PROVIDER_PRESETS


# ==================== FastAPI App ====================

app = FastAPI(
    title="Dheera API",
    description="Brain-inspired AI with hot-swappable LLM backends",
    version="0.3.1",
)

# CORS middleware (allow frontend to connect)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== Global State ====================

class DheeraServer:
    """Global server state"""
    def __init__(self):
        self.dheera: Optional[Dheera] = None
        self.llm_router = LLMRouter()
        self.episode_id: Optional[str] = None
        self.active_connections: List[WebSocket] = []

        # Initialize with default local provider
        self._init_default_providers()

    def _init_default_providers(self):
        """Initialize default providers - auto-discover Ollama models"""
        import requests

        # Auto-discover all Ollama models
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                models = data.get("models", [])

                # Add all discovered models
                for idx, model_info in enumerate(models):
                    model_name = model_info.get("name", "")
                    if model_name:
                        # Create a clean provider name
                        provider_name = f"ollama_{model_name.replace(':', '_').replace('.', '_')}"

                        # Create config for this model
                        config = LLMConfig(
                            provider="ollama",
                            model=model_name,
                            base_url="http://localhost:11434",
                            timeout=30,
                            max_tokens=256,
                            temperature=0.7,
                        )

                        self.llm_router.add_provider(provider_name, config)

                        # Set first model as active
                        if idx == 0:
                            self.llm_router.switch_provider(provider_name)

                print(f"✅ Auto-discovered {len(models)} Ollama models")
            else:
                # Fallback to phi3:mini if Ollama not available
                self.llm_router.add_provider("local_phi3", PROVIDER_PRESETS["ollama_phi3"])
                print("⚠️  Ollama not available, using default phi3:mini")
        except Exception as e:
            # Fallback to default
            self.llm_router.add_provider("local_phi3", PROVIDER_PRESETS["ollama_phi3"])
            print(f"⚠️  Could not discover Ollama models: {e}")

    def init_dheera(self, config_path: str = "config/dheera_config.yaml"):
        """Initialize Dheera engine"""
        if self.dheera is None:
            self.dheera = Dheera(config_path=config_path)
            self.episode_id = self.dheera.start_episode()

server = DheeraServer()


# ==================== Pydantic Models ====================

class ChatRequest(BaseModel):
    message: str
    force_action: Optional[int] = None
    force_search: bool = False
    system_prompt: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    metadata: Dict[str, Any]


class LLMProviderRequest(BaseModel):
    name: str
    provider: str  # ollama, openai, anthropic, litellm
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout: int = 30
    max_tokens: int = 256
    temperature: float = 0.7


class LLMSwitchRequest(BaseModel):
    provider_name: str


class ConfigUpdate(BaseModel):
    config_dict: Dict[str, Any]


# ==================== Health & Info ====================

@app.get("/")
async def root():
    return {
        "name": "Dheera API",
        "version": "0.3.1",
        "description": "Brain-inspired AI with hot-swappable LLM backends",
        "endpoints": {
            "chat": "/api/chat",
            "websocket": "/ws/chat",
            "llm": "/api/llm/*",
            "config": "/api/config",
            "stats": "/api/stats",
        },
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "dheera_initialized": server.dheera is not None,
        "episode_id": server.episode_id,
        "active_provider": server.llm_router.active_provider,
        "timestamp": datetime.utcnow().isoformat(),
    }


# ==================== Chat API ====================

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Send a message to Dheera (REST endpoint)"""
    if server.dheera is None:
        server.init_dheera()

    try:
        # Run blocking process_message in thread pool to avoid blocking event loop
        response, metadata = await asyncio.to_thread(
            server.dheera.process_message,
            user_message=request.message,
            force_action=request.force_action,
            force_search=request.force_search,
            system_prompt=request.system_prompt,
        )

        return ChatResponse(response=response, metadata=metadata)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """WebSocket endpoint for real-time chat"""
    await websocket.accept()
    server.active_connections.append(websocket)

    if server.dheera is None:
        server.init_dheera()

    try:
        await websocket.send_json({
            "type": "connected",
            "episode_id": server.episode_id,
            "active_provider": server.llm_router.active_provider,
        })

        while True:
            # Receive message from client
            data = await websocket.receive_json()

            message_type = data.get("type", "chat")

            if message_type == "chat":
                # Send "thinking" status
                await websocket.send_json({
                    "type": "status",
                    "status": "thinking",
                })

                # Process message
                response, metadata = server.dheera.process_message(
                    user_message=data.get("message", ""),
                    force_action=data.get("force_action"),
                    force_search=data.get("force_search", False),
                )

                # Send response
                await websocket.send_json({
                    "type": "response",
                    "response": response,
                    "metadata": metadata,
                })

            elif message_type == "ping":
                await websocket.send_json({"type": "pong"})

    except WebSocketDisconnect:
        server.active_connections.remove(websocket)
    except Exception as e:
        await websocket.send_json({
            "type": "error",
            "error": str(e),
        })
        server.active_connections.remove(websocket)


# ==================== LLM Router API ====================

@app.get("/api/llm/providers")
async def list_providers():
    """List all configured LLM providers"""
    return {
        "providers": server.llm_router.list_providers(),
        "active": server.llm_router.active_provider,
    }


@app.post("/api/llm/provider")
async def add_provider(request: LLMProviderRequest):
    """Add a new LLM provider"""
    config = LLMConfig(
        provider=request.provider,
        model=request.model,
        api_key=request.api_key,
        base_url=request.base_url,
        timeout=request.timeout,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
    )

    success = server.llm_router.add_provider(request.name, config)

    if not success:
        raise HTTPException(status_code=400, detail=f"Invalid provider: {request.provider}")

    return {
        "success": True,
        "name": request.name,
        "provider": request.provider,
        "model": request.model,
    }


@app.delete("/api/llm/provider/{name}")
async def remove_provider(name: str):
    """Remove an LLM provider"""
    success = server.llm_router.remove_provider(name)

    if not success:
        raise HTTPException(status_code=404, detail=f"Provider '{name}' not found")

    return {"success": True, "removed": name}


@app.post("/api/llm/switch")
async def switch_provider(request: LLMSwitchRequest):
    """Hot-swap to a different LLM provider"""
    success = server.llm_router.switch_provider(request.provider_name)

    if not success:
        raise HTTPException(status_code=404, detail=f"Provider '{request.provider_name}' not found")

    # Notify all WebSocket clients
    for connection in server.active_connections:
        try:
            await connection.send_json({
                "type": "provider_switched",
                "provider": request.provider_name,
            })
        except:
            pass

    return {
        "success": True,
        "active_provider": request.provider_name,
    }


@app.get("/api/llm/test/{name}")
async def test_provider(name: str):
    """Test an LLM provider"""
    result = server.llm_router.test_provider(name)
    return result


@app.get("/api/llm/presets")
async def get_presets():
    """Get preset LLM configurations"""
    return {
        "presets": {
            name: {
                "provider": config.provider,
                "model": config.model,
                "timeout": config.timeout,
                "max_tokens": config.max_tokens,
            }
            for name, config in PROVIDER_PRESETS.items()
        }
    }


@app.post("/api/llm/discover")
async def discover_ollama_models():
    """Discover and add all available Ollama models"""
    import requests

    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = data.get("models", [])

            added_models = []
            for model_info in models:
                model_name = model_info.get("name", "")
                if model_name:
                    provider_name = f"ollama_{model_name.replace(':', '_').replace('.', '_')}"

                    # Check if already exists
                    if provider_name not in server.llm_router.providers:
                        config = LLMConfig(
                            provider="ollama",
                            model=model_name,
                            base_url="http://localhost:11434",
                            timeout=30,
                            max_tokens=256,
                            temperature=0.7,
                        )

                        server.llm_router.add_provider(provider_name, config)
                        added_models.append({
                            "name": provider_name,
                            "model": model_name,
                        })

            return {
                "success": True,
                "discovered": len(models),
                "added": len(added_models),
                "models": added_models,
            }
        else:
            return {
                "success": False,
                "error": "Ollama not available"
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


# ==================== Stats & Monitoring ====================

@app.get("/api/stats")
async def get_stats():
    """Get Dheera statistics"""
    if server.dheera is None:
        return {"error": "Dheera not initialized"}

    return {
        "dheera": server.dheera.get_stats(),
        "llm_router": server.llm_router.get_stats(),
        "episode_id": server.episode_id,
    }


@app.get("/api/stats/llm")
async def get_llm_stats():
    """Get LLM router statistics"""
    return server.llm_router.get_stats()


# ==================== Configuration ====================

@app.get("/api/config")
async def get_config():
    """Get current Dheera configuration"""
    if server.dheera is None:
        server.init_dheera()

    return {
        "config": server.dheera.config,
        "identity": server.dheera.identity,
        "active_provider": server.llm_router.active_provider,
    }


@app.post("/api/config/update")
async def update_config(request: ConfigUpdate):
    """Update Dheera configuration (runtime hot-swap)"""
    if server.dheera is None:
        server.init_dheera()

    # Update config dict
    for key, value in request.config_dict.items():
        if key in server.dheera.config:
            server.dheera.config[key] = value

    return {
        "success": True,
        "updated_keys": list(request.config_dict.keys()),
    }


# ==================== RAG Management ====================

@app.post("/api/rag/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    """
    Upload documents to RAG knowledge base
    Supports: PDF, TXT, DOCX, MD
    """
    if not server.dheera:
        server.init_dheera()

    results = []

    for file in files:
        try:
            # Get file extension
            file_ext = Path(file.filename).suffix.lower()

            # Validate file type
            if file_ext not in ['.pdf', '.txt', '.docx', '.md']:
                results.append({
                    "filename": file.filename,
                    "success": False,
                    "error": f"Unsupported file type: {file_ext}"
                })
                continue

            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
                shutil.copyfileobj(file.file, tmp_file)
                tmp_path = tmp_file.name

            try:
                # Read file content based on type
                content = ""

                if file_ext == '.txt' or file_ext == '.md':
                    with open(tmp_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                elif file_ext == '.pdf':
                    # Try to import PyPDF2
                    try:
                        import PyPDF2
                        with open(tmp_path, 'rb') as f:
                            pdf_reader = PyPDF2.PdfReader(f)
                            for page in pdf_reader.pages:
                                content += page.extract_text() + "\n"
                    except ImportError:
                        content = "[PDF parsing requires PyPDF2: pip install pypdf2]"

                elif file_ext == '.docx':
                    # Try to import python-docx
                    try:
                        from docx import Document
                        doc = Document(tmp_path)
                        content = "\n".join([para.text for para in doc.paragraphs])
                    except ImportError:
                        content = "[DOCX parsing requires python-docx: pip install python-docx]"

                # Add to RAG knowledge base
                if server.dheera.rag_engine and content:
                    # Add document with metadata
                    doc_id = server.dheera.rag_engine.add_document(
                        content,
                        metadata={
                            "filename": file.filename,
                            "upload_time": datetime.now().isoformat(),
                            "file_type": file_ext
                        }
                    )

                    results.append({
                        "filename": file.filename,
                        "success": True,
                        "doc_id": doc_id,
                        "chars": len(content),
                        "message": f"Added to knowledge base ({len(content)} chars)"
                    })
                else:
                    results.append({
                        "filename": file.filename,
                        "success": False,
                        "error": "RAG engine not available or empty content"
                    })

            finally:
                # Clean up temp file
                os.unlink(tmp_path)

        except Exception as e:
            results.append({
                "filename": file.filename,
                "success": False,
                "error": str(e)
            })

    return {
        "success": True,
        "total": len(files),
        "results": results
    }


@app.get("/api/rag/stats")
async def get_rag_stats():
    """Get RAG knowledge base statistics"""
    if not server.dheera or not server.dheera.rag_engine:
        return {
            "total_documents": 0,
            "total_chunks": 0,
            "total_queries": 0,
            "avg_score": 0.0,
            "recent_queries": []
        }

    rag = server.dheera.rag_engine

    # Get collection stats
    try:
        collection = rag.collection
        count = collection.count()

        return {
            "total_documents": count,
            "total_chunks": count,  # Each chunk is stored as a document
            "total_queries": getattr(rag, 'query_count', 0),
            "avg_score": getattr(rag, 'avg_relevance', 0.0),
            "recent_queries": getattr(rag, 'recent_queries', [])
        }
    except:
        return {
            "total_documents": 0,
            "total_chunks": 0,
            "total_queries": 0,
            "avg_score": 0.0,
            "recent_queries": []
        }


# ==================== Session Management ====================

@app.post("/api/session/new")
async def new_session():
    """Start a new chat session"""
    if server.dheera is None:
        server.init_dheera()
    else:
        server.dheera.end_episode("New session started")
        server.episode_id = server.dheera.start_episode()

    return {
        "episode_id": server.episode_id,
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.post("/api/session/end")
async def end_session():
    """End current chat session"""
    if server.dheera and server.episode_id:
        server.dheera.end_episode("Session ended via API")
        server.episode_id = None

    return {"success": True}


# ==================== Static Files (for GUI) ====================

# Serve static frontend files (after building React app)
# app.mount("/", StaticFiles(directory="gui/build", html=True), name="static")


# ==================== Main ====================

if __name__ == "__main__":
    import uvicorn

    print("🚀 Starting Dheera API Server...")
    print("   API docs: http://localhost:8000/docs")
    print("   Health: http://localhost:8000/health")
    print("   WebSocket: ws://localhost:8000/ws/chat")

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload on code changes
        log_level="info",
    )
