#!/usr/bin/env python3
"""
Dheera Enhanced GUI - ChatGPT/Claude-style interface
Features:
- Session management (like ChatGPT/Claude)
- Advanced side menu (like LiteLLM)
- Chat history persistence
- Model comparison
- Settings panel
"""

import streamlit as st
import requests
import json
from datetime import datetime
import sys
import os
from typing import List, Dict, Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# ==================== Configuration ====================

API_BASE = "http://localhost:8000"

# ==================== Session Management ====================

class SessionManager:
    """Manage chat sessions (like ChatGPT/Claude)"""

    def __init__(self):
        if 'sessions' not in st.session_state:
            st.session_state.sessions = {}
        if 'current_session_id' not in st.session_state:
            st.session_state.current_session_id = None
        if 'session_counter' not in st.session_state:
            st.session_state.session_counter = 1

    def create_session(self, name: str = None) -> str:
        """Create a new session"""
        session_id = f"session_{st.session_state.session_counter}"
        st.session_state.session_counter += 1

        if name is None:
            name = f"Chat {st.session_state.session_counter - 1}"

        st.session_state.sessions[session_id] = {
            'id': session_id,
            'name': name,
            'messages': [],
            'created_at': datetime.now().isoformat(),
            'model': None,
            'metadata': {}
        }

        st.session_state.current_session_id = session_id
        return session_id

    def get_current_session(self) -> Dict:
        """Get current active session"""
        if st.session_state.current_session_id is None:
            self.create_session()
        return st.session_state.sessions[st.session_state.current_session_id]

    def switch_session(self, session_id: str):
        """Switch to a different session"""
        if session_id in st.session_state.sessions:
            st.session_state.current_session_id = session_id

    def delete_session(self, session_id: str):
        """Delete a session"""
        if session_id in st.session_state.sessions:
            del st.session_state.sessions[session_id]
            if st.session_state.current_session_id == session_id:
                if st.session_state.sessions:
                    st.session_state.current_session_id = list(st.session_state.sessions.keys())[0]
                else:
                    self.create_session()

    def rename_session(self, session_id: str, new_name: str):
        """Rename a session"""
        if session_id in st.session_state.sessions:
            st.session_state.sessions[session_id]['name'] = new_name

    def get_all_sessions(self) -> List[Dict]:
        """Get all sessions"""
        return list(st.session_state.sessions.values())

# ==================== Helper Functions ====================

def check_backend():
    """Check if backend is running"""
    try:
        response = requests.get(f"{API_BASE}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def get_providers():
    """Get list of LLM providers"""
    response = requests.get(f"{API_BASE}/api/llm/providers")
    return response.json()

def add_provider(name, provider_type, model, api_key=None, base_url=None, max_tokens=256, temperature=0.7):
    """Add a new LLM provider"""
    data = {
        "name": name,
        "provider": provider_type,
        "model": model,
        "api_key": api_key,
        "base_url": base_url,
        "timeout": 30,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    response = requests.post(f"{API_BASE}/api/llm/provider", json=data)
    return response.json()

def switch_provider(name):
    """Switch active LLM provider"""
    response = requests.post(f"{API_BASE}/api/llm/switch", json={"provider_name": name})
    return response.json()

def send_message(message):
    """Send message to Dheera"""
    response = requests.post(f"{API_BASE}/api/chat", json={"message": message})
    return response.json()

def get_stats():
    """Get Dheera statistics"""
    response = requests.get(f"{API_BASE}/api/stats")
    return response.json()

def test_provider(name):
    """Test a provider"""
    response = requests.get(f"{API_BASE}/api/llm/test/{name}")
    return response.json()

# ==================== Page Config ====================

st.set_page_config(
    page_title="Dheera - Brain-Inspired AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== Initialize Session Manager ====================

session_mgr = SessionManager()

# ==================== Sidebar (Enhanced Like LiteLLM) ====================

with st.sidebar:
    # Header
    st.markdown("# 🧠 **Dheera**")
    st.caption("Brain-Inspired AI Assistant")

    # Backend status indicator
    backend_running = check_backend()
    if backend_running:
        st.success("● Backend Online", icon="✅")
    else:
        st.error("● Backend Offline", icon="❌")
        st.caption("Run: `python3 api/server.py`")

    st.divider()

    # ==================== Session Management (ChatGPT/Claude Style) ====================

    st.markdown("### 💬 **Conversations**")

    # New chat button
    col1, col2 = st.columns([3, 1])
    with col1:
        if st.button("➕ New Chat", use_container_width=True, type="primary"):
            session_mgr.create_session()
            st.rerun()
    with col2:
        show_settings = st.button("⚙️", use_container_width=True)

    st.markdown("")  # Spacing

    # Session list
    sessions = session_mgr.get_all_sessions()
    current_session = session_mgr.get_current_session()

    # Sort sessions by creation time (newest first)
    sessions_sorted = sorted(sessions, key=lambda x: x['created_at'], reverse=True)

    for session in sessions_sorted:
        is_current = session['id'] == current_session['id']

        col1, col2 = st.columns([4, 1])

        with col1:
            # Session button
            button_type = "primary" if is_current else "secondary"
            label = f"{'📝' if is_current else '💬'} {session['name']}"

            if st.button(label, key=f"sess_{session['id']}", use_container_width=True, type=button_type):
                session_mgr.switch_session(session['id'])
                st.rerun()

        with col2:
            # Delete button
            if st.button("🗑️", key=f"del_{session['id']}", use_container_width=True):
                if len(sessions) > 1:  # Keep at least one session
                    session_mgr.delete_session(session['id'])
                    st.rerun()
                else:
                    st.warning("Can't delete the last session!")

    st.divider()

    # ==================== Quick Model Selector (LiteLLM Style) ====================

    st.markdown("### 🤖 **Active Model**")

    if backend_running:
        try:
            providers_data = get_providers()
            providers = providers_data.get("providers", [])
            active = providers_data.get("active", None)

            if providers:
                provider_names = [p["name"] for p in providers]
                current_idx = provider_names.index(active) if active in provider_names else 0

                selected = st.selectbox(
                    "Select Model",
                    provider_names,
                    index=current_idx,
                    label_visibility="collapsed"
                )

                if selected != active:
                    result = switch_provider(selected)
                    if result.get("success"):
                        st.success(f"Switched to {selected}")
                        st.rerun()

                # Show model info
                selected_provider = next((p for p in providers if p["name"] == selected), None)
                if selected_provider:
                    st.caption(f"**Provider:** {selected_provider['provider']}")
                    st.caption(f"**Model:** {selected_provider['model']}")
                    st.caption(f"**Status:** {'🟢 Available' if selected_provider.get('available') else '🔴 Unavailable'}")
            else:
                st.info("No providers configured")
        except:
            st.warning("Can't load providers")

    st.divider()

    # ==================== Advanced Options (LiteLLM Style) ====================

    with st.expander("⚙️ **Advanced Settings**"):
        st.markdown("##### Generation Parameters")

        temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
        max_tokens = st.slider("Max Tokens", 50, 2048, 256, 50)

        st.markdown("##### Features")

        show_metadata = st.checkbox("Show metadata", value=True)
        auto_scroll = st.checkbox("Auto-scroll", value=True)
        sound_notifications = st.checkbox("Sound notifications", value=False)

        # Save to session state
        st.session_state.temperature = temperature
        st.session_state.max_tokens = max_tokens
        st.session_state.show_metadata = show_metadata

    st.divider()

    # ==================== Quick Stats ====================

    with st.expander("📊 **Quick Stats**"):
        if backend_running:
            try:
                stats = get_stats()
                dheera_stats = stats.get("dheera", {})

                st.metric("Total Turns", dheera_stats.get("conversation_turns", 0))
                st.metric("DQN Steps", dheera_stats.get("dqn", {}).get("total_steps", 0))
                st.metric("RAG Docs", dheera_stats.get("rag", {}).get("total_documents", 0))
            except:
                st.caption("Stats unavailable")

    st.divider()

    # ==================== Navigation ====================

    st.markdown("### 📌 **Navigation**")
    page = st.radio(
        "Go to",
        ["💬 Chat", "🔧 Models", "📊 Analytics", "⚙️ Settings"],
        label_visibility="collapsed"
    )

# ==================== Main Content Area ====================

# ==================== Chat Page (ChatGPT/Claude Style) ====================

if page == "💬 Chat":
    # Header with session name
    col1, col2, col3 = st.columns([3, 2, 1])

    with col1:
        st.title(f"💬 {current_session['name']}")

    with col2:
        # Rename session
        new_name = st.text_input("Rename", current_session['name'], label_visibility="collapsed", key="rename_input")
        if new_name != current_session['name']:
            session_mgr.rename_session(current_session['id'], new_name)

    with col3:
        if st.button("🗑️ Clear Chat"):
            current_session['messages'] = []
            st.rerun()

    if not backend_running:
        st.error("❌ Backend is not running. Please start the server first.")
        st.code("python3 api/server.py", language="bash")
        st.stop()

    # Display chat messages
    chat_container = st.container()

    with chat_container:
        for message in current_session['messages']:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

                # Show metadata if enabled
                if message["role"] == "assistant" and "metadata" in message and st.session_state.get("show_metadata", True):
                    with st.expander("📊 Details", expanded=False):
                        metadata = message["metadata"]

                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Latency", f"{metadata.get('latency_ms', 0):.0f}ms")
                        with col2:
                            st.metric("Tokens", metadata.get('tokens_used', 0))
                        with col3:
                            st.metric("Reward", f"{metadata.get('reward', 0):.3f}")
                        with col4:
                            st.metric("Action", metadata.get('action_name', 'N/A')[:15])

                        # Show full metadata
                        with st.expander("Full Metadata"):
                            st.json(metadata)

    # Chat input (sticky at bottom)
    if prompt := st.chat_input("Type your message...", key="chat_input"):
        # Add user message
        current_session['messages'].append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = send_message(prompt)
                    response = result.get("response", "Error: No response")
                    metadata = result.get("metadata", {})

                    st.markdown(response)

                    # Add to session
                    current_session['messages'].append({
                        "role": "assistant",
                        "content": response,
                        "metadata": metadata
                    })

                    # Show metadata
                    if st.session_state.get("show_metadata", True):
                        with st.expander("📊 Details", expanded=False):
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Latency", f"{metadata.get('latency_ms', 0):.0f}ms")
                            with col2:
                                st.metric("Tokens", metadata.get('tokens_used', 0))
                            with col3:
                                st.metric("Reward", f"{metadata.get('reward', 0):.3f}")
                            with col4:
                                st.metric("Action", metadata.get('action_name', 'N/A')[:15])

                    st.rerun()  # Refresh to show new message

                except Exception as e:
                    st.error(f"Error: {e}")

# ==================== Models Page (LiteLLM Style) ====================

elif page == "🔧 Models":
    st.title("🔧 Model Management")

    if not backend_running:
        st.error("❌ Backend is not running.")
        st.stop()

    tab1, tab2, tab3, tab4 = st.tabs(["📋 Available Models", "➕ Add Model", "🔬 Test Models", "🎨 Presets"])

    with tab1:
        col_header1, col_header2 = st.columns([3, 1])
        with col_header1:
            st.subheader("Available Models")
        with col_header2:
            if st.button("🔄 Discover Ollama Models", help="Auto-discover all Ollama models"):
                try:
                    response = requests.post(f"{API_BASE}/api/llm/discover")
                    result = response.json()
                    if result.get("success"):
                        st.success(f"✅ Discovered {result.get('discovered', 0)} models, added {result.get('added', 0)} new")
                        st.rerun()
                    else:
                        st.error(f"❌ {result.get('error', 'Failed to discover models')}")
                except Exception as e:
                    st.error(f"Error: {e}")

        try:
            providers_data = get_providers()
            providers = providers_data.get("providers", [])
            active = providers_data.get("active", None)

            if providers:
                for provider in providers:
                    with st.container():
                        col1, col2, col3, col4, col5 = st.columns([1, 2, 2, 2, 1])

                        with col1:
                            is_active = provider["name"] == active
                            st.markdown(f"### {'🟢' if is_active else '⚪'}")

                        with col2:
                            st.markdown(f"**{provider['name']}**")
                            if is_active:
                                st.caption("🟢 Active")

                        with col3:
                            st.caption(f"Provider: {provider['provider']}")
                            st.caption(f"Model: {provider['model']}")

                        with col4:
                            stats = provider.get('stats', {})
                            st.caption(f"Requests: {stats.get('requests', 0)}")
                            st.caption(f"Tokens: {stats.get('total_tokens', 0)}")

                        with col5:
                            if not is_active:
                                if st.button("Activate", key=f"activate_{provider['name']}"):
                                    result = switch_provider(provider['name'])
                                    if result.get("success"):
                                        st.success(f"Activated {provider['name']}")
                                        st.rerun()
                            else:
                                st.success("✓ Active")

                        st.divider()
            else:
                st.info("No models configured. Add one in the 'Add Model' tab.")

        except Exception as e:
            st.error(f"Error: {e}")

    with tab2:
        st.subheader("Add New Model")

        col1, col2 = st.columns(2)

        with col1:
            provider_type = st.selectbox(
                "Provider Type",
                ["ollama", "openai", "anthropic", "litellm"],
                help="Choose your LLM provider"
            )

        with col2:
            name = st.text_input(
                "Model Name",
                placeholder="e.g., my_gpt4",
                help="Unique name for this model configuration"
            )

        if provider_type == "ollama":
            model = st.selectbox(
                "Ollama Model",
                ["phi3:mini", "gemma:2b", "llama3:8b", "qwen2:1.5b", "tinyllama", "mistral", "codellama"]
            )
            base_url = st.text_input("Ollama URL", value="http://localhost:11434")
            api_key = None

        elif provider_type == "openai":
            model = st.selectbox(
                "OpenAI Model",
                ["gpt-4-turbo-preview", "gpt-4", "gpt-3.5-turbo", "gpt-3.5-turbo-16k"]
            )
            api_key = st.text_input("OpenAI API Key", type="password", help="Get from platform.openai.com")
            base_url = None

        elif provider_type == "anthropic":
            model = st.selectbox(
                "Anthropic Model",
                ["claude-opus-4-20250514", "claude-sonnet-4-20250514", "claude-3-opus-20240229", "claude-3-sonnet-20240229"]
            )
            api_key = st.text_input("Anthropic API Key", type="password", help="Get from console.anthropic.com")
            base_url = None

        else:  # litellm
            model = st.text_input("Model ID", placeholder="e.g., groq/llama3-70b, together_ai/...", help="LiteLLM model identifier")
            api_key = st.text_input("API Key (if needed)", type="password")
            base_url = st.text_input("Base URL (optional)")

        # Advanced settings
        with st.expander("⚙️ Advanced Settings"):
            max_tokens = st.slider("Max Tokens", 50, 4096, 256)
            temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)

        if st.button("➕ Add Model", type="primary", use_container_width=True):
            if not name:
                st.error("Please enter a model name")
            elif not model:
                st.error("Please select/enter a model")
            else:
                try:
                    result = add_provider(
                        name=name,
                        provider_type=provider_type,
                        model=model,
                        api_key=api_key if api_key else None,
                        base_url=base_url if base_url else None,
                        max_tokens=max_tokens,
                        temperature=temperature
                    )
                    if result.get("success"):
                        st.success(f"✅ Added model: {name}")
                        st.balloons()
                        st.rerun()
                    else:
                        st.error("Failed to add model")
                except Exception as e:
                    st.error(f"Error: {e}")

    with tab3:
        st.subheader("Test Models")

        try:
            providers_data = get_providers()
            providers = providers_data.get("providers", [])

            if providers:
                col1, col2 = st.columns(2)

                with col1:
                    provider_names = [p["name"] for p in providers]
                    selected = st.selectbox("Select Model to Test", provider_names)

                with col2:
                    if st.button("🔬 Run Test", use_container_width=True, type="primary"):
                        with st.spinner("Testing..."):
                            result = test_provider(selected)

                            if result.get("success"):
                                st.success("✅ Model is working!")

                                col_a, col_b, col_c = st.columns(3)
                                with col_a:
                                    st.metric("Latency", f"{result.get('latency_ms', 0):.0f}ms")
                                with col_b:
                                    st.metric("Available", "Yes" if result.get("available") else "No")
                                with col_c:
                                    st.metric("Status", "✅ Pass")

                                st.code(result.get("response", ""), language="text")
                            else:
                                st.error("❌ Test failed")
                                st.error(result.get("error", "Unknown error"))
            else:
                st.info("No models to test")
        except Exception as e:
            st.error(f"Error: {e}")

    with tab4:
        st.subheader("🎨 Model Presets")

        st.markdown("""
        Quick-add popular model configurations:
        """)

        presets = {
            "🚀 Fastest (Qwen 1.5B)": {
                "provider": "ollama",
                "model": "qwen2:1.5b",
                "name": "qwen_fast"
            },
            "⚡ Fast (Gemma 2B)": {
                "provider": "ollama",
                "model": "gemma:2b",
                "name": "gemma_2b"
            },
            "🎯 Balanced (Phi3 Mini)": {
                "provider": "ollama",
                "model": "phi3:mini",
                "name": "phi3_mini"
            },
            "🧠 Smart (Llama3 8B)": {
                "provider": "ollama",
                "model": "llama3:8b",
                "name": "llama3_8b"
            },
            "☁️ GPT-4 Turbo": {
                "provider": "openai",
                "model": "gpt-4-turbo-preview",
                "name": "gpt4_turbo"
            },
            "☁️ GPT-3.5 Turbo": {
                "provider": "openai",
                "model": "gpt-3.5-turbo",
                "name": "gpt35_turbo"
            },
            "☁️ Claude Opus": {
                "provider": "anthropic",
                "model": "claude-opus-4-20250514",
                "name": "claude_opus"
            },
        }

        for preset_name, preset_config in presets.items():
            col1, col2 = st.columns([3, 1])

            with col1:
                st.markdown(f"**{preset_name}**")
                st.caption(f"{preset_config['provider']} - {preset_config['model']}")

            with col2:
                if st.button("Add", key=f"preset_{preset_config['name']}"):
                    try:
                        result = add_provider(
                            name=preset_config['name'],
                            provider_type=preset_config['provider'],
                            model=preset_config['model']
                        )
                        if result.get("success"):
                            st.success(f"Added {preset_name}")
                            st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")

# ==================== Analytics Page ====================

elif page == "📊 Analytics":
    st.title("📊 Analytics Dashboard")

    if not backend_running:
        st.error("❌ Backend is not running.")
        st.stop()

    try:
        stats = get_stats()
        dheera_stats = stats.get("dheera", {})
        llm_stats = stats.get("llm_router", {})

        # Key metrics
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric("Sessions", len(session_mgr.get_all_sessions()))
        with col2:
            st.metric("Turns", dheera_stats.get("conversation_turns", 0))
        with col3:
            st.metric("DQN Steps", dheera_stats.get("dqn", {}).get("total_steps", 0))
        with col4:
            st.metric("RAG Docs", dheera_stats.get("rag", {}).get("total_documents", 0))
        with col5:
            st.metric("LLM Requests", llm_stats.get("total_requests", 0))

        st.divider()

        # Provider comparison
        st.subheader("🔌 Provider Performance")

        providers = llm_stats.get("providers", [])

        if providers and HAS_PLOTLY:
            provider_data = []
            for p in providers:
                stats_data = p.get("stats", {})
                provider_data.append({
                    "Provider": p["name"],
                    "Requests": stats_data.get("requests", 0),
                    "Tokens": stats_data.get("total_tokens", 0),
                    "Avg Latency (ms)": stats_data.get("total_latency_ms", 0) / max(stats_data.get("requests", 1), 1),
                    "Errors": stats_data.get("errors", 0),
                })

            col1, col2 = st.columns(2)

            with col1:
                fig1 = px.bar(provider_data, x="Provider", y="Requests", title="Requests per Provider")
                st.plotly_chart(fig1, use_container_width=True)

            with col2:
                fig2 = px.bar(provider_data, x="Provider", y="Avg Latency (ms)", title="Avg Latency per Provider")
                st.plotly_chart(fig2, use_container_width=True)

        elif providers:
            st.dataframe(provider_data, use_container_width=True)
        else:
            st.info("No provider data available")

        st.divider()

        # System stats
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🧠 DQN Agent")
            st.json(dheera_stats.get("dqn", {}))

        with col2:
            st.subheader("💾 RAG System")
            st.json(dheera_stats.get("rag", {}))

    except Exception as e:
        st.error(f"Error loading analytics: {e}")

# ==================== Settings Page ====================

elif page == "⚙️ Settings":
    st.title("⚙️ Settings")

    tab1, tab2, tab3 = st.tabs(["🎨 Appearance", "🔧 Configuration", "ℹ️ About"])

    with tab1:
        st.subheader("Appearance Settings")

        theme = st.selectbox("Theme", ["Light", "Dark", "Auto"])
        font_size = st.slider("Font Size", 12, 20, 14)
        compact_mode = st.checkbox("Compact Mode")

        st.info("💡 Some settings require page refresh")

    with tab2:
        st.subheader("Dheera Configuration")

        if backend_running:
            try:
                config_data = requests.get(f"{API_BASE}/api/config").json()

                st.markdown("**Current Configuration:**")
                st.json(config_data.get("config", {}))

                st.markdown("**Identity:**")
                st.json(config_data.get("identity", {}))
            except:
                st.error("Can't load configuration")

    with tab3:
        st.subheader("About Dheera")

        st.markdown("""
        ### 🧠 Dheera v0.3.1
        **Brain-Inspired AI with Hot-Swappable LLM Backends**

        **Features:**
        - 💬 ChatGPT/Claude-style session management
        - 🔄 Hot-swappable LLM providers
        - 🧠 Rainbow DQN agent with curiosity-driven learning
        - 💾 RAG (Retrieval-Augmented Generation)
        - 🎯 RLHF (Reinforcement Learning from Human Feedback)
        - ⚡ SpikingBrain-inspired neural networks
        - 📊 Real-time monitoring

        **Supported Providers:**
        - 🏠 Ollama (phi3, gemma, llama3, qwen2, etc.)
        - ☁️ OpenAI (GPT-4, GPT-3.5)
        - ☁️ Anthropic (Claude Opus, Sonnet)
        - 🌐 LiteLLM (100+ models)

        **Architecture:**
        - Frontend: Streamlit
        - Backend: FastAPI + WebSocket
        - AI Engine: PyTorch + Transformers

        ---

        Created with ❤️ using Claude Code
        """)

        st.divider()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("[📖 Docs](GUI_README.md)")
        with col2:
            st.markdown("[🐛 Issues](https://github.com)")
        with col3:
            st.markdown("[⭐ Star](https://github.com)")

# ==================== Footer ====================

st.divider()
st.caption("🧠 Dheera v0.3.1 - Brain-Inspired AI | Enhanced GUI with Session Management")
