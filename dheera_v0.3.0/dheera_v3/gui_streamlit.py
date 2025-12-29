#!/usr/bin/env python3
"""
Dheera Streamlit GUI
Modern web interface with hot-swappable LLM backends
"""

import streamlit as st
import requests
import json
from datetime import datetime
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Try importing plotly for charts
try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# ==================== Configuration ====================

API_BASE = "http://localhost:8000"

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


def add_provider(name, provider_type, model, api_key=None, base_url=None):
    """Add a new LLM provider"""
    data = {
        "name": name,
        "provider": provider_type,
        "model": model,
        "api_key": api_key,
        "base_url": base_url,
        "timeout": 30,
        "max_tokens": 256,
        "temperature": 0.7,
    }
    response = requests.post(f"{API_BASE}/api/llm/provider", json=data)
    return response.json()


def switch_provider(name):
    """Switch active LLM provider"""
    response = requests.post(f"{API_BASE}/api/llm/switch", json={"provider_name": name})
    return response.json()


def send_message(message):
    """Send message to Dheera"""
    response = requests.post(
        f"{API_BASE}/api/chat",
        json={"message": message}
    )
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

# ==================== Sidebar ====================

with st.sidebar:
    st.title("🧠 Dheera")
    st.caption("Brain-Inspired AI with Hot-Swappable LLMs")

    st.divider()

    # Backend status
    backend_running = check_backend()
    if backend_running:
        st.success("✅ Backend Online")
    else:
        st.error("❌ Backend Offline")
        st.caption("Start with: `python3 api/server.py`")

    st.divider()

    # Page selection
    page = st.radio(
        "Navigation",
        ["💬 Chat", "🔧 LLM Settings", "📊 Monitoring"],
        label_visibility="collapsed"
    )

    st.divider()

    # Quick stats
    if backend_running:
        try:
            stats = get_stats()
            dheera_stats = stats.get("dheera", {})

            st.metric("Conversation Turns", dheera_stats.get("conversation_turns", 0))
            st.metric("DQN Steps", dheera_stats.get("dqn", {}).get("total_steps", 0))

            llm_stats = stats.get("llm_router", {})
            active = llm_stats.get("active_provider", "None")
            st.metric("Active Provider", active if active else "None")

        except Exception as e:
            st.warning(f"Stats unavailable: {str(e)[:50]}")

# ==================== Chat Page ====================

if page == "💬 Chat":
    st.header("💬 Chat with Dheera")

    if not backend_running:
        st.error("❌ Backend is not running. Please start the server first.")
        st.code("python3 api/server.py", language="bash")
        st.stop()

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "metadata" in message:
                with st.expander("📊 Metadata"):
                    col1, col2, col3 = st.columns(3)
                    metadata = message["metadata"]
                    with col1:
                        st.metric("Latency", f"{metadata.get('latency_ms', 0):.0f}ms")
                    with col2:
                        st.metric("Tokens", metadata.get('tokens_used', 0))
                    with col3:
                        st.metric("Reward", f"{metadata.get('reward', 0):.3f}")

                    st.json(metadata)

    # Chat input
    if prompt := st.chat_input("Type your message..."):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get response from Dheera
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = send_message(prompt)
                    response = result.get("response", "Error: No response")
                    metadata = result.get("metadata", {})

                    st.markdown(response)

                    # Show metadata in expander
                    with st.expander("📊 Metadata"):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Latency", f"{metadata.get('latency_ms', 0):.0f}ms")
                        with col2:
                            st.metric("Tokens", metadata.get('tokens_used', 0))
                        with col3:
                            st.metric("Reward", f"{metadata.get('reward', 0):.3f}")

                        st.json(metadata)

                    # Add assistant message to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "metadata": metadata
                    })

                except Exception as e:
                    st.error(f"Error: {e}")

    # Clear chat button
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# ==================== LLM Settings Page ====================

elif page == "🔧 LLM Settings":
    st.header("🔧 LLM Provider Settings")

    if not backend_running:
        st.error("❌ Backend is not running.")
        st.stop()

    tab1, tab2, tab3 = st.tabs(["📋 Current Providers", "➕ Add Provider", "🔬 Test"])

    with tab1:
        st.subheader("Current Providers")

        try:
            providers_data = get_providers()
            providers = providers_data.get("providers", [])
            active = providers_data.get("active", None)

            if not providers:
                st.info("No providers configured. Add one in the 'Add Provider' tab.")
            else:
                for provider in providers:
                    col1, col2, col3, col4 = st.columns([2, 2, 2, 1])

                    with col1:
                        is_active = provider["name"] == active
                        icon = "✅" if is_active else "⭕"
                        st.write(f"{icon} **{provider['name']}**")

                    with col2:
                        st.write(f"{provider['provider']}")

                    with col3:
                        st.write(f"{provider['model']}")

                    with col4:
                        if not is_active:
                            if st.button("Switch", key=f"switch_{provider['name']}"):
                                result = switch_provider(provider['name'])
                                if result.get("success"):
                                    st.success(f"✅ Switched to {provider['name']}")
                                    st.rerun()

                    st.divider()

        except Exception as e:
            st.error(f"Error loading providers: {e}")

    with tab2:
        st.subheader("Add New Provider")

        provider_type = st.selectbox(
            "Provider Type",
            ["ollama", "openai", "anthropic", "litellm"]
        )

        name = st.text_input("Provider Name", placeholder="e.g., my_gpt4")

        if provider_type == "ollama":
            model = st.selectbox(
                "Model",
                ["phi3:mini", "gemma:2b", "llama3:8b", "qwen2:1.5b", "tinyllama"]
            )
            base_url = st.text_input("Base URL", value="http://localhost:11434")
            api_key = None

        elif provider_type == "openai":
            model = st.selectbox(
                "Model",
                ["gpt-4-turbo-preview", "gpt-3.5-turbo", "gpt-4"]
            )
            api_key = st.text_input("API Key", type="password")
            base_url = None

        elif provider_type == "anthropic":
            model = st.selectbox(
                "Model",
                ["claude-opus-4-20250514", "claude-sonnet-4-20250514"]
            )
            api_key = st.text_input("API Key", type="password")
            base_url = None

        else:  # litellm
            model = st.text_input("Model", placeholder="e.g., gpt-3.5-turbo, claude-2")
            api_key = st.text_input("API Key (optional)", type="password")
            base_url = st.text_input("Base URL (optional)")

        if st.button("➕ Add Provider"):
            if not name:
                st.error("Please enter a provider name")
            elif not model:
                st.error("Please enter a model name")
            else:
                try:
                    result = add_provider(
                        name=name,
                        provider_type=provider_type,
                        model=model,
                        api_key=api_key if api_key else None,
                        base_url=base_url if base_url else None
                    )
                    if result.get("success"):
                        st.success(f"✅ Added provider: {name}")
                        st.rerun()
                    else:
                        st.error(f"Failed to add provider")
                except Exception as e:
                    st.error(f"Error: {e}")

    with tab3:
        st.subheader("Test Providers")

        try:
            providers_data = get_providers()
            providers = providers_data.get("providers", [])

            if providers:
                provider_names = [p["name"] for p in providers]
                selected = st.selectbox("Select Provider to Test", provider_names)

                if st.button("🔬 Test"):
                    with st.spinner("Testing..."):
                        result = test_provider(selected)

                        if result.get("success"):
                            st.success("✅ Provider is working!")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Latency", f"{result.get('latency_ms', 0):.0f}ms")
                            with col2:
                                st.metric("Available", "Yes" if result.get("available") else "No")
                            st.code(result.get("response", ""))
                        else:
                            st.error(f"❌ Test failed: {result.get('error', 'Unknown error')}")
            else:
                st.info("No providers to test")
        except Exception as e:
            st.error(f"Error: {e}")

# ==================== Monitoring Page ====================

elif page == "📊 Monitoring":
    st.header("📊 System Monitoring")

    if not backend_running:
        st.error("❌ Backend is not running.")
        st.stop()

    try:
        stats = get_stats()
        dheera_stats = stats.get("dheera", {})
        llm_stats = stats.get("llm_router", {})

        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Turns", dheera_stats.get("conversation_turns", 0))
        with col2:
            st.metric("DQN Steps", dheera_stats.get("dqn", {}).get("total_steps", 0))
        with col3:
            st.metric("RAG Docs", dheera_stats.get("rag", {}).get("total_documents", 0))
        with col4:
            st.metric("LLM Requests", llm_stats.get("total_requests", 0))

        st.divider()

        # DQN Stats
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🧠 DQN Agent")
            dqn = dheera_stats.get("dqn", {})
            st.json(dqn)

        with col2:
            st.subheader("💾 RAG System")
            rag = dheera_stats.get("rag", {})
            st.json(rag)

        st.divider()

        # LLM Provider Stats
        st.subheader("🔌 LLM Provider Statistics")
        providers = llm_stats.get("providers", [])

        if providers:
            provider_df = []
            for p in providers:
                stats_data = p.get("stats", {})
                provider_df.append({
                    "Provider": p["name"],
                    "Requests": stats_data.get("requests", 0),
                    "Tokens": stats_data.get("total_tokens", 0),
                    "Avg Latency (ms)": stats_data.get("total_latency_ms", 0) / max(stats_data.get("requests", 1), 1),
                    "Errors": stats_data.get("errors", 0),
                })

            st.dataframe(provider_df, use_container_width=True)

            # Bar chart of requests per provider (if plotly available)
            if HAS_PLOTLY and len(provider_df) > 0:
                fig = px.bar(
                    provider_df,
                    x="Provider",
                    y="Requests",
                    title="Requests per Provider"
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No provider statistics available")

    except Exception as e:
        st.error(f"Error loading stats: {e}")

# ==================== Footer ====================

st.divider()
st.caption("🧠 Dheera v0.3.1 - Brain-Inspired AI | Built with FastAPI + Streamlit")
