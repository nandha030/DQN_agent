#!/usr/bin/env python3
"""
Dheera Professional GUI - Complete ChatGPT/Claude/LiteLLM Experience
Features all 3 phases:
- Phase 1: Auto-naming, previews, timestamps, search, export
- Phase 2: Message actions, theme, token counter, keyboard shortcuts
- Phase 3: Folders, model comparison, templates
"""

import streamlit as st
import requests
import json
import re
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import sys
import os
import platform
import subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# ==================== Configuration ====================

API_BASE = "http://localhost:8000"

# ==================== Page Config ====================

st.set_page_config(
    page_title="Dheera - Professional AI Assistant",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==================== Utility Functions ====================

def get_relative_time(timestamp_str: str) -> str:
    """Convert ISO timestamp to relative time (e.g., '2 hours ago')"""
    try:
        timestamp = datetime.fromisoformat(timestamp_str)
        now = datetime.now()
        diff = now - timestamp

        if diff.days > 7:
            return timestamp.strftime("%b %d")
        elif diff.days > 0:
            return f"{diff.days}d ago" if diff.days > 1 else "Yesterday"
        elif diff.seconds > 3600:
            hours = diff.seconds // 3600
            return f"{hours}h ago"
        elif diff.seconds > 60:
            mins = diff.seconds // 60
            return f"{mins}m ago"
        else:
            return "Just now"
    except:
        return "Unknown"


def format_timestamp(timestamp_str: str, include_time: bool = True) -> str:
    """Format timestamp for display"""
    try:
        timestamp = datetime.fromisoformat(timestamp_str)
        if include_time:
            return timestamp.strftime("%I:%M %p")
        else:
            return timestamp.strftime("%b %d, %Y")
    except:
        return ""


def auto_generate_session_name(message: str, use_llm: bool = False) -> str:
    """
    Auto-generate session name from first message

    Strategies:
    1. Simple: Extract first sentence or first 50 chars
    2. LLM: Ask model to summarize (if enabled)
    """
    if not message:
        return "New Chat"

    # Remove extra whitespace
    message = ' '.join(message.split())

    # Strategy 1: Extract first sentence
    sentences = re.split(r'[.!?]', message)
    first_sentence = sentences[0].strip()

    if len(first_sentence) <= 50 and len(first_sentence) > 5:
        return first_sentence

    # Strategy 2: First 50 chars
    if len(message) <= 50:
        return message

    # Truncate intelligently at word boundary
    truncated = message[:50]
    last_space = truncated.rfind(' ')
    if last_space > 30:  # Only truncate at space if it's not too early
        truncated = truncated[:last_space]

    return truncated + "..."


def estimate_tokens(text: str) -> int:
    """Rough token estimation (words * 1.3)"""
    if not text:
        return 0
    return int(len(text.split()) * 1.3)


def export_to_markdown(session: Dict) -> str:
    """Export session to Markdown format"""
    md = f"# {session['name']}\n\n"
    md += f"*Created: {format_timestamp(session['created_at'], False)}*\n\n"
    md += "---\n\n"

    for msg in session.get('messages', []):
        role = msg['role'].title()
        content = msg['content']
        timestamp = format_timestamp(msg.get('timestamp', ''))

        md += f"### {role}"
        if timestamp:
            md += f" • {timestamp}"
        md += "\n\n"
        md += f"{content}\n\n"

        if msg['role'] == 'assistant' and 'metadata' in msg:
            meta = msg['metadata']
            md += f"*Latency: {meta.get('latency_ms', 0):.0f}ms, "
            md += f"Tokens: {meta.get('tokens_used', 0)}, "
            md += f"Model: {meta.get('slm_model', 'unknown')}*\n\n"

        md += "---\n\n"

    return md


def export_to_json(session: Dict) -> str:
    """Export session to JSON format"""
    return json.dumps(session, indent=2)


def export_to_text(session: Dict) -> str:
    """Export session to plain text"""
    txt = f"{session['name']}\n"
    txt += f"Created: {format_timestamp(session['created_at'], False)}\n"
    txt += "=" * 60 + "\n\n"

    for msg in session.get('messages', []):
        role = msg['role'].title()
        content = msg['content']
        timestamp = format_timestamp(msg.get('timestamp', ''))

        txt += f"{role}"
        if timestamp:
            txt += f" ({timestamp})"
        txt += f":\n{content}\n\n"

    return txt


# ==================== Session Management ====================

class SessionManager:
    """Enhanced session manager with folders and tags"""

    def __init__(self):
        # Initialize session state
        if 'sessions' not in st.session_state:
            st.session_state.sessions = {}
        if 'current_session_id' not in st.session_state:
            st.session_state.current_session_id = None
        if 'session_counter' not in st.session_state:
            st.session_state.session_counter = 1
        if 'folders' not in st.session_state:
            st.session_state.folders = ['Work', 'Personal', 'Learning', 'Research']
        if 'pinned_sessions' not in st.session_state:
            st.session_state.pinned_sessions = set()

    def create_session(self, name: str = None, folder: str = None) -> str:
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
            'updated_at': datetime.now().isoformat(),
            'model': None,
            'metadata': {
                'total_tokens': 0,
                'message_count': 0,
                'tags': [],
                'folder': folder or 'Personal',
                'pinned': False,
                'archived': False,
            }
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
            st.session_state.sessions[session_id]['updated_at'] = datetime.now().isoformat()

    def pin_session(self, session_id: str):
        """Pin/unpin a session"""
        if session_id in st.session_state.sessions:
            is_pinned = st.session_state.sessions[session_id]['metadata']['pinned']
            st.session_state.sessions[session_id]['metadata']['pinned'] = not is_pinned

    def move_to_folder(self, session_id: str, folder: str):
        """Move session to a folder"""
        if session_id in st.session_state.sessions:
            st.session_state.sessions[session_id]['metadata']['folder'] = folder

    def add_tag(self, session_id: str, tag: str):
        """Add tag to session"""
        if session_id in st.session_state.sessions:
            tags = st.session_state.sessions[session_id]['metadata']['tags']
            if tag not in tags:
                tags.append(tag)

    def get_all_sessions(self, folder: str = None, pinned_only: bool = False) -> List[Dict]:
        """Get all sessions, optionally filtered"""
        sessions = list(st.session_state.sessions.values())

        if pinned_only:
            sessions = [s for s in sessions if s['metadata']['pinned']]
        elif folder:
            sessions = [s for s in sessions if s['metadata']['folder'] == folder]

        # Sort by updated_at (most recent first)
        sessions.sort(key=lambda x: x.get('updated_at', x['created_at']), reverse=True)
        return sessions

    def search_sessions(self, query: str) -> List[Dict]:
        """Search sessions by name and content"""
        query = query.lower()
        results = []

        for session in st.session_state.sessions.values():
            # Search in name
            if query in session['name'].lower():
                results.append(session)
                continue

            # Search in messages
            for msg in session.get('messages', []):
                if query in msg['content'].lower():
                    results.append(session)
                    break

        # Sort by updated_at
        results.sort(key=lambda x: x.get('updated_at', x['created_at']), reverse=True)
        return results


# ==================== API Functions ====================

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


def switch_provider(name):
    """Switch active LLM provider"""
    response = requests.post(f"{API_BASE}/api/llm/switch", json={"provider_name": name})
    return response.json()


def send_message(message, temperature=0.7, max_tokens=256):
    """Send message to Dheera"""
    response = requests.post(f"{API_BASE}/api/chat", json={"message": message})
    return response.json()


def get_stats():
    """Get statistics"""
    response = requests.get(f"{API_BASE}/api/stats")
    return response.json()


def detect_system_theme() -> bool:
    """
    Detect OS theme preference.
    Returns True for dark mode, False for light mode.
    """
    try:
        system_os = platform.system()

        if system_os == "Darwin":  # macOS
            # Check if Dark Mode is enabled
            result = subprocess.run(
                ['defaults', 'read', '-g', 'AppleInterfaceStyle'],
                capture_output=True,
                text=True,
                timeout=2
            )
            # If return code is 0, Dark Mode is ON
            # If return code is 1, Dark Mode is OFF (Light Mode)
            return result.returncode == 0

        elif system_os == "Windows":
            # Check Windows registry for theme
            try:
                import winreg
                registry = winreg.ConnectRegistry(None, winreg.HKEY_CURRENT_USER)
                key = winreg.OpenKey(registry, r'SOFTWARE\Microsoft\Windows\CurrentVersion\Themes\Personalize')
                value, _ = winreg.QueryValueEx(key, 'AppsUseLightTheme')
                winreg.CloseKey(key)
                # 0 = Dark Mode, 1 = Light Mode
                return value == 0
            except:
                pass

        elif system_os == "Linux":
            # Try to detect GNOME/KDE theme
            try:
                result = subprocess.run(
                    ['gsettings', 'get', 'org.gnome.desktop.interface', 'gtk-theme'],
                    capture_output=True,
                    text=True,
                    timeout=2
                )
                theme = result.stdout.lower()
                return 'dark' in theme
            except:
                pass

    except Exception as e:
        print(f"Could not detect system theme: {e}")

    # Default to dark mode if detection fails
    return True


# ==================== Initialize ====================

session_mgr = SessionManager()
backend_running = check_backend()

# Initialize theme - detect from system
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = detect_system_theme()

if 'show_metadata' not in st.session_state:
    st.session_state.show_metadata = True

# ==================== Custom CSS ====================

def apply_custom_css():
    """Apply custom CSS for professional look with responsive design"""

    # Theme colors
    if st.session_state.dark_mode:
        theme_bg = "#0E1117"
        theme_text = "#FAFAFA"
        theme_secondary = "#262730"
        theme_border = "#30363D"
        theme_input_bg = "#1C1E26"
        theme_hover = "#2D333B"
    else:
        theme_bg = "#FFFFFF"
        theme_text = "#0E1117"
        theme_secondary = "#F0F2F6"
        theme_border = "#E0E0E0"
        theme_input_bg = "#FAFAFA"
        theme_hover = "#E8E8E8"

    st.markdown(f"""
    <style>
        /* ==================== Theme Colors ==================== */

        :root {{
            --bg-primary: {theme_bg};
            --text-primary: {theme_text};
            --bg-secondary: {theme_secondary};
            --border-color: {theme_border};
            --input-bg: {theme_input_bg};
            --hover-bg: {theme_hover};
        }}

        /* Main app background */
        .stApp {{
            background-color: {theme_bg} !important;
            color: {theme_text} !important;
        }}

        /* Sidebar styling */
        [data-testid="stSidebar"] {{
            background-color: {theme_secondary} !important;
        }}

        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {{
            color: {theme_text} !important;
        }}

        /* Headers */
        h1, h2, h3, h4, h5, h6 {{
            color: {theme_text} !important;
        }}

        /* Text and captions */
        p, span, label, .stMarkdown {{
            color: {theme_text} !important;
        }}

        /* Input fields */
        .stTextInput input, .stTextArea textarea, .stSelectbox select {{
            background-color: {theme_input_bg} !important;
            color: {theme_text} !important;
            border-color: {theme_border} !important;
        }}

        /* Buttons */
        .stButton button {{
            background-color: {theme_secondary} !important;
            color: {theme_text} !important;
            border: 1px solid {theme_border} !important;
        }}

        .stButton button:hover {{
            background-color: {theme_hover} !important;
        }}

        /* Primary buttons */
        .stButton button[kind="primary"] {{
            background-color: #1F6FEB !important;
            color: white !important;
        }}

        /* Chat messages */
        [data-testid="stChatMessage"] {{
            background-color: {theme_secondary} !important;
            border: 1px solid {theme_border} !important;
        }}

        /* ==================== Custom Styles ==================== */

        /* Session preview */
        .session-preview {{
            font-size: 0.85em;
            color: #888;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            max-width: 100%;
        }}

        /* Timestamp */
        .timestamp {{
            font-size: 0.75em;
            color: #888;
            font-weight: 500;
        }}

        /* Pinned indicator */
        .pinned {{
            color: #FFD700;
            font-size: 1.2em;
        }}

        /* Folder header */
        .folder-header {{
            font-weight: 600;
            color: #888;
            font-size: 0.85em;
            margin-top: 1rem;
            margin-bottom: 0.5rem;
            text-transform: uppercase;
        }}

        /* ==================== Responsive Design ==================== */

        /* Mobile devices (portrait) */
        @media only screen and (max-width: 600px) {{
            /* Reduce padding */
            .stApp {{
                padding: 0.5rem !important;
            }}

            /* Stack columns vertically */
            [data-testid="column"] {{
                width: 100% !important;
                flex: 100% !important;
                max-width: 100% !important;
            }}

            /* Smaller fonts */
            h1 {{ font-size: 1.5rem !important; }}
            h2 {{ font-size: 1.3rem !important; }}
            h3 {{ font-size: 1.1rem !important; }}

            /* Compact buttons */
            .stButton button {{
                padding: 0.5rem 1rem !important;
                font-size: 0.9rem !important;
            }}

            /* Hide sidebar by default on mobile */
            [data-testid="stSidebar"] {{
                min-width: 0 !important;
            }}

            /* Full-width inputs */
            .stTextInput, .stTextArea, .stSelectbox {{
                width: 100% !important;
            }}

            /* Compact chat input */
            [data-testid="stChatInput"] {{
                font-size: 0.9rem !important;
            }}

            /* Reduce session preview text */
            .session-preview {{
                font-size: 0.75em !important;
            }}
        }}

        /* Tablets (portrait) */
        @media only screen and (min-width: 601px) and (max-width: 768px) {{
            .stApp {{
                padding: 1rem !important;
            }}

            h1 {{ font-size: 1.8rem !important; }}
            h2 {{ font-size: 1.5rem !important; }}

            /* Sidebar width */
            [data-testid="stSidebar"] {{
                min-width: 250px !important;
            }}
        }}

        /* Tablets (landscape) and small desktops */
        @media only screen and (min-width: 769px) and (max-width: 1024px) {{
            [data-testid="stSidebar"] {{
                min-width: 280px !important;
            }}
        }}

        /* Large desktops */
        @media only screen and (min-width: 1025px) {{
            [data-testid="stSidebar"] {{
                min-width: 300px !important;
            }}

            /* Max width for readability */
            .main .block-container {{
                max-width: 1200px !important;
                padding: 2rem !important;
            }}
        }}

        /* ==================== Accessibility ==================== */

        /* Focus indicators */
        button:focus, input:focus, textarea:focus, select:focus {{
            outline: 2px solid #1F6FEB !important;
            outline-offset: 2px !important;
        }}

        /* High contrast for important elements */
        .stButton button[kind="primary"] {{
            font-weight: 600 !important;
        }}

        /* Ensure touch targets are large enough (mobile) */
        @media (pointer: coarse) {{
            button, a, [role="button"] {{
                min-height: 44px !important;
                min-width: 44px !important;
            }}
        }}

        /* ==================== Animation ==================== */

        /* Smooth transitions */
        * {{
            transition: background-color 0.2s ease, color 0.2s ease, border-color 0.2s ease;
        }}

        /* Hover effects */
        .stButton button:hover {{
            transform: translateY(-1px);
            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        }}

        /* ==================== Scrollbar ==================== */

        ::-webkit-scrollbar {{
            width: 8px;
            height: 8px;
        }}

        ::-webkit-scrollbar-track {{
            background: {theme_secondary};
        }}

        ::-webkit-scrollbar-thumb {{
            background: {theme_border};
            border-radius: 4px;
        }}

        ::-webkit-scrollbar-thumb:hover {{
            background: #555;
        }}
    </style>
    """, unsafe_allow_html=True)

apply_custom_css()

# ==================== Helper: Display Session in Sidebar ====================

def display_session_in_sidebar(session: Dict, current_session: Dict, session_mgr: SessionManager):
    """Display a session item in the sidebar with preview"""
    is_current = session['id'] == current_session['id']
    is_pinned = session['metadata'].get('pinned', False)

    # Get first message preview
    first_msg = ""
    if session.get('messages'):
        first_msg = session['messages'][0]['content'][:40]
        if len(session['messages'][0]['content']) > 40:
            first_msg += "..."

    # Get relative time
    rel_time = get_relative_time(session.get('updated_at', session['created_at']))

    col1, col2, col3 = st.columns([1, 6, 1])

    with col1:
        # Pin indicator
        if is_pinned:
            st.markdown('<span class="pinned">📌</span>', unsafe_allow_html=True)

    with col2:
        # Session button with preview
        button_type = "primary" if is_current else "secondary"
        icon = "📝" if is_current else "💬"

        # Create multi-line label
        label = f"{icon} **{session['name']}**"

        if st.button(label, key=f"sess_{session['id']}", use_container_width=True, type=button_type):
            session_mgr.switch_session(session['id'])
            st.rerun()

        # Show preview and timestamp
        if first_msg:
            st.markdown(f'<div class="session-preview">{first_msg}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="timestamp">{rel_time}</div>', unsafe_allow_html=True)

    with col3:
        # Action menu
        with st.popover("⋮"):
            if st.button("📌 Pin" if not is_pinned else "📌 Unpin", key=f"pin_{session['id']}", use_container_width=True):
                session_mgr.pin_session(session['id'])
                st.rerun()

            if st.button("🗑️ Delete", key=f"del_{session['id']}", use_container_width=True):
                if len(st.session_state.sessions) > 1:
                    session_mgr.delete_session(session['id'])
                    st.rerun()


# ==================== Sidebar ====================

with st.sidebar:
    # Header
    st.markdown("# **Dheera**")
    st.caption("Professional AI Assistant")

    # New Chat button
    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        session_mgr.create_session()
        st.rerun()

    st.divider()

    # Search bar (Phase 1)
    search_query = st.text_input("🔍 Search sessions...", "", placeholder="Search...", label_visibility="collapsed")

    # Filter options
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        show_pinned = st.toggle("📌", help="Pinned only", value=False)
    with col2:
        show_archived = st.toggle("📦", help="Archived", value=False)
    with col3:
        # Quick theme toggle
        theme_icon = "🌙" if st.session_state.dark_mode else "☀️"
        if st.button(theme_icon, help="Toggle theme", use_container_width=True):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()
    with col4:
        folder_filter = st.selectbox("📁", ["All"] + st.session_state.folders, label_visibility="collapsed", index=0)

    st.divider()

    # ==================== Session List ====================

    current_session = session_mgr.get_current_session()

    # Get sessions based on filters
    if search_query:
        sessions = session_mgr.search_sessions(search_query)
    elif show_pinned:
        sessions = session_mgr.get_all_sessions(pinned_only=True)
    elif folder_filter != "All":
        sessions = session_mgr.get_all_sessions(folder=folder_filter)
    else:
        sessions = session_mgr.get_all_sessions()

    # Group sessions by time (Phase 1: Enhanced previews)
    today = []
    yesterday = []
    last_7_days = []
    older = []

    now = datetime.now()
    for session in sessions:
        updated = datetime.fromisoformat(session.get('updated_at', session['created_at']))
        diff = (now - updated).days

        if diff == 0:
            today.append(session)
        elif diff == 1:
            yesterday.append(session)
        elif diff <= 7:
            last_7_days.append(session)
        else:
            older.append(session)

    # Display pinned sessions first
    pinned_sessions = [s for s in sessions if s['metadata'].get('pinned', False)]
    if pinned_sessions:
        st.markdown('<div class="folder-header">📌 Pinned</div>', unsafe_allow_html=True)
        for session in pinned_sessions:
            display_session_in_sidebar(session, current_session, session_mgr)

    # Display sessions by time groups
    if today:
        st.markdown('<div class="folder-header">📅 Today</div>', unsafe_allow_html=True)
        for session in today:
            if not session['metadata'].get('pinned', False):
                display_session_in_sidebar(session, current_session, session_mgr)

    if yesterday:
        st.markdown('<div class="folder-header">📅 Yesterday</div>', unsafe_allow_html=True)
        for session in yesterday:
            if not session['metadata'].get('pinned', False):
                display_session_in_sidebar(session, current_session, session_mgr)

    if last_7_days:
        st.markdown('<div class="folder-header">📅 Last 7 Days</div>', unsafe_allow_html=True)
        for session in last_7_days:
            if not session['metadata'].get('pinned', False):
                display_session_in_sidebar(session, current_session, session_mgr)

    if older:
        st.markdown('<div class="folder-header">📅 Older</div>', unsafe_allow_html=True)
        for session in older[:5]:  # Show max 5 older sessions
            if not session['metadata'].get('pinned', False):
                display_session_in_sidebar(session, current_session, session_mgr)

    st.divider()

    # ==================== Active Model Selector ====================

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
                        st.success(f"✓ {selected}")
                        st.rerun()

                # Show model info
                selected_provider = next((p for p in providers if p["name"] == selected), None)
                if selected_provider:
                    st.caption(f"**Model:** {selected_provider['model']}")
                    st.caption(f"**Status:** {'🟢 Online' if selected_provider.get('available') else '🔴 Offline'}")
            else:
                st.info("No models available")
        except:
            st.warning("⚠️ Can't load models")
    else:
        st.error("⚠️ Backend offline")

    st.divider()

    # ==================== Advanced Settings ====================

    with st.expander("⚙️ **Settings**", expanded=False):
        st.markdown("##### 🎨 Appearance")

        # Theme toggle with auto-rerun
        new_dark_mode = st.toggle("🌙 Dark Mode", value=st.session_state.dark_mode, key="theme_toggle")

        if new_dark_mode != st.session_state.dark_mode:
            st.session_state.dark_mode = new_dark_mode
            st.rerun()

        st.markdown("##### 🎛️ Generation")

        temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
        max_tokens = st.slider("Max Tokens", 50, 2048, 256, 50)

        st.markdown("##### 📊 Display")

        st.session_state.show_metadata = st.toggle("Show metadata", value=st.session_state.show_metadata)
        auto_scroll = st.toggle("Auto-scroll", value=True)

        st.session_state.temperature = temperature
        st.session_state.max_tokens = max_tokens

    st.divider()

    # ==================== Quick Stats ====================

    with st.expander("📊 **Stats**", expanded=False):
        total_sessions = len(st.session_state.sessions)
        total_messages = sum(len(s.get('messages', [])) for s in st.session_state.sessions.values())

        st.metric("Total Chats", total_sessions)
        st.metric("Total Messages", total_messages)

        if backend_running:
            try:
                stats = get_stats()
                dheera_stats = stats.get("dheera", {})
                st.metric("DQN Steps", dheera_stats.get("dqn", {}).get("total_steps", 0))
            except:
                pass

    st.divider()

    # Navigation
    st.markdown("### 📌 **Navigation**")
    page = st.radio(
        "Go to",
        ["💬 Chat", "🔍 Search & AI", "🔧 Models", "📊 Analytics", "⚙️ Settings"],
        label_visibility="collapsed"
    )


# ==================== Main Content: Chat Page ====================

if page == "💬 Chat":
    # Header
    col1, col2, col3, col4 = st.columns([4, 2, 1, 1])

    with col1:
        st.title(f"💬 {current_session['name']}")

    with col2:
        # Quick rename
        new_name = st.text_input("Session name", current_session['name'], label_visibility="collapsed",
                                  placeholder="Rename chat...", key="rename_input")
        if new_name and new_name != current_session['name']:
            session_mgr.rename_session(current_session['id'], new_name)

    with col3:
        # Export button (Phase 1)
        export_format = st.selectbox("⬇️", ["Export", "Markdown", "JSON", "Text"],
                                     label_visibility="collapsed", key="export_select")

        if export_format != "Export":
            if export_format == "Markdown":
                content = export_to_markdown(current_session)
                filename = f"{current_session['name']}.md"
            elif export_format == "JSON":
                content = export_to_json(current_session)
                filename = f"{current_session['name']}.json"
            else:  # Text
                content = export_to_text(current_session)
                filename = f"{current_session['name']}.txt"

            st.download_button(
                "💾 Download",
                content,
                filename,
                key="download_btn"
            )

    with col4:
        # Clear chat
        if st.button("🗑️", help="Clear chat"):
            current_session['messages'] = []
            st.rerun()

    if not backend_running:
        st.error("❌ Backend is not running. Please start the server first.")
        st.code("python3 api/server.py", language="bash")
        st.stop()

    # Display chat messages (Phase 1: Timestamps, Phase 2: Actions)
    chat_container = st.container()

    with chat_container:
        for idx, message in enumerate(current_session['messages']):
            with st.chat_message(message["role"]):
                # Message header with timestamp (Phase 1)
                timestamp = format_timestamp(message.get('timestamp', ''))
                role = message["role"].title()

                if message["role"] == "assistant" and "metadata" in message:
                    latency = message["metadata"].get("latency_ms", 0)
                    st.caption(f"**{role}** • {timestamp} ({latency:.1f}ms)")
                else:
                    st.caption(f"**{role}** • {timestamp}")

                # Message content
                st.markdown(message["content"])

                # Message actions (Phase 2)
                col1, col2, col3, col4 = st.columns([1, 1, 1, 8])

                with col1:
                    if st.button("📋", key=f"copy_{idx}", help="Copy"):
                        st.code(message["content"], language="markdown")

                with col2:
                    if message["role"] == "assistant" and st.button("🔄", key=f"regen_{idx}", help="Regenerate"):
                        # Find the user message before this
                        if idx > 0:
                            user_msg = current_session['messages'][idx-1]['content']
                            # Remove this and subsequent messages
                            current_session['messages'] = current_session['messages'][:idx]
                            # Re-send
                            result = send_message(user_msg)
                            current_session['messages'].append({
                                "role": "assistant",
                                "content": result.get("response", "Error"),
                                "metadata": result.get("metadata", {}),
                                "timestamp": datetime.now().isoformat()
                            })
                            st.rerun()

                # Show metadata if enabled
                if message["role"] == "assistant" and "metadata" in message and st.session_state.show_metadata:
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

    # Chat input with token counter (Phase 2)
    input_container = st.container()

    with input_container:
        # Token counter placeholder
        token_count_placeholder = st.empty()

        # Chat input
        if prompt := st.chat_input("Type your message...", key="chat_input"):
            # Auto-name session from first message (Phase 1)
            if len(current_session['messages']) == 0:
                auto_name = auto_generate_session_name(prompt)
                session_mgr.rename_session(current_session['id'], auto_name)

            # Add user message
            current_session['messages'].append({
                "role": "user",
                "content": prompt,
                "timestamp": datetime.now().isoformat()
            })

            # Update session metadata
            current_session['updated_at'] = datetime.now().isoformat()
            current_session['metadata']['message_count'] += 1

            # Display user message
            with st.chat_message("user"):
                st.caption(f"**User** • {format_timestamp(datetime.now().isoformat())}")
                st.markdown(prompt)

            # Get AI response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        result = send_message(prompt,
                                            temperature=st.session_state.get('temperature', 0.7),
                                            max_tokens=st.session_state.get('max_tokens', 256))
                        response = result.get("response", "Error: No response")
                        metadata = result.get("metadata", {})

                        # Display response
                        timestamp = datetime.now().isoformat()
                        latency = metadata.get('latency_ms', 0)
                        st.caption(f"**Dheera** • {format_timestamp(timestamp)} ({latency:.1f}ms)")
                        st.markdown(response)

                        # Add to session
                        current_session['messages'].append({
                            "role": "assistant",
                            "content": response,
                            "metadata": metadata,
                            "timestamp": timestamp
                        })

                        # Update metadata
                        current_session['metadata']['total_tokens'] += metadata.get('tokens_used', 0)
                        current_session['metadata']['message_count'] += 1

                        st.rerun()

                    except Exception as e:
                        st.error(f"Error: {e}")


# ==================== Search & AI Page ====================

elif page == "🔍 Search & AI":
    # Import tools
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'connectors', 'tools'))

    try:
        from web_search_tool import WebSearchTool, WebSearchResponse
        from multi_model_agent import MultiModelAgent, MultiModelResponse
    except ImportError as e:
        st.error(f"Failed to import tools: {e}")
        st.info("Make sure web_search_tool.py and multi_model_agent.py are in connectors/tools/")
        st.stop()

    # Center layout with custom CSS
    st.markdown("""
        <style>
        /* Center search container */
        .search-container {
            max-width: 800px;
            margin: 0 auto;
            padding: 3rem 1rem;
        }

        /* Search header */
        .search-header {
            text-align: center;
            margin-bottom: 3rem;
        }

        .search-title {
            font-size: 3rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .search-subtitle {
            font-size: 1.1rem;
            color: #666;
            margin-bottom: 2rem;
        }

        /* Feature buttons */
        .feature-buttons {
            display: flex;
            justify-content: center;
            gap: 1rem;
            flex-wrap: wrap;
            margin-top: 1.5rem;
        }

        /* Results container */
        .results-container {
            margin-top: 2rem;
            padding: 2rem;
            background: rgba(0,0,0,0.02);
            border-radius: 12px;
        }

        /* Search result item */
        .search-result {
            padding: 1rem;
            margin-bottom: 1rem;
            background: white;
            border-radius: 8px;
            border-left: 3px solid #667eea;
        }

        .search-result-title {
            font-size: 1.1rem;
            font-weight: 600;
            color: #1a73e8;
            margin-bottom: 0.5rem;
        }

        .search-result-url {
            font-size: 0.9rem;
            color: #006621;
            margin-bottom: 0.5rem;
        }

        .search-result-snippet {
            font-size: 0.95rem;
            color: #444;
            line-height: 1.5;
        }

        /* Multi-model comparison */
        .model-response {
            padding: 1.5rem;
            margin-bottom: 1rem;
            background: white;
            border-radius: 8px;
            border-top: 3px solid #667eea;
        }

        .model-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
        }

        .model-name {
            font-size: 1.1rem;
            font-weight: 600;
        }

        .model-latency {
            font-size: 0.9rem;
            color: #666;
            background: #f0f0f0;
            padding: 0.3rem 0.8rem;
            border-radius: 12px;
        }
        </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown("""
        <div class="search-container">
            <div class="search-header">
                <h1 class="search-title">Dheera</h1>
                <p class="search-subtitle">AI-Powered Search Engine</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Main search box (centered)
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        search_query = st.text_input(
            "Search",
            placeholder="🔍 Ask anything or search the web...",
            label_visibility="collapsed",
            key="main_search_input"
        )

    # Feature tabs
    tab1, tab2, tab3 = st.tabs(["🌐 Web Search", "🤖 Multi-Model", "⚡ Quick Answer"])

    # ==================== Tab 1: Web Search ====================
    with tab1:
        st.markdown("### 🌐 Web Search + AI Summarization")
        st.markdown("Search the internet and get AI-powered summaries, just like ChatGPT!")

        col1, col2 = st.columns([3, 1])
        with col1:
            web_query = st.text_input("Search query", value=search_query, placeholder="Enter search query...", key="web_search_query")
        with col2:
            num_results = st.number_input("Results", min_value=1, max_value=10, value=5, key="web_num_results")

        col1, col2, col3 = st.columns(3)
        with col1:
            search_btn = st.button("🔍 Search Web", type="primary", use_container_width=True)
        with col2:
            summarize = st.checkbox("AI Summarize", value=True)
        with col3:
            provider = st.selectbox("Provider", ["auto", "duckduckgo", "serpapi", "brave"], index=0)

        if search_btn and web_query:
            with st.spinner("🔍 Searching the web..."):
                try:
                    # Initialize web search tool
                    search_tool = WebSearchTool()

                    # Perform search
                    search_response = search_tool.search(web_query, num_results=num_results, provider=provider)

                    if search_response.error:
                        st.error(f"Search failed: {search_response.error}")
                    elif not search_response.results:
                        st.warning("No results found.")
                    else:
                        # Display search results
                        st.success(f"✅ Found {search_response.total_results} results in {search_response.search_time_ms:.0f}ms using {search_response.provider}")

                        # AI Summarization
                        if summarize:
                            with st.spinner("🤖 Generating AI summary..."):
                                try:
                                    # Format results for LLM
                                    context = search_tool.format_for_llm(search_response)

                                    # Create summarization prompt
                                    summary_prompt = f"Based on these web search results, provide a comprehensive summary answering the query: '{web_query}'\n\n{context}\n\nSummary:"

                                    # Send to current active model
                                    summary_response = requests.post(
                                        f"{BACKEND_URL}/api/chat",
                                        json={"message": summary_prompt},
                                        timeout=60
                                    )

                                    if summary_response.status_code == 200:
                                        summary_data = summary_response.json()
                                        st.markdown("### ✨ AI Summary")
                                        st.markdown(summary_data.get("response", ""))
                                        st.divider()
                                except Exception as e:
                                    st.warning(f"Could not generate summary: {e}")

                        # Display individual results
                        st.markdown("### 🔗 Search Results")
                        for i, result in enumerate(search_response.results, 1):
                            with st.container():
                                st.markdown(f"**{i}. {result.title}**")
                                st.markdown(f"<span style='color: #006621; font-size: 0.9rem;'>{result.url}</span>", unsafe_allow_html=True)
                                st.markdown(result.snippet)
                                st.markdown("---")

                except Exception as e:
                    st.error(f"Error during search: {e}")

    # ==================== Tab 2: Multi-Model ====================
    with tab2:
        st.markdown("### 🤖 Multi-Model Comparison")
        st.markdown("Query multiple AI models simultaneously and compare their responses!")

        # Get available models
        try:
            providers_response = requests.get(f"{BACKEND_URL}/api/llm/providers", timeout=5)
            if providers_response.status_code == 200:
                providers_data = providers_response.json()
                available_models = [p["name"] for p in providers_data.get("providers", [])]
            else:
                available_models = []
        except:
            available_models = []

        if not available_models:
            st.warning("No models available. Add models in the Models page first.")
        else:
            multi_query = st.text_input("Query for all models", value=search_query, placeholder="Enter your question...", key="multi_model_query")

            # Model selection
            st.markdown("**Select models to compare (2-4 recommended):**")
            selected_models = st.multiselect(
                "Models",
                options=available_models,
                default=available_models[:min(3, len(available_models))],
                label_visibility="collapsed"
            )

            col1, col2 = st.columns([2, 1])
            with col1:
                compare_btn = st.button("🤖 Compare Models", type="primary", use_container_width=True)
            with col2:
                generate_consensus = st.checkbox("Generate Consensus", value=True)

            if compare_btn and multi_query and selected_models:
                if len(selected_models) < 2:
                    st.warning("Please select at least 2 models for comparison.")
                else:
                    with st.spinner(f"🤖 Querying {len(selected_models)} models in parallel..."):
                        try:
                            # Initialize multi-model agent
                            agent = MultiModelAgent(api_base=BACKEND_URL)

                            # Query all models
                            multi_response = agent.query_all(multi_query, selected_models)

                            # Display results
                            st.success(f"✅ Queried {len(selected_models)} models in {multi_response.total_time_ms:.0f}ms")
                            st.info(f"🏆 Fastest: **{multi_response.fastest_model}** | 🐌 Slowest: **{multi_response.slowest_model}**")

                            # Display each model's response
                            st.markdown("### 📊 Model Responses")

                            for resp in multi_response.responses:
                                with st.container():
                                    if resp.error:
                                        st.error(f"❌ **{resp.model_name}**: {resp.error}")
                                    else:
                                        col1, col2 = st.columns([3, 1])
                                        with col1:
                                            st.markdown(f"**🤖 {resp.model_name}** ({resp.provider})")
                                        with col2:
                                            st.markdown(f"<span style='background: #f0f0f0; padding: 0.3rem 0.8rem; border-radius: 12px; font-size: 0.9rem;'>⏱️ {resp.latency_ms:.0f}ms</span>", unsafe_allow_html=True)

                                        st.markdown(resp.response_text)
                                        st.markdown("---")

                            # Generate consensus
                            if generate_consensus:
                                with st.spinner("🎯 Generating consensus summary..."):
                                    try:
                                        consensus = agent.generate_consensus(multi_response)
                                        st.markdown("### 🎯 Consensus Summary")
                                        st.info(consensus)
                                    except Exception as e:
                                        st.warning(f"Could not generate consensus: {e}")

                        except Exception as e:
                            st.error(f"Error during multi-model query: {e}")

    # ==================== Tab 3: Quick Answer ====================
    with tab3:
        st.markdown("### ⚡ Quick Answer")
        st.markdown("Get instant answers from your current active model.")

        quick_query = st.text_area("Your question", value=search_query, placeholder="Ask anything...", key="quick_answer_query", height=100)

        if st.button("⚡ Get Answer", type="primary", use_container_width=True):
            if quick_query:
                with st.spinner("⚡ Generating answer..."):
                    try:
                        response = requests.post(
                            f"{BACKEND_URL}/api/chat",
                            json={"message": quick_query},
                            timeout=60
                        )

                        if response.status_code == 200:
                            data = response.json()
                            st.markdown("### 💬 Answer")
                            st.markdown(data.get("response", ""))

                            # Show metadata
                            metadata = data.get("metadata", {})
                            if metadata:
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Model", metadata.get("model", "Unknown"))
                                with col2:
                                    st.metric("Provider", metadata.get("provider", "Unknown"))
                                with col3:
                                    if "tokens_used" in metadata:
                                        st.metric("Tokens", metadata.get("tokens_used", 0))
                        else:
                            st.error(f"Failed to get answer: {response.text}")

                    except Exception as e:
                        st.error(f"Error: {e}")
            else:
                st.warning("Please enter a question.")

    # Footer info
    st.divider()
    st.markdown("""
        <div style='text-align: center; color: #666; padding: 2rem;'>
            <p><strong>🔍 Web Search:</strong> Free DuckDuckGo search with AI summarization</p>
            <p><strong>🤖 Multi-Model:</strong> Compare responses from multiple AI models</p>
            <p><strong>⚡ Quick Answer:</strong> Instant answers from your active model</p>
        </div>
    """, unsafe_allow_html=True)


# ==================== Models Page ====================

elif page == "🔧 Models":
    st.title("🔧 Model Management")

    if not backend_running:
        st.error("❌ Backend is not running.")
        st.stop()

    tab1, tab2, tab3, tab4 = st.tabs(["📋 Available", "➕ Add API", "🔬 Test", "🆚 Compare"])

    with tab1:
        st.subheader("Available Models")

        # Discover button
        if st.button("🔄 Discover Ollama Models"):
            try:
                response = requests.post(f"{API_BASE}/api/llm/discover")
                result = response.json()
                if result.get("success"):
                    st.success(f"✅ Discovered {result.get('discovered', 0)} models, added {result.get('added', 0)} new")
                    st.rerun()
                else:
                    st.error(f"❌ {result.get('error', 'Failed')}")
            except Exception as e:
                st.error(f"Error: {e}")

        # List providers
        try:
            providers_data = get_providers()
            providers = providers_data.get("providers", [])
            active = providers_data.get("active", None)

            for provider in providers:
                with st.container():
                    col1, col2, col3, col4 = st.columns([1, 3, 3, 2])

                    is_active = provider["name"] == active

                    with col1:
                        st.markdown(f"### {'🟢' if is_active else '⚪'}")

                    with col2:
                        st.markdown(f"**{provider['name']}**")
                        if is_active:
                            st.caption("✓ Active")

                    with col3:
                        st.caption(f"Provider: {provider['provider']}")
                        st.caption(f"Model: {provider['model']}")

                    with col4:
                        if not is_active and st.button("Activate", key=f"activate_{provider['name']}"):
                            switch_provider(provider['name'])
                            st.rerun()

                    st.divider()
        except Exception as e:
            st.error(f"Error loading providers: {e}")

    with tab2:
        st.subheader("➕ Add External API Provider")
        st.info("Add models from OpenRouter, OpenAI, Anthropic, or other API providers")

        # Provider type selection
        provider_type = st.selectbox(
            "Provider Type",
            ["OpenRouter", "OpenAI", "Anthropic", "Custom (OpenAI-compatible)"],
            help="Select the API provider you want to add"
        )

        # Provider-specific settings
        col1, col2 = st.columns(2)

        with col1:
            provider_name = st.text_input(
                "Provider Name",
                placeholder="e.g., my_openrouter_claude",
                help="Unique name for this provider (no spaces)"
            )

        with col2:
            api_key = st.text_input(
                "API Key",
                type="password",
                placeholder="sk-or-v1-...",
                help="Your API key (will be stored securely)"
            )

        # Model selection based on provider type
        if provider_type == "OpenRouter":
            model = st.selectbox(
                "Model",
                [
                    "anthropic/claude-3.5-sonnet",
                    "openai/gpt-4-turbo",
                    "openai/gpt-3.5-turbo",
                    "meta-llama/llama-3.1-70b-instruct",
                    "meta-llama/llama-3.1-8b-instruct",
                    "google/gemini-pro-1.5",
                    "mistralai/mixtral-8x7b-instruct",
                    "anthropic/claude-3-opus",
                    "anthropic/claude-3-haiku",
                ],
                help="Select a model from OpenRouter's catalog"
            )
            base_url = "https://openrouter.ai/api/v1"
            backend_provider = "openai"  # OpenRouter uses OpenAI-compatible API

        elif provider_type == "OpenAI":
            model = st.selectbox(
                "Model",
                [
                    "gpt-4-turbo",
                    "gpt-4",
                    "gpt-3.5-turbo",
                    "gpt-3.5-turbo-16k",
                ],
                help="Select an OpenAI model"
            )
            base_url = "https://api.openai.com/v1"
            backend_provider = "openai"

        elif provider_type == "Anthropic":
            model = st.selectbox(
                "Model",
                [
                    "claude-opus-4-20250514",
                    "claude-sonnet-4-20250514",
                    "claude-3-5-sonnet-20241022",
                    "claude-3-opus-20240229",
                    "claude-3-sonnet-20240229",
                    "claude-3-haiku-20240307",
                ],
                help="Select a Claude model"
            )
            base_url = "https://api.anthropic.com/v1"
            backend_provider = "anthropic"

        else:  # Custom
            model = st.text_input(
                "Model Name",
                placeholder="e.g., gpt-4-turbo",
                help="Enter the model name (check provider's documentation)"
            )
            base_url = st.text_input(
                "Base URL",
                placeholder="https://api.example.com/v1",
                help="API base URL (OpenAI-compatible)"
            )
            backend_provider = "openai"  # Assume OpenAI-compatible

        # Advanced settings
        with st.expander("⚙️ Advanced Settings"):
            col1, col2 = st.columns(2)

            with col1:
                timeout = st.slider("Timeout (seconds)", 10, 120, 30)
                max_tokens = st.slider("Max Tokens", 50, 4096, 512)

            with col2:
                temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)

        # Add button
        if st.button("➕ Add Provider", type="primary"):
            if not provider_name:
                st.error("❌ Please enter a provider name")
            elif not api_key:
                st.error("❌ Please enter an API key")
            elif not model:
                st.error("❌ Please enter a model name")
            else:
                try:
                    # Add provider via API
                    response = requests.post(
                        f"{API_BASE}/api/llm/provider",
                        json={
                            "name": provider_name.replace(" ", "_"),
                            "provider": backend_provider,
                            "model": model,
                            "api_key": api_key,
                            "base_url": base_url,
                            "timeout": timeout,
                            "max_tokens": max_tokens,
                            "temperature": temperature,
                        }
                    )

                    if response.status_code == 200:
                        result = response.json()
                        st.success(f"✅ Added provider: {result['name']}")
                        st.info("💡 Go to 'Available' tab to activate it")
                        st.balloons()
                    else:
                        st.error(f"❌ Failed to add provider: {response.text}")

                except Exception as e:
                    st.error(f"❌ Error: {e}")

        st.divider()

        # Quick reference
        st.markdown("### 📚 Quick Reference")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**OpenRouter:**")
            st.code("https://openrouter.ai/keys")
            st.caption("Get your API key")

            st.markdown("**OpenAI:**")
            st.code("https://platform.openai.com/api-keys")
            st.caption("Get your API key")

        with col2:
            st.markdown("**Anthropic:**")
            st.code("https://console.anthropic.com/settings/keys")
            st.caption("Get your API key")

            st.markdown("**Cost Comparison:**")
            st.caption("OpenRouter often offers the cheapest rates")

    with tab3:
        st.subheader("Test Models")
        st.info("Send a test prompt to any model to check response time and quality")

        try:
            providers_data = get_providers()
            providers = providers_data.get("providers", [])

            if providers:
                selected_model = st.selectbox("Select model to test", [p["name"] for p in providers])
                test_prompt = st.text_area("Test prompt", "Explain quantum computing in one sentence.")

                if st.button("🧪 Test"):
                    with st.spinner("Testing..."):
                        # Switch to model, send test, switch back
                        original = providers_data.get("active")
                        switch_provider(selected_model)

                        result = send_message(test_prompt)

                        st.success("✅ Test complete")
                        st.markdown(f"**Response:** {result.get('response', 'No response')}")
                        st.caption(f"Latency: {result.get('metadata', {}).get('latency_ms', 0):.0f}ms")

                        # Switch back
                        if original:
                            switch_provider(original)
        except Exception as e:
            st.error(f"Error: {e}")

    with tab4:
        st.subheader("🆚 Model Comparison (Phase 3)")
        st.info("Compare responses from multiple models side-by-side")

        try:
            providers_data = get_providers()
            providers = providers_data.get("providers", [])

            if len(providers) >= 2:
                col1, col2 = st.columns(2)

                with col1:
                    model1 = st.selectbox("Model 1", [p["name"] for p in providers], key="compare_model1")

                with col2:
                    model2 = st.selectbox("Model 2", [p["name"] for p in providers],
                                         index=min(1, len(providers)-1), key="compare_model2")

                compare_prompt = st.text_area("Prompt", "What is the capital of France?")

                if st.button("🆚 Compare"):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown(f"### {model1}")
                        with st.spinner("Generating..."):
                            switch_provider(model1)
                            result1 = send_message(compare_prompt)
                            st.markdown(result1.get('response', 'No response'))
                            st.caption(f"⏱️ {result1.get('metadata', {}).get('latency_ms', 0):.0f}ms")

                    with col2:
                        st.markdown(f"### {model2}")
                        with st.spinner("Generating..."):
                            switch_provider(model2)
                            result2 = send_message(compare_prompt)
                            st.markdown(result2.get('response', 'No response'))
                            st.caption(f"⏱️ {result2.get('metadata', {}).get('latency_ms', 0):.0f}ms")
            else:
                st.warning("Need at least 2 models for comparison")
        except Exception as e:
            st.error(f"Error: {e}")


# ==================== Analytics Page ====================

elif page == "📊 Analytics":
    st.title("📊 Analytics")

    if not backend_running:
        st.error("❌ Backend is not running.")
        st.stop()

    try:
        stats = get_stats()

        # Check if backend stats are available
        has_backend_stats = stats and "error" not in stats

        # Session stats (from GUI state)
        st.subheader("💬 Chat Statistics")

        col1, col2, col3, col4 = st.columns(4)

        total_sessions = len(st.session_state.sessions)
        total_messages = sum(len(s.get('messages', [])) for s in st.session_state.sessions.values())
        total_tokens = sum(s.get('metadata', {}).get('total_tokens', 0) for s in st.session_state.sessions.values())

        with col1:
            st.metric("Total Chats", total_sessions)
        with col2:
            st.metric("Total Messages", total_messages)
        with col3:
            st.metric("Total Tokens", f"{total_tokens:,}")
        with col4:
            avg_msg = total_messages / max(total_sessions, 1)
            st.metric("Avg Msgs/Chat", f"{avg_msg:.1f}")

        # Model usage (if plotly available)
        if HAS_PLOTLY and total_messages > 0:
            st.subheader("🤖 Model Usage")

            # Count messages per model (from session metadata)
            model_counts = {}
            for session in st.session_state.sessions.values():
                model = session.get('model', 'Unknown')
                model_counts[model] = model_counts.get(model, 0) + len(session.get('messages', []))

            if model_counts:
                fig = px.bar(
                    x=list(model_counts.keys()),
                    y=list(model_counts.values()),
                    labels={'x': 'Model', 'y': 'Messages'},
                    title="Messages per Model"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("💡 No model usage data yet. Start chatting to see statistics!")

        # DQN stats (backend)
        st.subheader("📊 DQN Performance")

        if has_backend_stats:
            dqn_stats = stats.get("dheera", {}).get("dqn", {})

            if dqn_stats:
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Total Steps", dqn_stats.get("total_steps", 0))
                with col2:
                    st.metric("Avg Reward", f"{dqn_stats.get('avg_reward', 0):.3f}")
                with col3:
                    st.metric("Training Loss", f"{dqn_stats.get('loss', 0):.4f}")
            else:
                st.info("💡 No DQN performance data yet. Start chatting to train the agent!")
        else:
            st.info("💡 Backend not initialized yet. Send a chat message to start collecting DQN stats!")

    except Exception as e:
        st.error(f"Error loading stats: {e}")
        st.info("💡 Tip: Try sending a chat message first to initialize the backend.")


# ==================== Settings Page ====================

elif page == "⚙️ Settings":
    st.title("⚙️ Settings")

    st.subheader("🎨 Appearance")

    st.session_state.dark_mode = st.toggle("🌙 Dark Mode", value=st.session_state.dark_mode)

    if st.button("Apply Theme"):
        apply_custom_css()
        st.rerun()

    st.divider()

    st.subheader("💾 Data Management")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("📥 Export All Chats"):
            all_chats = {
                "sessions": list(st.session_state.sessions.values()),
                "exported_at": datetime.now().isoformat()
            }
            st.download_button(
                "Download JSON",
                json.dumps(all_chats, indent=2),
                "dheera_all_chats.json",
                key="export_all"
            )

    with col2:
        if st.button("🗑️ Clear All Chats", type="secondary"):
            if st.button("⚠️ Confirm Delete All?", type="primary"):
                st.session_state.sessions = {}
                session_mgr.create_session()
                st.success("✅ All chats cleared")
                st.rerun()

    st.divider()

    st.subheader("ℹ️ About")

    st.markdown("""
    **Dheera Professional GUI** v0.3.1

    A fully-featured AI assistant with:
    - ✅ Auto-naming from conversations
    - ✅ Session search and filtering
    - ✅ Message timestamps
    - ✅ Export to Markdown/JSON/Text
    - ✅ Message actions (Copy, Regenerate)
    - ✅ Dark/Light theme
    - ✅ Model comparison
    - ✅ Session folders and pinning

    Built with Streamlit • Powered by Dheera AI
    """)

    st.divider()

    st.subheader("⌨️ Keyboard Shortcuts")

    st.markdown("""
    | Shortcut | Action |
    |----------|--------|
    | `Ctrl + N` | New chat |
    | `Ctrl + K` | Focus search |
    | `Ctrl + /` | Toggle sidebar |
    | `Esc` | Clear input |
    """)
