"""
Dheera Chat Interface
Terminal chat interface with improved feedback detection.
"""

import sys
import time
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Any
from datetime import datetime
from enum import Enum


class MessageRole(Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass
class ChatMessage:
    """A single chat message."""
    role: MessageRole
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }
    
    @classmethod
    def user(cls, content: str, **metadata) -> "ChatMessage":
        return cls(role=MessageRole.USER, content=content, metadata=metadata)
    
    @classmethod
    def assistant(cls, content: str, **metadata) -> "ChatMessage":
        return cls(role=MessageRole.ASSISTANT, content=content, metadata=metadata)


@dataclass
class ChatSession:
    """A chat session containing message history."""
    session_id: str
    messages: List[ChatMessage] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_message(self, message: ChatMessage):
        self.messages.append(message)
    
    def add_user_message(self, content: str, **metadata) -> ChatMessage:
        msg = ChatMessage.user(content, **metadata)
        self.add_message(msg)
        return msg
    
    def add_assistant_message(self, content: str, **metadata) -> ChatMessage:
        msg = ChatMessage.assistant(content, **metadata)
        self.add_message(msg)
        return msg
    
    def get_history(self, limit: Optional[int] = None) -> List[Dict[str, str]]:
        messages = self.messages[-limit:] if limit else self.messages
        return [{"role": m.role.value, "content": m.content} for m in messages]
    
    @property
    def turn_count(self) -> int:
        return len([m for m in self.messages if m.role == MessageRole.USER])


class ChatInterface:
    """Terminal-based chat interface for Dheera."""
    
    COLORS = {
        "reset": "\033[0m",
        "bold": "\033[1m",
        "dim": "\033[2m",
        "green": "\033[92m",
        "blue": "\033[94m",
        "yellow": "\033[93m",
        "red": "\033[91m",
        "cyan": "\033[96m",
        "magenta": "\033[95m",
    }
    
    COMMANDS = {
        "/help": "Show available commands",
        "/stats": "Show agent statistics",
        "/feedback": "Provide feedback (e.g., /feedback good)",
        "/save": "Save current session",
        "/clear": "Clear conversation history",
        "/quit": "Exit the chat",
        "/debug": "Toggle debug mode",
        "/action": "Show last action details",
    }
    
    # Improved feedback patterns - using word boundaries
    POSITIVE_PATTERNS = [
        r'\bgood\b', r'\bgreat\b', r'\bhelpful\b', r'\bthanks\b', r'\bthank you\b',
        r'\bperfect\b', r'\bexcellent\b', r'\bawesome\b', r'\blove\b', r'\bloved\b',
        r'\bamazing\b', r'\bnice\b', r'\bwonderful\b', r'\bbrilliant\b', r'\byes\b',
        r'\bcorrect\b', r'\bright\b', r'\bexactly\b', r'👍', r'\+1', r'❤️', r'🎉',
    ]
    
    NEGATIVE_PATTERNS = [
        r'\bbad\b', r'\bwrong\b', r'\bunhelpful\b', r'\bincorrect\b', r'\bterrible\b',
        r'\bawful\b', r'\bhate\b', r'\buseless\b', r'\bconfused\b', r'\bfrustrat',
        r'\bnope\b', r'\bno[\s,\.]', r'^no$', r'👎', r'-1',
    ]
    
    def __init__(
        self,
        agent_name: str = "Dheera",
        show_colors: bool = True,
        debug_mode: bool = False,
        on_message: Optional[Callable[[str, ChatSession], str]] = None,
        on_feedback: Optional[Callable[[float, ChatSession], None]] = None,
    ):
        self.agent_name = agent_name
        self.show_colors = show_colors and self._supports_color()
        self.debug_mode = debug_mode
        self.on_message = on_message
        self.on_feedback = on_feedback
        
        self.session: Optional[ChatSession] = None
        self.last_action_info: Dict[str, Any] = {}
        self.running = False
        
    def _supports_color(self) -> bool:
        return hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()
    
    def _color(self, text: str, color: str) -> str:
        if not self.show_colors:
            return text
        return f"{self.COLORS.get(color, '')}{text}{self.COLORS['reset']}"
    
    def _print_header(self):
        header = f"""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   {self._color('D H E E R A', 'cyan')} {self._color('(धीर)', 'dim')}                                    ║
║   {self._color('Adaptive AI Agent with Learning Core', 'dim')}                   ║
║   {self._color('Created by Nandha', 'yellow')}                                    ║
║                                                              ║
║   Type {self._color('/help', 'yellow')} for commands, {self._color('/quit', 'yellow')} to exit               ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""
        print(header)
    
    def _print_prompt(self):
        prompt = self._color("You: ", "green")
        print(prompt, end="", flush=True)
    
    def _print_response(self, response: str):
        name = self._color(f"{self.agent_name}: ", "cyan")
        print(f"\n{name}{response}\n")
    
    def _print_debug(self, info: Dict[str, Any]):
        if not self.debug_mode:
            return
        print(self._color("\n[Debug Info]", "dim"))
        for key, value in info.items():
            if key == "q_values":
                print(self._color(f"  {key}: {[f'{v:.3f}' for v in value]}", "dim"))
            else:
                print(self._color(f"  {key}: {value}", "dim"))
        print()
    
    def _print_error(self, message: str):
        print(self._color(f"Error: {message}", "red"))
    
    def _print_info(self, message: str):
        print(self._color(f"ℹ {message}", "yellow"))
    
    def _handle_command(self, command: str) -> bool:
        parts = command.strip().split(maxsplit=1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""
        
        if cmd == "/help":
            print(self._color("\nAvailable commands:", "bold"))
            for cmd_name, description in self.COMMANDS.items():
                print(f"  {self._color(cmd_name, 'yellow')}: {description}")
            print()
            return True
        
        elif cmd == "/quit" or cmd == "/exit":
            self._print_info("Goodbye! Session ended.")
            return False
        
        elif cmd == "/debug":
            self.debug_mode = not self.debug_mode
            status = "enabled" if self.debug_mode else "disabled"
            self._print_info(f"Debug mode {status}")
            return True
        
        elif cmd == "/clear":
            if self.session:
                self.session.messages.clear()
            self._print_info("Conversation cleared")
            return True
        
        elif cmd == "/stats":
            self._show_stats()
            return True
        
        elif cmd == "/action":
            self._show_action_info()
            return True
        
        elif cmd == "/feedback":
            self._handle_explicit_feedback(args)
            return True
        
        elif cmd == "/save":
            self._save_session()
            return True
        
        else:
            self._print_error(f"Unknown command: {cmd}. Type /help for help.")
            return True
    
    def _handle_explicit_feedback(self, feedback_text: str):
        """Process explicit /feedback command."""
        feedback_text = feedback_text.lower().strip()
        
        if not feedback_text:
            self._print_info("Usage: /feedback good|bad|great|terrible|<score>")
            return
        
        # Check for numeric score
        try:
            reward = float(feedback_text)
            reward = max(-1.0, min(1.0, reward))  # Clamp to [-1, 1]
        except ValueError:
            # Check patterns
            reward = self._calculate_feedback_score(feedback_text, explicit=True)
        
        if self.on_feedback and self.session:
            self.on_feedback(reward, self.session)
        
        if reward > 0:
            self._print_info(f"Positive feedback recorded: +{reward:.2f} 👍")
        elif reward < 0:
            self._print_info(f"Negative feedback recorded: {reward:.2f} 👎")
        else:
            self._print_info(f"Neutral feedback recorded: {reward:.2f}")
    
    def _calculate_feedback_score(self, text: str, explicit: bool = False) -> float:
        """Calculate feedback score from text using regex patterns."""
        text_lower = text.lower()
        
        pos_matches = sum(1 for p in self.POSITIVE_PATTERNS if re.search(p, text_lower))
        neg_matches = sum(1 for p in self.NEGATIVE_PATTERNS if re.search(p, text_lower))
        
        if explicit:
            # Explicit feedback: stronger signal
            if pos_matches > neg_matches:
                return 1.0
            elif neg_matches > pos_matches:
                return -0.5
            else:
                return 0.1
        else:
            # Implicit feedback: weaker signal, only if clear
            if pos_matches > 0 and neg_matches == 0:
                return 0.3  # Mild positive
            elif neg_matches > 0 and pos_matches == 0:
                return -0.2  # Mild negative
        
        return 0.0  # No clear feedback
    
    def _detect_implicit_feedback(self, message: str) -> Optional[float]:
        """Detect implicit feedback in user message."""
        score = self._calculate_feedback_score(message, explicit=False)
        return score if score != 0.0 else None
    
    def _show_stats(self):
        if not self.session:
            self._print_info("No active session")
            return
        
        print(self._color("\n📊 Session Statistics:", "bold"))
        print(f"  Session ID: {self.session.session_id}")
        print(f"  Turns: {self.session.turn_count}")
        print(f"  Messages: {len(self.session.messages)}")
        
        duration = time.time() - self.session.created_at
        print(f"  Duration: {duration/60:.1f} minutes")
        
        if self.last_action_info:
            print(f"\n  Last action: {self.last_action_info.get('action_name', 'N/A')}")
            epsilon = self.last_action_info.get('epsilon', 0)
            print(f"  Epsilon: {epsilon:.4f} ({(1-epsilon)*100:.1f}% learned behavior)")
        print()
    
    def _show_action_info(self):
        if not self.last_action_info:
            self._print_info("No action recorded yet")
            return
        
        print(self._color("\n🎯 Last Action Details:", "bold"))
        for key, value in self.last_action_info.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            elif key == "q_values":
                print(f"  {key}:")
                actions = ["ASK_CLARIFY", "DETAILED", "CONCISE", "STEP_BY_STEP", "CALL_TOOL", "REFLECT"]
                for i, (name, q) in enumerate(zip(actions, value)):
                    bar = "█" * int(max(0, (q + 0.2) * 20))
                    print(f"    {i}. {name:12s}: {q:+.3f} {bar}")
            else:
                print(f"  {key}: {value}")
        print()
    
    def _save_session(self):
        if not self.session:
            self._print_info("No active session to save")
            return
        
        import json
        filename = f"session_{self.session.session_id}.json"
        
        data = {
            "session_id": self.session.session_id,
            "created_at": self.session.created_at,
            "messages": [m.to_dict() for m in self.session.messages],
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        self._print_info(f"Session saved to {filename}")
    
    def start(self, session_id: Optional[str] = None):
        """Start the chat interface."""
        session_id = session_id or f"session_{int(time.time())}"
        self.session = ChatSession(session_id=session_id)
        
        self._print_header()
        self.running = True
        
        while self.running:
            try:
                self._print_prompt()
                user_input = input().strip()
                
                if not user_input:
                    continue
                
                if user_input.startswith("/"):
                    self.running = self._handle_command(user_input)
                    continue
                
                # Detect implicit feedback BEFORE processing
                implicit_feedback = self._detect_implicit_feedback(user_input)
                if implicit_feedback and self.on_feedback:
                    self.on_feedback(implicit_feedback, self.session)
                    feedback_type = "+" if implicit_feedback > 0 else ""
                    print(self._color(f"  [Implicit feedback: {feedback_type}{implicit_feedback:.2f}]", "dim"))
                
                self.session.add_user_message(user_input)
                
                if self.on_message:
                    response = self.on_message(user_input, self.session)
                else:
                    response = "Not connected to agent."
                
                self.session.add_assistant_message(response)
                self._print_response(response)
                self._print_debug(self.last_action_info)
                
            except KeyboardInterrupt:
                print("\n")
                self._print_info("Interrupted. Type /quit to exit.")
            except EOFError:
                self.running = False
        
        return self.session
    
    def set_action_info(self, info: Dict[str, Any]):
        self.last_action_info = info
