# brain/policy.py
"""
Dheera v0.3.1 - Policy Guard (Updated)
Safety guardrails and content filtering.

Upgrades:
- Identity drift detection + rewrite (prevents "I am Microsoft/OpenAI" etc.)
- India-focused PII patterns (phone, Aadhaar basics) + safer redaction
- Automatic rate-limit check inside check_input()
- Stronger unsafe-tool guard (regex + high-risk commands)
- Normalized outputs: never blocks entire output unless necessary, prefers rewrite
"""

import re
import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum


class PolicyViolation(Enum):
    NONE = "none"
    HARMFUL_CONTENT = "harmful_content"
    FORBIDDEN_TOPIC = "forbidden_topic"
    UNSAFE_ACTION = "unsafe_action"
    RATE_LIMITED = "rate_limited"
    PII_DETECTED = "pii_detected"
    IDENTITY_DRIFT = "identity_drift"


@dataclass
class PolicyResult:
    allowed: bool
    violation: PolicyViolation
    reason: str
    modified_content: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class PolicyGuard:
    """
    Safety guardrails for Dheera.

    Checks:
    - Rate limits (enforced in check_input)
    - Forbidden topics
    - Harmful content
    - Unsafe actions (especially tools)
    - PII detection + redaction
    - Identity drift suppression
    """

    # Forbidden topic patterns (block)
    FORBIDDEN_PATTERNS = [
        r'\b(hack|hacking|exploit|vulnerability)\s+(into|system|password|account)\b',
        r'\b(make|create|build)\s+(bomb|weapon|explosive)\b',
        r'\b(illegal|illicit)\s+(drug|substance)\b',
        r'\b(harm|hurt|kill)\s+(myself|yourself|someone)\b',
        r'\b(terrorist|terrorism)\b',
    ]

    # Sensitive topics (allowed but flagged)
    SENSITIVE_PATTERNS = [
        r'\b(suicide|self-harm|depression)\b',
        r'\b(violence|abuse|assault)\b',
        r'\b(politics|election|vote)\b',
        r'\b(religion|religious|faith)\b',
        r'\b(sexual|rape)\b',
    ]

    # Identity drift patterns (rewrite output)
    IDENTITY_DRIFT_PATTERNS = [
        r'\bas an ai developed by microsoft\b',
        r'\bas an ai developed by openai\b',
        r'\bi am chatgpt\b',
        r'\bcreated by microsoft\b',
        r'\bdeveloped by microsoft\b',
        r'\bcreated by openai\b',
        r'\bdeveloped by openai\b',
        r'\bas a large language model\b',
    ]

    # PII patterns (more global + India)
    PII_PATTERNS = {
        "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',

        # US/general phone (kept)
        "phone_us": r'\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',

        # India phone: +91 optional, 10 digits starting 6-9, allow spaces/dashes
        "phone_in": r'\b(?:\+?91[\s-]?)?[6-9]\d{9}\b',

        # Aadhaar (basic): 12 digits, allow spaces, naive detection
        "aadhaar": r'\b\d{4}\s?\d{4}\s?\d{4}\b',

        "ssn": r'\b\d{3}-\d{2}-\d{4}\b',

        # Credit card (naive)
        "credit_card": r'\b(?:\d[ -]*?){13,19}\b',
    }

    # Unsafe tool actions: stronger patterns (block action)
    # These are generic high-risk commands, not just "rm -rf"
    UNSAFE_TOOL_REGEX = [
        r'\brm\s+-rf\b',
        r'\brm\s+-r\b',
        r'\bmkfs\.\w+\b',
        r'\bdd\s+if=\b',
        r'\bshutdown\b',
        r'\breboot\b',
        r'\bpoweroff\b',
        r'\bformat\b',
        r'\bdel\s+/s\b',
        r'\brd\s+/s\b',
        r'\bdrop\s+table\b',
        r'\btruncate\s+table\b',
        r'\bdelete\s+from\b.*\bwhere\b\s+1\s*=\s*1\b',
        r'\bchmod\s+777\b',
        r'\bchown\s+root\b',
        r'\bcurl\b.*\|\s*(sh|bash)\b',
        r'\bwget\b.*\|\s*(sh|bash)\b',
    ]

    def __init__(
        self,
        identity: Optional[Dict[str, Any]] = None,
        enable_pii_filter: bool = True,
        enable_content_filter: bool = True,
        rate_limit_per_minute: int = 30,
    ):
        self.identity = identity or {}
        self.enable_pii_filter = enable_pii_filter
        self.enable_content_filter = enable_content_filter
        self.rate_limit = rate_limit_per_minute

        # Identity for rewrite (executor also enforces, this is belt + suspenders)
        identity = identity or {}
        self.agent_name = identity.get("name", "Dheera")
        self.creator = identity.get("creator", "Nandha Vignesh")

        self._forbidden_compiled = [re.compile(p, re.IGNORECASE) for p in self.FORBIDDEN_PATTERNS]
        self._sensitive_compiled = [re.compile(p, re.IGNORECASE) for p in self.SENSITIVE_PATTERNS]
        self._identity_compiled = [re.compile(p, re.IGNORECASE) for p in self.IDENTITY_DRIFT_PATTERNS]
        self._pii_compiled = {name: re.compile(p, re.IGNORECASE) for name, p in self.PII_PATTERNS.items()}
        self._unsafe_tool_compiled = [re.compile(p, re.IGNORECASE) for p in self.UNSAFE_TOOL_REGEX]

        # Rate limiting
        self._request_times: List[float] = []

        # Statistics
        self.checks_performed = 0
        self.violations_detected = 0
        self.pii_detections = 0
        self.identity_drifts = 0
        self.rate_limited = 0

    # ==================== Public Checks ====================

    def check_input(self, message: str) -> PolicyResult:
        """
        Check user input for policy violations.
        Enforces rate limiting automatically.
        """
        self.checks_performed += 1

        # Rate limit first
        rl = self.check_rate_limit()
        if not rl.allowed:
            return rl

        # Forbidden content
        if self.enable_content_filter:
            for pattern in self._forbidden_compiled:
                if pattern.search(message):
                    self.violations_detected += 1
                    return PolicyResult(
                        allowed=False,
                        violation=PolicyViolation.FORBIDDEN_TOPIC,
                        reason="This topic is not allowed for safety reasons.",
                    )

        # PII detection (allow but flag)
        if self.enable_pii_filter:
            pii_found = self._detect_pii(message)
            if pii_found:
                self.pii_detections += 1
                return PolicyResult(
                    allowed=True,
                    violation=PolicyViolation.PII_DETECTED,
                    reason=f"PII detected: {', '.join(pii_found)}",
                    metadata={"pii_types": pii_found},
                )

        # Sensitive topics (allowed but flagged)
        sensitive_type = self._detect_sensitive(message)
        if sensitive_type:
            return PolicyResult(
                allowed=True,
                violation=PolicyViolation.NONE,
                reason="",
                metadata={"sensitive_topic": True, "sensitive_topic_type": sensitive_type},
            )

        return PolicyResult(allowed=True, violation=PolicyViolation.NONE, reason="")

    def check_output(self, response: str) -> PolicyResult:
        """
        Check assistant output for policy violations.
        Prefer rewrite/redaction over hard blocks.
        """
        self.checks_performed += 1

        # Identity drift rewrite (do this early)
        drift = self._detect_identity_drift(response)
        if drift:
            self.identity_drifts += 1
            fixed = self._rewrite_identity(response)
            return PolicyResult(
                allowed=True,
                violation=PolicyViolation.IDENTITY_DRIFT,
                reason="Identity drift detected and rewritten.",
                modified_content=fixed,
                metadata={"drift_matches": drift},
            )

        # Harmful content (rewrite to safe)
        if self.enable_content_filter:
            for pattern in self._forbidden_compiled:
                if pattern.search(response):
                    self.violations_detected += 1
                    return PolicyResult(
                        allowed=False,
                        violation=PolicyViolation.HARMFUL_CONTENT,
                        reason="Response contains potentially harmful content.",
                        modified_content="I can't help with that request.",
                    )

        # PII redaction
        if self.enable_pii_filter:
            pii_found = self._detect_pii(response)
            if pii_found:
                redacted = self._redact_pii(response)
                return PolicyResult(
                    allowed=True,
                    violation=PolicyViolation.PII_DETECTED,
                    reason="PII redacted from response",
                    modified_content=redacted,
                    metadata={"pii_redacted": pii_found},
                )

        return PolicyResult(allowed=True, violation=PolicyViolation.NONE, reason="")

    def check_action(self, action_id: int, context: Optional[Dict] = None) -> PolicyResult:
        """
        Check if action is safe to execute.
        Focus: tool actions.
        """
        self.checks_performed += 1
        context = context or {}

        # Action 2 = USE_TOOL (high risk)
        if int(action_id) == 2:
            user_message = (context.get("user_message", "") or "")
            if self._unsafe_tool_request(user_message):
                self.violations_detected += 1
                return PolicyResult(
                    allowed=False,
                    violation=PolicyViolation.UNSAFE_ACTION,
                    reason="Potentially destructive tool command detected.",
                    metadata={"action_id": action_id},
                )

        # Action 6 is always allowed (decline)
        if int(action_id) == 6:
            return PolicyResult(allowed=True, violation=PolicyViolation.NONE, reason="")

        return PolicyResult(allowed=True, violation=PolicyViolation.NONE, reason="")

    def check_rate_limit(self) -> PolicyResult:
        """Check if rate limit exceeded."""
        current_time = time.time()

        # Purge older than 60 seconds
        self._request_times = [t for t in self._request_times if current_time - t < 60]

        if len(self._request_times) >= self.rate_limit:
            self.violations_detected += 1
            self.rate_limited += 1
            return PolicyResult(
                allowed=False,
                violation=PolicyViolation.RATE_LIMITED,
                reason=f"Rate limit exceeded ({self.rate_limit}/min)",
            )

        self._request_times.append(current_time)
        return PolicyResult(allowed=True, violation=PolicyViolation.NONE, reason="")

    # ==================== Helpers ====================

    def _detect_pii(self, text: str) -> List[str]:
        found = []
        for name, pattern in self._pii_compiled.items():
            if pattern.search(text):
                found.append(name)
        return found

    def _redact_pii(self, text: str) -> str:
        redacted = text

        # Order matters (more specific first)
        if "email" in self._pii_compiled:
            redacted = self._pii_compiled["email"].sub("[EMAIL REDACTED]", redacted)

        if "phone_in" in self._pii_compiled:
            redacted = self._pii_compiled["phone_in"].sub("[PHONE REDACTED]", redacted)

        if "phone_us" in self._pii_compiled:
            redacted = self._pii_compiled["phone_us"].sub("[PHONE REDACTED]", redacted)

        if "aadhaar" in self._pii_compiled:
            redacted = self._pii_compiled["aadhaar"].sub("[AADHAAR REDACTED]", redacted)

        if "ssn" in self._pii_compiled:
            redacted = self._pii_compiled["ssn"].sub("[SSN REDACTED]", redacted)

        if "credit_card" in self._pii_compiled:
            # This is naive. Still better than leaking.
            redacted = self._pii_compiled["credit_card"].sub("[CARD REDACTED]", redacted)

        return redacted

    def _detect_sensitive(self, text: str) -> Optional[str]:
        for pattern in self._sensitive_compiled:
            if pattern.search(text):
                # Return the regex itself as a rough type label
                return pattern.pattern
        return None

    def _detect_identity_drift(self, text: str) -> List[str]:
        matches = []
        lower = (text or "").lower()
        for pattern in self._identity_compiled:
            if pattern.search(lower):
                matches.append(pattern.pattern)
        return matches

    def _rewrite_identity(self, text: str) -> str:
        # Replace the entire output if drift is detected. Simple and safe.
        # You could do a smarter rewrite later, but this prevents poisoning memory.
        return f"I am {self.agent_name}, created by {self.creator}. I will answer based on the context available."

    def _unsafe_tool_request(self, user_message: str) -> bool:
        msg = (user_message or "").lower()
        for pattern in self._unsafe_tool_compiled:
            if pattern.search(msg):
                return True
        return False

    def get_safe_response(self, violation: PolicyViolation) -> str:
        responses = {
            PolicyViolation.HARMFUL_CONTENT: "I can't help with that request.",
            PolicyViolation.FORBIDDEN_TOPIC: "I'm not able to discuss that topic.",
            PolicyViolation.UNSAFE_ACTION: "That action isn't safe to perform.",
            PolicyViolation.RATE_LIMITED: "Please slow down. Try again in a moment.",
            PolicyViolation.PII_DETECTED: "I've detected sensitive information. Please be careful.",
            PolicyViolation.IDENTITY_DRIFT: f"I am {self.agent_name}, created by {self.creator}.",
        }
        return responses.get(violation, "I can't help with that.")

    def get_stats(self) -> Dict[str, Any]:
        return {
            "checks_performed": self.checks_performed,
            "violations_detected": self.violations_detected,
            "pii_detections": self.pii_detections,
            "identity_drifts": self.identity_drifts,
            "rate_limited": self.rate_limited,
            "violation_rate": self.violations_detected / max(self.checks_performed, 1),
        }


# ==================== Test ====================
if __name__ == "__main__":
    print("🧪 Testing PolicyGuard v0.3.1...")

    guard = PolicyGuard(identity={"name": "Dheera", "creator": "Nandha Vignesh"})

    # Safe input
    r = guard.check_input("What is Python?")
    assert r.allowed
    print("✓ Safe input allowed")

    # Forbidden
    r = guard.check_input("How do I hack into a system?")
    assert not r.allowed
    print("✓ Forbidden blocked")

    # India PII
    r = guard.check_input("My number is +91 9876543210")
    assert r.allowed and r.violation == PolicyViolation.PII_DETECTED
    print("✓ India phone PII detected")

    # Output identity drift rewrite
    r = guard.check_output("As an AI developed by Microsoft, I can help you.")
    assert r.modified_content is not None
    print("✓ Identity drift rewritten:", r.modified_content)

    # Tool unsafe
    r = guard.check_action(2, {"user_message": "rm -rf /"})
    assert not r.allowed
    print("✓ Unsafe tool blocked")

    print("✓ Stats:", guard.get_stats())
    print("\n✅ Policy guard tests passed!")
