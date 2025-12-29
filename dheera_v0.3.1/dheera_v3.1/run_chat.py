#!/usr/bin/env python3
"""
Dheera v0.3.1 - Interactive Chat Interface
धीर (Sanskrit): Courageous, Wise, Patient

UI-only layer.
No intelligence logic lives here.
"""

import os
import sys
import argparse
import readline
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dheera import Dheera


# ==================== UI HELPERS ====================

def print_banner():
    banner = """
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║   🧠 Dheera v0.3.1 - Cognitive AI Assistant                   ║
║   धीर (Sanskrit): Courageous, Wise, Patient                    ║
║                                                               ║
║   Commands:                                                   ║
║     /help     - Show this help                                ║
║     /stats    - Show statistics                               ║
║     /save     - Save checkpoint                               ║
║     /clear    - Clear conversation                            ║
║     /quit     - Exit chat                                     ║
║                                                               ║
║   Feedback:                                                   ║
║     ++  Strong positive    +  Positive                        ║
║     --  Strong negative    -  Negative                        ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
"""
    print(banner)


def print_help():
    print("""
Available Commands:
  /help, /h      - Show this help message
  /stats, /s     - Show statistics
  /save          - Save model checkpoint
  /clear, /c     - Clear conversation
  /history       - Show conversation history
  /action <n>    - Force action (0-7) for next turn
  /search        - Force search for next turn
  /debug         - Toggle debug mode
  /quit, /q      - Exit

Feedback:
  ++   Strong positive
  +    Positive
  -    Negative
  --   Strong negative
""")


def format_stats(stats: Dict[str, Any]) -> str:
    lines = ["\n📊 Dheera Stats", "-" * 40]

    dqn = stats.get("dqn", {})
    lines.append(f"DQN steps: {dqn.get('total_steps', 0)}")
    lines.append(f"Training steps: {dqn.get('training_steps', 0)}")

    rag = stats.get("rag", {})
    lines.append(f"RAG documents: {rag.get('total_documents', 0)}")

    rlhf = stats.get("rlhf", {})
    lines.append(f"RLHF feedback: {rlhf.get('total_feedback', 0)}")

    slm = stats.get("slm", {})
    lines.append(f"SLM requests: {slm.get('total_requests', 0)}")
    lines.append(f"Avg latency: {slm.get('avg_latency_ms', 0):.0f}ms")

    lines.append("-" * 40)
    return "\n".join(lines)


def print_debug(metadata: Dict[str, Any]):
    print("  [DEBUG]")
    print(f"    Action        : {metadata.get('action_name')}")
    print(f"    Reward        : {metadata.get('reward'):.3f}")
    print(f"    Latency       : {metadata.get('latency_ms'):.0f} ms")
    print(f"    Intent        : {metadata.get('intent')}")

    if metadata.get("search_performed"):
        print("    Search        : YES")
    if metadata.get("rag_used"):
        print("    RAG           : YES")

    # --- New loop observability ---
    goal = metadata.get("goal")
    if goal:
        print(f"    Goal          : {goal.get('primary_goal')}")
        print(f"    Goal risk     : {goal.get('risk')}")

    plan_steps = metadata.get("plan_steps")
    if plan_steps is not None:
        print(f"    Plan steps    : {plan_steps}")

    eval_score = metadata.get("auto_eval_score")
    if eval_score is not None:
        print(f"    AutoEval      : {eval_score}")

    eval_failures = metadata.get("auto_eval_failures")
    if eval_failures:
        print(f"    Eval failures : {eval_failures}")

    print()


# ==================== MAIN LOOP ====================

def main():
    parser = argparse.ArgumentParser(description="Dheera Chat UI")
    parser.add_argument("--config", default="config/dheera_config.yaml")
    parser.add_argument("--identity", default="config/identity.yaml")
    parser.add_argument("--db", default="dheera.db")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    print_banner()

    # Initialize Dheera
    try:
        dheera = Dheera(
            config_path=args.config,
            identity_path=args.identity,
            db_path=args.db,
        )

        if args.checkpoint:
            dheera.load_checkpoint(args.checkpoint)
            print(f"✓ Loaded checkpoint: {args.checkpoint}")

    except Exception as e:
        print(f"❌ Failed to initialize Dheera: {e}")
        import traceback
        traceback.print_exc()
        return 1

    episode_id = dheera.start_episode()
    print(f"Session started: {episode_id[:8]}...\n")

    debug_mode = args.debug
    force_action = None
    force_search = False

    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue

            # ---------------- Commands ----------------
            if user_input.startswith("/"):
                parts = user_input.lower().split()
                cmd = parts[0]
                args_cmd = parts[1:]

                if cmd in ("/quit", "/q", "/exit"):
                    print("\nGoodbye! 👋")
                    dheera.end_episode("User exit")
                    break

                elif cmd in ("/help", "/h"):
                    print_help()
                    continue

                elif cmd in ("/stats", "/s"):
                    print(format_stats(dheera.get_stats()))
                    continue

                elif cmd == "/save":
                    dheera.save_checkpoint()
                    continue

                elif cmd in ("/clear", "/c"):
                    dheera.end_episode("Cleared")
                    episode_id = dheera.start_episode()
                    print(f"\n✓ New session: {episode_id[:8]}...\n")
                    continue

                elif cmd == "/history":
                    for i, t in enumerate(dheera.conversation_history, 1):
                        print(f"{i}. U: {t['user'][:50]}")
                        print(f"   D: {t['assistant'][:50]}")
                    continue

                elif cmd == "/action":
                    if args_cmd and args_cmd[0].isdigit():
                        force_action = int(args_cmd[0])
                        print(f"✓ Next action forced to {force_action}")
                    else:
                        print("Usage: /action <0-7>")
                    continue

                elif cmd == "/search":
                    force_search = True
                    print("✓ Forced search enabled for next turn")
                    continue

                elif cmd == "/debug":
                    debug_mode = not debug_mode
                    print(f"✓ Debug mode {'ON' if debug_mode else 'OFF'}")
                    continue

                else:
                    print("Unknown command. Type /help.")
                    continue

            # ---------------- Agent Loop ----------------
            response, metadata = dheera.process_message(
                user_message=user_input,
                force_action=force_action,
                force_search=force_search,
            )

            force_action = None
            force_search = False

            print(f"\nDheera: {response}\n")

            if debug_mode:
                print_debug(metadata)

        except KeyboardInterrupt:
            print("\nInterrupted. Use /quit to exit.")
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()

    dheera.save_checkpoint()
    print("Session saved. Bye.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
