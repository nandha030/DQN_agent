#!/usr/bin/env python3
"""
Dheera Chat Runner
Launch Dheera in interactive chat mode.

Usage:
    python run_chat.py                      # Default settings
    python run_chat.py --model mistral      # Use specific model
    python run_chat.py --provider openai    # Use OpenAI
    python run_chat.py --debug              # Enable debug mode
    python run_chat.py --config config.yaml # Use config file
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from dheera import Dheera, DheeraConfig, create_dheera
from connectors.chat_interface import ChatInterface, ChatSession


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Launch Dheera in interactive chat mode",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_chat.py                        # Default (Ollama + phi3:mini)
  python run_chat.py --model mistral        # Use Mistral model
  python run_chat.py --provider openai --model gpt-4o-mini
  python run_chat.py --debug                # Show debug info
  python run_chat.py --no-color             # Disable colors
        """
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to YAML configuration file"
    )
    
    parser.add_argument(
        "--provider", "-p",
        type=str,
        default="ollama",
        choices=["ollama", "openai", "anthropic"],
        help="SLM provider (default: ollama)"
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="phi3:mini",
        help="Model name (default: phi3:mini)"
    )
    
    parser.add_argument(
        "--debug", "-d",
        action="store_true",
        help="Enable debug mode (show action details)"
    )
    
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored output"
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to DQN checkpoint to load"
    )
    
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Start with fresh DQN (ignore existing checkpoint)"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    print("""
    ╔═══════════════════════════════════════════════════════╗
    ║                                                       ║
    ║       D H E E R A  (धीर)                             ║
    ║       Adaptive AI Agent with Learning Core            ║
    ║                                                       ║
    ║       "Courageous • Wise • Patient"                  ║
    ║                                                       ║
    ╚═══════════════════════════════════════════════════════╝
    """)
    
    # Create Dheera instance
    print("Initializing Dheera...\n")
    
    if args.config:
        dheera = Dheera(config_path=args.config)
    else:
        config = DheeraConfig(
            slm_provider=args.provider,
            slm_model=args.model,
            debug_mode=args.debug,
        )
        dheera = Dheera(config=config)
    
    # Handle checkpoint loading
    if args.checkpoint:
        dheera.load_checkpoint(args.checkpoint)
    elif args.fresh:
        print("  Starting with fresh DQN (checkpoint ignored)")
    
    # Set debug mode
    if args.debug:
        dheera.set_debug_mode(True)
    
    # Check SLM availability
    if not dheera.slm.is_available():
        print("\n⚠️  Warning: SLM is not available!")
        print(f"   Provider: {args.provider}")
        print(f"   Model: {args.model}")
        
        if args.provider == "ollama":
            print("\n   Make sure Ollama is running:")
            print("   $ ollama serve")
            print(f"   $ ollama pull {args.model}")
        elif args.provider == "openai":
            print("\n   Make sure OPENAI_API_KEY is set:")
            print("   $ export OPENAI_API_KEY='your-key'")
        elif args.provider == "anthropic":
            print("\n   Make sure ANTHROPIC_API_KEY is set:")
            print("   $ export ANTHROPIC_API_KEY='your-key'")
        
        print("\n   Continuing anyway (responses may be limited)...\n")
    
    # Create chat interface
    def handle_message(user_message: str, session: ChatSession) -> str:
        """Handle incoming user message."""
        response = dheera.process_message(user_message, session)
        
        # Update chat interface with action info
        chat.set_action_info(dheera.last_action_info)
        
        return response
    
    def handle_feedback(reward: float, session: ChatSession):
        """Handle user feedback."""
        dheera.provide_feedback(reward, session)
        
        if args.debug:
            print(f"  [Feedback recorded: {reward:+.2f}]")
    
    chat = ChatInterface(
        agent_name="Dheera",
        show_colors=not args.no_color,
        debug_mode=args.debug,
        on_message=handle_message,
        on_feedback=handle_feedback,
    )
    
    # Print stats
    stats = dheera.get_stats()
    print(f"\n📊 Agent Stats:")
    print(f"   DQN Steps: {stats['dqn']['total_steps']}")
    print(f"   Epsilon: {stats['dqn']['epsilon']:.3f}")
    print(f"   Buffer: {stats['dqn']['buffer_size']} transitions")
    print(f"   Episodes: {stats['memory'].get('total_episodes', 0)}")
    print()
    
    # Start chat
    try:
        session = chat.start()
        
        # End conversation when done
        dheera.end_conversation(summary="Chat session ended by user")
        
    except KeyboardInterrupt:
        print("\n\nInterrupted. Saving checkpoint...")
        dheera.save_checkpoint()
        dheera.end_conversation(summary="Chat interrupted")
    
    # Print final stats
    print("\n📊 Final Stats:")
    final_stats = dheera.get_stats()
    print(f"   Total Interactions: {final_stats['agent']['interactions']}")
    print(f"   DQN Steps: {final_stats['dqn']['total_steps']}")
    print(f"   Episodes Stored: {final_stats['memory'].get('total_episodes', 0)}")
    
    print("\n👋 Goodbye! Dheera will remember this conversation.\n")


if __name__ == "__main__":
    main()
