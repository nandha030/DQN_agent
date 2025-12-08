#!/usr/bin/env python3
# run_chat.py
"""
Dheera Chat Runner
Launch Dheera in interactive chat mode.
Version 0.2.0 - Enhanced with web search and 7-action support.

Usage:
    python run_chat.py                      # Default settings
    python run_chat.py --model mistral      # Use specific model
    python run_chat.py --provider openai    # Use OpenAI
    python run_chat.py --debug              # Enable debug mode
    python run_chat.py --config config.yaml # Use config file
    python run_chat.py --no-search          # Disable web search
"""

import sys
import argparse
import signal
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from dheera import Dheera, create_dheera


# ANSI color codes
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    
    # Foreground
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    
    # Bright
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"


class NoColors:
    """Dummy color class for --no-color mode."""
    def __getattr__(self, name):
        return ""


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Launch Dheera in interactive chat mode",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_chat.py                        # Default (Ollama + phi3:latest)
  python run_chat.py --model mistral        # Use Mistral model
  python run_chat.py --provider openai --model gpt-4o-mini
  python run_chat.py --debug                # Show debug info
  python run_chat.py --no-color             # Disable colors
  python run_chat.py --no-search            # Disable web search
  python run_chat.py --search-test          # Test search on startup

Commands during chat:
  /help       - Show help
  /stats      - Show agent statistics
  /search     - Force a web search
  /actions    - Show action distribution
  /save       - Save checkpoint
  /debug      - Toggle debug mode
  /quit       - Exit chat
  
Feedback:
  ++          - Strong positive feedback (+1.0)
  +           - Positive feedback (+0.5)
  -           - Negative feedback (-0.5)
  --          - Strong negative feedback (-1.0)
        """
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="./config/dheera_config.yaml",
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
        default="phi3:latest",
        help="Model name (default: phi3:latest)"
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
        "--no-search",
        action="store_true",
        help="Disable web search capability"
    )
    
    parser.add_argument(
        "--search-test",
        action="store_true",
        help="Test web search on startup"
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
    
    parser.add_argument(
        "--echo-mode",
        action="store_true",
        help="Use echo mode (no SLM required, for testing)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    return parser.parse_args()


def print_banner(c: Colors):
    """Print the Dheera banner."""
    print(f"""
{c.CYAN}{c.BOLD}    ╔═══════════════════════════════════════════════════════╗
    ║                                                       ║
    ║       {c.BRIGHT_MAGENTA}D H E E R A  (धीर){c.CYAN}                             ║
    ║       {c.WHITE}Adaptive AI Agent with Learning Core{c.CYAN}            ║
    ║                                                       ║
    ║       {c.YELLOW}"Courageous • Wise • Patient"{c.CYAN}                  ║
    ║                                                       ║
    ║       {c.DIM}v0.2.0 - Web Search Enabled{c.RESET}{c.CYAN}                   ║
    ╚═══════════════════════════════════════════════════════╝{c.RESET}
    """)


def print_help(c: Colors):
    """Print help information."""
    print(f"""
{c.CYAN}{c.BOLD}Commands:{c.RESET}
  {c.YELLOW}/help{c.RESET}       - Show this help message
  {c.YELLOW}/stats{c.RESET}      - Show agent statistics
  {c.YELLOW}/search{c.RESET}     - Force search on next message
  {c.YELLOW}/actions{c.RESET}    - Show action distribution
  {c.YELLOW}/save{c.RESET}       - Save checkpoint
  {c.YELLOW}/debug{c.RESET}      - Toggle debug mode
  {c.YELLOW}/clear{c.RESET}      - Clear conversation history
  {c.YELLOW}/quit{c.RESET}       - Exit chat (or Ctrl+C)

{c.CYAN}{c.BOLD}Feedback:{c.RESET}
  {c.GREEN}++{c.RESET}          - Strong positive (+1.0)
  {c.GREEN}+{c.RESET}           - Positive (+0.5)
  {c.RED}-{c.RESET}           - Negative (-0.5)
  {c.RED}--{c.RESET}          - Strong negative (-1.0)

{c.CYAN}{c.BOLD}Search Tips:{c.RESET}
  Start messages with "search", "find", or "what is the latest"
  to trigger web search automatically.
""")


def print_stats(dheera: Dheera, c: Colors):
    """Print agent statistics."""
    stats = dheera.get_stats()
    
    print(f"\n{c.CYAN}{c.BOLD}📊 Agent Statistics:{c.RESET}")
    
    # Agent stats
    print(f"\n{c.YELLOW}Agent:{c.RESET}")
    print(f"  Interactions: {stats.get('interactions', 0)}")
    print(f"  Current session turns: {stats.get('session_turns', 0)}")
    
    # DQN stats
    dqn = stats.get('dqn', {})
    print(f"\n{c.YELLOW}DQN Brain:{c.RESET}")
    print(f"  Total steps: {dqn.get('total_steps', 0)}")
    print(f"  Epsilon: {dqn.get('epsilon', 1.0):.4f}")
    print(f"  Buffer size: {dqn.get('buffer_size', 0)}")
    print(f"  Last loss: {dqn.get('last_loss', 0):.6f}")
    
    # Memory stats
    memory = stats.get('memory', {})
    print(f"\n{c.YELLOW}Memory:{c.RESET}")
    print(f"  Episodes stored: {memory.get('total_episodes', 0)}")
    print(f"  Total turns: {memory.get('total_turns', 0)}")
    if memory.get('avg_reward'):
        print(f"  Avg reward: {memory.get('avg_reward', 0):.4f}")
    
    # Search stats
    search = stats.get('search', {})
    if search:
        print(f"\n{c.YELLOW}Web Search:{c.RESET}")
        print(f"  Total searches: {search.get('total_searches', 0)}")
        print(f"  Cache size: {search.get('cache_size', 0)}")
        print(f"  Cache hits: {search.get('cache_hits', 0)}")
    
    # Tools stats
    tools = stats.get('tools', {})
    if tools:
        print(f"\n{c.YELLOW}Tools:{c.RESET}")
        print(f"  Registered: {tools.get('registered_count', 0)}")
        print(f"  Total calls: {tools.get('total_calls', 0)}")
    
    print()


def print_action_stats(dheera: Dheera, c: Colors):
    """Print action distribution."""
    try:
        action_stats = dheera.dqn.get_action_stats()
        
        print(f"\n{c.CYAN}{c.BOLD}🎯 Action Distribution:{c.RESET}")
        
        dist = action_stats.get('action_distribution', {})
        for action_name, data in dist.items():
            count = data.get('count', 0)
            pct = data.get('percentage', 0)
            bar_len = int(pct / 5)  # 20 chars max
            bar = "█" * bar_len + "░" * (20 - bar_len)
            
            # Color based on action type
            if 'SEARCH' in action_name:
                color = c.BRIGHT_CYAN
            elif 'TOOL' in action_name:
                color = c.BRIGHT_YELLOW
            elif 'DIRECT' in action_name:
                color = c.BRIGHT_GREEN
            else:
                color = c.WHITE
            
            print(f"  {color}{action_name:20}{c.RESET} {bar} {count:4} ({pct:5.1f}%)")
        
        print()
    except Exception as e:
        print(f"{c.RED}Error getting action stats: {e}{c.RESET}")


def test_search(dheera: Dheera, c: Colors):
    """Test web search functionality."""
    print(f"\n{c.CYAN}🔍 Testing web search...{c.RESET}")
    
    try:
        result = dheera.search("Python programming language")
        
        if result.get('success'):
            results = result.get('results', {})
            count = results.get('result_count', 0)
            print(f"{c.GREEN}✓ Search successful! Found {count} results.{c.RESET}")
            
            # Show first result
            items = results.get('results', [])
            if items:
                first = items[0]
                print(f"  First result: {first.get('title', 'N/A')[:50]}...")
        else:
            print(f"{c.YELLOW}⚠ Search returned no results{c.RESET}")
            
    except Exception as e:
        print(f"{c.RED}✗ Search test failed: {e}{c.RESET}")
    
    print()


def format_response(response: str, action_info: dict, debug: bool, c: Colors) -> str:
    """Format the response with optional debug info."""
    output = f"\n{c.BRIGHT_MAGENTA}{c.BOLD}Dheera:{c.RESET} {response}\n"
    
    if debug and action_info:
        action_name = action_info.get('action_name', 'UNKNOWN')
        search_performed = action_info.get('search_performed', False)
        
        # Color code action
        if 'SEARCH' in action_name:
            action_color = c.BRIGHT_CYAN
        elif 'TOOL' in action_name:
            action_color = c.BRIGHT_YELLOW
        elif 'DIRECT' in action_name:
            action_color = c.BRIGHT_GREEN
        else:
            action_color = c.WHITE
        
        debug_line = f"{c.DIM}  [Action: {action_color}{action_name}{c.RESET}{c.DIM}"
        
        if search_performed:
            debug_line += f" | 🔍 Search"
        
        if 'latency_ms' in action_info:
            debug_line += f" | {action_info['latency_ms']:.0f}ms"
        
        debug_line += f"]{c.RESET}"
        output += debug_line + "\n"
    
    return output


def main():
    """Main entry point."""
    args = parse_args()
    
    # Setup colors
    c = NoColors() if args.no_color else Colors()
    
    # Print banner
    print_banner(c)
    
    # Create Dheera instance
    print(f"{c.CYAN}Initializing Dheera...{c.RESET}\n")
    
    try:
        dheera = create_dheera(
            config_path=args.config if Path(args.config).exists() else None,
            debug=args.debug,
            fresh=args.fresh,
        )
    except Exception as e:
        print(f"{c.RED}Error creating Dheera: {e}{c.RESET}")
        print(f"{c.YELLOW}Trying with defaults...{c.RESET}")
        dheera = create_dheera(debug=args.debug, fresh=True)
    
    # Handle checkpoint loading
    if args.checkpoint:
        try:
            dheera.load_checkpoint(args.checkpoint)
            print(f"{c.GREEN}✓ Loaded checkpoint: {args.checkpoint}{c.RESET}")
        except Exception as e:
            print(f"{c.YELLOW}⚠ Could not load checkpoint: {e}{c.RESET}")
    
    # Enable/disable search
    if args.no_search:
        dheera.disable_search()
        print(f"{c.YELLOW}⚠ Web search disabled{c.RESET}")
    
    # Echo mode for testing
    if args.echo_mode:
        dheera.slm.use_echo_mode()
        print(f"{c.YELLOW}⚠ Echo mode enabled (no SLM){c.RESET}")
    
    # Check SLM availability
    slm_available = False
    try:
        slm_available = dheera.slm.is_available()
    except:
        pass
    
    if not slm_available:
        print(f"\n{c.YELLOW}⚠  Warning: SLM may not be available!{c.RESET}")
        print(f"   Provider: {args.provider}")
        print(f"   Model: {args.model}")
        
        if args.provider == "ollama":
            print(f"\n   Make sure Ollama is running:")
            print(f"   $ ollama serve")
            print(f"   $ ollama pull {args.model}")
        
        print(f"\n   Continuing anyway (responses may fail)...")
        print(f"   Use --echo-mode to test without SLM\n")
    else:
        print(f"{c.GREEN}✓ SLM available ({args.provider}/{args.model}){c.RESET}")
    
    # Test search if requested
    if args.search_test:
        test_search(dheera, c)
    
    # Print initial stats
    stats = dheera.get_stats()
    print(f"\n{c.CYAN}📊 Agent Status:{c.RESET}")
    print(f"   DQN Steps: {stats.get('dqn', {}).get('total_steps', 0)}")
    print(f"   Epsilon: {stats.get('dqn', {}).get('epsilon', 1.0):.3f}")
    print(f"   Episodes: {stats.get('memory', {}).get('total_episodes', 0)}")
    if not args.no_search:
        print(f"   Web Search: {c.GREEN}Enabled{c.RESET}")
    print()
    
    print(f"{c.DIM}Type /help for commands, /quit to exit{c.RESET}\n")
    
    # State
    debug_mode = args.debug
    force_search = False
    running = True
    
    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        nonlocal running
        running = False
        print(f"\n\n{c.YELLOW}Interrupted. Saving and exiting...{c.RESET}")
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Main chat loop
    while running:
        try:
            # Get user input
            user_input = input(f"{c.GREEN}{c.BOLD}You:{c.RESET} ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.startswith('/'):
                cmd = user_input.lower().split()[0]
                
                if cmd == '/quit' or cmd == '/exit' or cmd == '/q':
                    running = False
                    continue
                
                elif cmd == '/help' or cmd == '/h':
                    print_help(c)
                    continue
                
                elif cmd == '/stats':
                    print_stats(dheera, c)
                    continue
                
                elif cmd == '/actions':
                    print_action_stats(dheera, c)
                    continue
                
                elif cmd == '/search':
                    force_search = True
                    print(f"{c.CYAN}🔍 Search enabled for next message{c.RESET}")
                    continue
                
                elif cmd == '/save':
                    try:
                        dheera.save_checkpoint()
                        print(f"{c.GREEN}✓ Checkpoint saved{c.RESET}")
                    except Exception as e:
                        print(f"{c.RED}Error saving: {e}{c.RESET}")
                    continue
                
                elif cmd == '/debug':
                    debug_mode = not debug_mode
                    status = "enabled" if debug_mode else "disabled"
                    print(f"{c.CYAN}Debug mode {status}{c.RESET}")
                    continue
                
                elif cmd == '/clear':
                    dheera.clear_context()
                    print(f"{c.CYAN}Conversation cleared{c.RESET}")
                    continue
                
                elif cmd == '/test':
                    test_search(dheera, c)
                    continue
                
                else:
                    print(f"{c.YELLOW}Unknown command: {cmd}. Type /help for commands.{c.RESET}")
                    continue
            
            # Handle feedback
            if user_input in ['++', '+', '-', '--']:
                feedback_map = {'++': 1.0, '+': 0.5, '-': -0.5, '--': -1.0}
                reward = feedback_map[user_input]
                
                dheera.provide_feedback(reward)
                
                if reward > 0:
                    emoji = "👍" if reward == 0.5 else "🎉"
                    print(f"{c.GREEN}{emoji} Positive feedback recorded (+{reward}){c.RESET}")
                else:
                    emoji = "👎" if reward == -0.5 else "😞"
                    print(f"{c.RED}{emoji} Negative feedback recorded ({reward}){c.RESET}")
                continue
            
            # Process message
            try:
                response = dheera.process_message(
                    user_input,
                    force_search=force_search,
                )
                
                # Get action info
                action_info = getattr(dheera, 'last_action_info', {})
                
                # Format and print response
                formatted = format_response(response, action_info, debug_mode, c)
                print(formatted)
                
                # Reset force search
                force_search = False
                
            except Exception as e:
                print(f"{c.RED}Error: {e}{c.RESET}")
                if args.verbose:
                    import traceback
                    traceback.print_exc()
        
        except EOFError:
            running = False
        except KeyboardInterrupt:
            running = False
    
    # Cleanup
    print(f"\n{c.CYAN}Saving checkpoint...{c.RESET}")
    try:
        dheera.save_checkpoint()
        dheera.end_conversation(summary="Chat session ended by user")
    except Exception as e:
        print(f"{c.YELLOW}Warning: Could not save: {e}{c.RESET}")
    
    # Print final stats
    print(f"\n{c.CYAN}{c.BOLD}📊 Session Summary:{c.RESET}")
    final_stats = dheera.get_stats()
    print(f"   Interactions: {final_stats.get('interactions', 0)}")
    print(f"   DQN Steps: {final_stats.get('dqn', {}).get('total_steps', 0)}")
    print(f"   Episodes: {final_stats.get('memory', {}).get('total_episodes', 0)}")
    
    search_stats = final_stats.get('search', {})
    if search_stats:
        print(f"   Web Searches: {search_stats.get('total_searches', 0)}")
    
    print(f"\n{c.MAGENTA}👋 Goodbye! Dheera will remember this conversation.{c.RESET}\n")


if __name__ == "__main__":
    main()
