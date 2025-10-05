#!/usr/bin/env python3
"""
Hydax AI TTS Engine Launcher
Simple launcher script for the TTS engine
"""

import sys
import os
import argparse
from pathlib import Path

def main():
    """Main launcher function"""
    parser = argparse.ArgumentParser(
        description="Hydax AI TTS Engine Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python launch.py                    # Launch web interface
  python launch.py --web              # Launch web interface
  python launch.py --examples         # Run examples
  python launch.py --test             # Run basic test
  python launch.py --web --port 8080  # Launch web interface on port 8080
        """
    )
    
    parser.add_argument(
        "--web", 
        action="store_true", 
        help="Launch web interface (default)"
    )
    
    parser.add_argument(
        "--examples", 
        action="store_true", 
        help="Run examples"
    )
    
    parser.add_argument(
        "--test", 
        action="store_true", 
        help="Run basic test"
    )
    
    parser.add_argument(
        "--port", 
        type=int, 
        default=7860, 
        help="Web interface port (default: 7860)"
    )
    
    parser.add_argument(
        "--host", 
        type=str, 
        default="127.0.0.1", 
        help="Web interface host (default: 127.0.0.1)"
    )
    
    parser.add_argument(
        "--share", 
        action="store_true", 
        help="Share web interface publicly"
    )
    
    args = parser.parse_args()
    
    # If no specific action is requested, default to web interface
    if not any([args.examples, args.test]):
        args.web = True
    
    print("🎤 Hydax AI TTS Engine Launcher")
    print("=" * 40)
    
    try:
        if args.web:
            print(f"🌐 Launching web interface on {args.host}:{args.port}")
            if args.share:
                print("🔗 Sharing interface publicly")
            
            from web_interface import TTSWebInterface
            web_interface = TTSWebInterface()
            web_interface.launch(
                share=args.share,
                server_name=args.host,
                server_port=args.port
            )
        
        elif args.examples:
            print("📚 Running examples...")
            from examples import main as run_examples
            run_examples()
        
        elif args.test:
            print("🧪 Running basic test...")
            from tts_engine import HydaxTTS
            
            tts = HydaxTTS()
            audio = tts.synthesize("Hello, this is a test of the Hydax AI TTS engine.")
            
            if len(audio) > 0:
                tts.save_audio(audio, "test_output.wav")
                info = tts.get_audio_info(audio)
                print(f"✅ Test successful! Generated {info['duration']:.2f}s of audio")
                print(f"📁 Audio saved to: test_output.wav")
            else:
                print("❌ Test failed - no audio generated")
                return 1
    
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure all dependencies are installed:")
        print("   pip install -r requirements.txt")
        return 1
    
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        return 0
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
