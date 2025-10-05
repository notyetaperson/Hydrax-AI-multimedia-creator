#!/usr/bin/env python3
"""
Hydax AI Integrated Launcher - Complete Audio & Video Suite
Unified launcher for TTS, Music Generation, and Video Generation capabilities
"""

import sys
import os
import argparse
from pathlib import Path

def main():
    """Main integrated launcher function"""
    parser = argparse.ArgumentParser(
        description="Hydax AI Integrated Launcher - Complete Audio & Video Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python integrated_launcher.py                    # Launch integrated interface
  python integrated_launcher.py --tts              # Launch TTS interface only
  python integrated_launcher.py --music            # Launch Music interface only
  python integrated_launcher.py --video            # Launch Video interface only
  python integrated_launcher.py --examples         # Run all examples
  python integrated_launcher.py --test             # Run basic tests
  python integrated_launcher.py --port 8080        # Launch on custom port
        """
    )
    
    parser.add_argument(
        "--tts", 
        action="store_true", 
        help="Launch TTS interface only"
    )
    
    parser.add_argument(
        "--music", 
        action="store_true", 
        help="Launch Music interface only"
    )
    
    parser.add_argument(
        "--video", 
        action="store_true", 
        help="Launch Video interface only"
    )
    
    parser.add_argument(
        "--examples", 
        action="store_true", 
        help="Run all examples"
    )
    
    parser.add_argument(
        "--test", 
        action="store_true", 
        help="Run basic tests"
    )
    
    parser.add_argument(
        "--port", 
        type=int, 
        default=7860, 
        help="Port for web interface (default: 7860)"
    )
    
    parser.add_argument(
        "--host", 
        type=str, 
        default="127.0.0.1", 
        help="Host for web interface (default: 127.0.0.1)"
    )
    
    parser.add_argument(
        "--share", 
        action="store_true", 
        help="Share web interface publicly"
    )
    
    args = parser.parse_args()
    
    # If no specific action is requested, default to integrated interface
    if not any([args.tts, args.music, args.video, args.examples, args.test]):
        args.tts = True
        args.music = True
        args.video = True
    
    print("🎤🎵🎬 Hydax AI Integrated Launcher - Complete Audio & Video Suite")
    print("=" * 70)
    
    try:
        if args.examples:
            print("📚 Running all examples...")
            self._run_all_examples()
        
        elif args.test:
            print("🧪 Running basic tests...")
            self._run_basic_tests()
        
        elif args.tts and args.music and args.video:
            print("🚀 Launching integrated TTS + Music + Video interface...")
            self._launch_integrated_interface(args)
        
        elif args.tts:
            print("🎤 Launching TTS interface...")
            self._launch_tts_interface(args)
        
        elif args.music:
            print("🎵 Launching Music interface...")
            self._launch_music_interface(args)
        
        elif args.video:
            print("🎬 Launching Video interface...")
            self._launch_video_interface(args)
    
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

def _run_all_examples():
    """Run all examples for TTS, Music, and Video"""
    try:
        # Run TTS examples
        print("🎤 Running TTS examples...")
        from examples import main as run_tts_examples
        run_tts_examples()
        
        print("\n" + "="*50 + "\n")
        
        # Run Music examples
        print("🎵 Running Music examples...")
        from music_examples import main as run_music_examples
        run_music_examples()
        
        print("\n" + "="*50 + "\n")
        
        # Run Video examples
        print("🎬 Running Video examples...")
        from video_examples import VideoExamples
        video_examples = VideoExamples()
        video_results = video_examples.run_all_examples()
        print(f"✓ Video examples completed! Generated {len(video_results)} video files.")
        
        print("\n🎉 All examples completed successfully!")
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")

def _run_basic_tests():
    """Run basic tests for all systems"""
    try:
        # Test TTS
        print("🧪 Testing TTS system...")
        from tts_engine import HydaxTTS
        
        tts = HydaxTTS()
        audio = tts.synthesize("Hello, this is a test of the TTS system.")
        
        if len(audio) > 0:
            tts.save_audio(audio, "test_tts_output.wav")
            info = tts.get_audio_info(audio)
            print(f"✅ TTS test successful! Generated {info['duration']:.2f}s of audio")
        else:
            print("❌ TTS test failed")
        
        print("\n" + "-"*30 + "\n")
        
        # Test Music Generation
        print("🧪 Testing Music Generation system...")
        from music_generator import MusicGenerator
        
        generator = MusicGenerator()
        music = generator.generate_music(
            prompt="A simple test melody",
            duration=30
        )
        
        if len(music) > 0:
            generator.save_music(music, "test_music_output.wav")
            info = generator.get_music_info(music)
            print(f"✅ Music test successful! Generated {info['duration']:.2f}s of music")
        else:
            print("❌ Music test failed")
        
        print("\n" + "-"*30 + "\n")
        
        # Test Video Generation
        print("🧪 Testing Video Generation system...")
        from video_generator import VideoGenerator
        
        video_gen = VideoGenerator()
        video_path = video_gen.generate_video(
            prompt="A simple test video with basic animation",
            duration=30,
            style="cinematic",
            scene_type="abstract"
        )
        
        if video_path and os.path.exists(video_path):
            info = video_gen.get_video_info(video_path)
            print(f"✅ Video test successful! Generated {info.get('duration', 0):.2f}s video")
        else:
            print("❌ Video test failed")
        
        print("\n🎉 All tests completed!")
        
    except Exception as e:
        print(f"❌ Test error: {e}")

def _launch_integrated_interface(args):
    """Launch integrated interface with TTS, Music, and Video"""
    try:
        from integrated_web_interface import IntegratedWebInterface
        
        interface = IntegratedWebInterface()
        interface.launch(
            share=args.share,
            server_name=args.host,
            server_port=args.port
        )
        
    except Exception as e:
        print(f"❌ Failed to launch integrated interface: {e}")
        print("💡 Falling back to separate interfaces...")
        
        # Fallback to separate interfaces
        _launch_separate_interfaces(args)

def _launch_tts_interface(args):
    """Launch TTS interface only"""
    try:
        from web_interface import TTSWebInterface
        
        interface = TTSWebInterface()
        interface.launch(
            share=args.share,
            server_name=args.host,
            server_port=args.port
        )
        
    except Exception as e:
        print(f"❌ Failed to launch TTS interface: {e}")

def _launch_music_interface(args):
    """Launch Music interface only"""
    try:
        from music_web_interface import MusicWebInterface
        
        interface = MusicWebInterface()
        # Use different port for music interface to avoid conflicts
        music_port = args.port + 1 if args.port == 7860 else args.port
        interface.launch(
            share=args.share,
            server_name=args.host,
            server_port=music_port
        )
        
    except Exception as e:
        print(f"❌ Failed to launch Music interface: {e}")

def _launch_video_interface(args):
    """Launch Video interface only"""
    try:
        from video_web_interface import VideoWebInterface
        
        interface = VideoWebInterface()
        # Use different port for video interface to avoid conflicts
        video_port = args.port + 2 if args.port == 7860 else args.port
        interface.launch(
            share=args.share,
            server_name=args.host,
            server_port=video_port
        )
        
    except Exception as e:
        print(f"❌ Failed to launch Video interface: {e}")

def _launch_separate_interfaces(args):
    """Launch separate interfaces as fallback"""
    print("🔄 Launching separate interfaces...")
    
    # Launch TTS interface
    print(f"🎤 TTS Interface: http://{args.host}:{args.port}")
    _launch_tts_interface(args)
    
    # Launch Music interface
    music_port = args.port + 1
    print(f"🎵 Music Interface: http://{args.host}:{music_port}")
    _launch_music_interface(args)
    
    # Launch Video interface
    video_port = args.port + 2
    print(f"🎬 Video Interface: http://{args.host}:{video_port}")
    _launch_video_interface(args)

if __name__ == "__main__":
    sys.exit(main())
