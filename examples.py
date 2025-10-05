"""
Examples and Usage Guide for Hydax AI TTS Engine
"""

import os
import numpy as np
from pathlib import Path

# Import our TTS modules
from tts_engine import HydaxTTS
from voice_cloning import VoiceCloner
from emotion_control import EmotionController
from audio_processing import AudioProcessor

def example_basic_tts():
    """Example: Basic TTS synthesis"""
    print("🎤 Example 1: Basic TTS Synthesis")
    print("=" * 50)
    
    # Initialize TTS engine
    tts = HydaxTTS()
    
    # Basic synthesis
    text = "Hello, this is a test of the Hydax AI TTS engine. How does it sound?"
    audio = tts.synthesize(text)
    
    if len(audio) > 0:
        # Save the audio
        tts.save_audio(audio, "examples/basic_output.wav")
        
        # Get audio information
        info = tts.get_audio_info(audio)
        print(f"✓ Generated audio: {info['duration']:.2f}s, {info['samples']} samples")
        print(f"✓ Audio saved to: examples/basic_output.wav")
    else:
        print("✗ Failed to generate audio")
    
    print()

def example_emotion_control():
    """Example: Emotion and style control"""
    print("😊 Example 2: Emotion and Style Control")
    print("=" * 50)
    
    # Initialize components
    tts = HydaxTTS()
    emotion_controller = EmotionController()
    
    text = "I am so excited to demonstrate the emotion control features!"
    
    # Test different emotions
    emotions = ['happy', 'sad', 'angry', 'excited', 'calm', 'whisper']
    
    for emotion in emotions:
        print(f"Generating {emotion} emotion...")
        
        # Synthesize basic audio
        audio = tts.synthesize(text)
        
        if len(audio) > 0:
            # Apply emotion
            emotional_audio = emotion_controller.apply_emotion(audio, emotion, intensity=1.0)
            
            # Save the result
            filename = f"examples/emotion_{emotion}.wav"
            tts.save_audio(emotional_audio, filename)
            print(f"✓ Saved {emotion} emotion to: {filename}")
        else:
            print(f"✗ Failed to generate audio for {emotion}")
    
    print()

def example_voice_cloning():
    """Example: Voice cloning (requires audio file)"""
    print("🎭 Example 3: Voice Cloning")
    print("=" * 50)
    
    # Initialize voice cloner
    cloner = VoiceCloner()
    
    # Note: This example requires a reference audio file
    reference_audio = "examples/reference_voice.wav"
    
    if os.path.exists(reference_audio):
        print(f"Cloning voice from: {reference_audio}")
        
        # Clone the voice
        success = cloner.clone_voice_from_audio(
            voice_name="my_voice",
            audio_file=reference_audio,
            reference_text="This is a sample text for voice cloning."
        )
        
        if success:
            print("✓ Voice cloned successfully!")
            
            # Test the cloned voice
            test_text = "Hello, this is my cloned voice speaking!"
            audio = cloner.synthesize_with_cloned_voice(
                text=test_text,
                voice_name="my_voice",
                language="en",
                speed=1.0
            )
            
            if len(audio) > 0:
                # Save the result
                cloner.save_audio(audio, "examples/cloned_voice_output.wav")
                print("✓ Cloned voice test saved to: examples/cloned_voice_output.wav")
            else:
                print("✗ Failed to synthesize with cloned voice")
        else:
            print("✗ Voice cloning failed")
    else:
        print(f"⚠️  Reference audio file not found: {reference_audio}")
        print("   To test voice cloning, place a 3-30 second audio file at this location")
    
    print()

def example_audio_processing():
    """Example: Audio processing and enhancement"""
    print("🔧 Example 4: Audio Processing and Enhancement")
    print("=" * 50)
    
    # Initialize components
    tts = HydaxTTS()
    processor = AudioProcessor()
    
    text = "This audio will be processed and enhanced with various techniques."
    
    # Generate base audio
    audio = tts.synthesize(text)
    
    if len(audio) > 0:
        # Test different enhancement types
        enhancement_types = ["light", "medium", "full"]
        
        for enhancement_type in enhancement_types:
            print(f"Applying {enhancement_type} enhancement...")
            
            # Apply enhancement
            enhanced_audio = processor.enhance_audio(
                audio=audio,
                enhancement_type=enhancement_type
            )
            
            # Save the result
            filename = f"examples/enhanced_{enhancement_type}.wav"
            tts.save_audio(enhanced_audio, filename)
            print(f"✓ Saved {enhancement_type} enhancement to: {filename}")
        
        # Test audio effects
        print("Applying audio effects...")
        
        # Add reverb
        reverb_audio = processor.add_reverb(audio, reverb_type="hall", intensity=0.3)
        tts.save_audio(reverb_audio, "examples/with_reverb.wav")
        print("✓ Saved reverb effect to: examples/with_reverb.wav")
        
        # Apply EQ
        eq_audio = processor.apply_eq(audio, eq_type="speech")
        tts.save_audio(eq_audio, "examples/with_eq.wav")
        print("✓ Saved EQ effect to: examples/with_eq.wav")
        
        # Analyze audio quality
        quality_metrics = processor.analyze_audio_quality(audio)
        print(f"✓ Audio quality analysis: {quality_metrics}")
        
    else:
        print("✗ Failed to generate base audio")
    
    print()

def example_batch_processing():
    """Example: Batch processing multiple texts"""
    print("📦 Example 5: Batch Processing")
    print("=" * 50)
    
    # Initialize TTS engine
    tts = HydaxTTS()
    
    # List of texts to process
    texts = [
        "This is the first sentence in our batch processing example.",
        "Here is the second sentence with different content.",
        "The third sentence demonstrates batch processing capabilities.",
        "Finally, this is the fourth and last sentence in the batch."
    ]
    
    print(f"Processing {len(texts)} texts in batch...")
    
    # Process all texts
    results = tts.batch_synthesize(
        texts=texts,
        language="en",
        speed=1.0,
        pitch=1.0,
        model_type="coqui"
    )
    
    # Save all results
    for i, (text, audio) in enumerate(zip(texts, results)):
        if len(audio) > 0:
            filename = f"examples/batch_{i+1:02d}.wav"
            tts.save_audio(audio, filename)
            print(f"✓ Saved batch item {i+1}: {filename}")
        else:
            print(f"✗ Failed to process batch item {i+1}")
    
    print()

def example_custom_emotion():
    """Example: Creating custom emotions"""
    print("✨ Example 6: Custom Emotion Creation")
    print("=" * 50)
    
    # Initialize components
    tts = HydaxTTS()
    emotion_controller = EmotionController()
    
    # Create a custom emotion
    custom_emotion = "mysterious"
    success = emotion_controller.create_custom_emotion(
        name=custom_emotion,
        pitch_shift=-1.5,  # Lower pitch
        speed=0.8,         # Slower speech
        energy=0.7,        # Lower energy
        prosody="smooth",  # Smooth prosody
        breathing="deep",  # Deep breathing
        emphasis="low"     # Low emphasis
    )
    
    if success:
        print(f"✓ Created custom emotion: {custom_emotion}")
        
        # Test the custom emotion
        text = "This is a mysterious voice speaking in the shadows."
        audio = tts.synthesize(text)
        
        if len(audio) > 0:
            # Apply custom emotion
            mysterious_audio = emotion_controller.apply_emotion(audio, custom_emotion, intensity=1.0)
            
            # Save the result
            tts.save_audio(mysterious_audio, "examples/custom_mysterious.wav")
            print("✓ Custom emotion test saved to: examples/custom_mysterious.wav")
        else:
            print("✗ Failed to generate base audio")
    else:
        print("✗ Failed to create custom emotion")
    
    print()

def example_advanced_workflow():
    """Example: Advanced workflow combining multiple features"""
    print("🚀 Example 7: Advanced Workflow")
    print("=" * 50)
    
    # Initialize all components
    tts = HydaxTTS()
    cloner = VoiceCloner()
    emotion_controller = EmotionController()
    processor = AudioProcessor()
    
    # Step 1: Generate base audio with emotion
    text = "Welcome to the advanced workflow demonstration of the Hydax AI TTS engine."
    print("Step 1: Generating base audio with emotion...")
    
    base_audio = tts.synthesize(text)
    if len(base_audio) > 0:
        # Apply excitement emotion
        excited_audio = emotion_controller.apply_emotion(base_audio, "excited", intensity=0.8)
        print("✓ Applied excitement emotion")
        
        # Step 2: Apply speaking style
        print("Step 2: Applying dramatic speaking style...")
        dramatic_audio = emotion_controller.apply_style(excited_audio, "dramatic", intensity=0.6)
        print("✓ Applied dramatic style")
        
        # Step 3: Audio enhancement
        print("Step 3: Applying audio enhancement...")
        enhanced_audio = processor.enhance_audio(
            dramatic_audio,
            enhancement_type="full",
            noise_reduction=True,
            normalize_audio=True,
            remove_silence=True,
            enhance_clarity=True
        )
        print("✓ Applied full audio enhancement")
        
        # Step 4: Add reverb for dramatic effect
        print("Step 4: Adding hall reverb...")
        final_audio = processor.add_reverb(enhanced_audio, reverb_type="hall", intensity=0.2)
        print("✓ Added hall reverb")
        
        # Step 5: Save the final result
        print("Step 5: Saving final audio...")
        tts.save_audio(final_audio, "examples/advanced_workflow.wav")
        print("✓ Advanced workflow completed! Saved to: examples/advanced_workflow.wav")
        
        # Step 6: Analyze the final audio
        print("Step 6: Analyzing final audio quality...")
        quality_metrics = processor.analyze_audio_quality(final_audio)
        print(f"✓ Final audio quality: {quality_metrics}")
        
    else:
        print("✗ Failed to generate base audio")
    
    print()

def create_example_directory():
    """Create examples directory if it doesn't exist"""
    examples_dir = Path("examples")
    examples_dir.mkdir(exist_ok=True)
    print(f"📁 Created examples directory: {examples_dir}")

def main():
    """Run all examples"""
    print("🎤 Hydax AI TTS Engine - Examples and Usage Guide")
    print("=" * 60)
    print()
    
    # Create examples directory
    create_example_directory()
    
    try:
        # Run all examples
        example_basic_tts()
        example_emotion_control()
        example_voice_cloning()
        example_audio_processing()
        example_batch_processing()
        example_custom_emotion()
        example_advanced_workflow()
        
        print("🎉 All examples completed successfully!")
        print()
        print("📁 Check the 'examples' directory for generated audio files.")
        print("🌐 Run 'python web_interface.py' to launch the web interface.")
        print("📚 See README.md for more detailed documentation.")
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")
        print("💡 Make sure all dependencies are installed: pip install -r requirements.txt")

if __name__ == "__main__":
    main()
