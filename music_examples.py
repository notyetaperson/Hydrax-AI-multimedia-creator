"""
Examples and Usage Guide for Hydax AI Music Generator
"""

import os
import numpy as np
from pathlib import Path

# Import our music generation modules
from music_generator import MusicGenerator
from music_styles import MusicStyleController
from music_processing import MusicProcessor

def example_basic_music_generation():
    """Example: Basic music generation"""
    print("🎵 Example 1: Basic Music Generation")
    print("=" * 50)
    
    # Initialize music generator
    generator = MusicGenerator()
    
    # Basic generation
    prompt = "A peaceful ambient track with soft piano and strings"
    music = generator.generate_music(
        prompt=prompt,
        duration=60,  # 1 minute for testing
        style="ambient",
        mood="calm"
    )
    
    if len(music) > 0:
        # Save the result
        generator.save_music(music, "examples/basic_music.wav")
        
        # Get music info
        info = generator.get_music_info(music)
        print(f"✓ Generated music: {info['duration']:.2f}s, {info['samples']} samples")
        print(f"✓ Music saved to: examples/basic_music.wav")
    else:
        print("✗ Failed to generate music")
    
    print()

def example_style_control():
    """Example: Music style and genre control"""
    print("🎨 Example 2: Style and Genre Control")
    print("=" * 50)
    
    # Initialize components
    generator = MusicGenerator()
    style_controller = MusicStyleController()
    
    prompt = "An energetic electronic dance track"
    
    # Test different styles
    styles = ['ambient', 'electronic', 'classical', 'jazz', 'rock', 'pop']
    
    for style in styles:
        print(f"Generating {style} style...")
        
        # Get style parameters
        style_params = style_controller.get_style_parameters(style)
        
        # Generate music with style
        music = generator.generate_music(
            prompt=prompt,
            duration=30,  # 30 seconds for testing
            style=style,
            tempo=style_params.get('tempo_range', (120, 140))[0]
        )
        
        if len(music) > 0:
            # Save the result
            filename = f"examples/style_{style}.wav"
            generator.save_music(music, filename)
            print(f"✓ Saved {style} style to: {filename}")
        else:
            print(f"✗ Failed to generate {style} style")
    
    print()

def example_genre_specific():
    """Example: Genre-specific music generation"""
    print("🎭 Example 3: Genre-Specific Generation")
    print("=" * 50)
    
    # Initialize components
    generator = MusicGenerator()
    style_controller = MusicStyleController()
    
    # Test different genres
    genres = ['house', 'trance', 'dubstep', 'ambient', 'orchestral']
    
    for genre in genres:
        print(f"Generating {genre} genre...")
        
        # Get genre parameters
        genre_params = style_controller.get_genre_parameters(genre)
        
        # Generate music with genre
        music = generator.generate_music(
            prompt=f"A {genre} track with characteristic elements",
            duration=45,  # 45 seconds for testing
            genre=genre,
            tempo=genre_params.get('bpm', 120),
            key=genre_params.get('key', 'C')
        )
        
        if len(music) > 0:
            # Save the result
            filename = f"examples/genre_{genre}.wav"
            generator.save_music(music, filename)
            print(f"✓ Saved {genre} genre to: {filename}")
        else:
            print(f"✗ Failed to generate {genre} genre")
    
    print()

def example_mood_control():
    """Example: Mood control in music generation"""
    print("😊 Example 4: Mood Control")
    print("=" * 50)
    
    # Initialize components
    generator = MusicGenerator()
    style_controller = MusicStyleController()
    
    prompt = "A beautiful melody that captures the essence of the moment"
    
    # Test different moods
    moods = ['happy', 'sad', 'energetic', 'calm', 'mysterious', 'romantic']
    
    for mood in moods:
        print(f"Generating {mood} mood...")
        
        # Get mood parameters
        mood_params = style_controller.get_mood_parameters(mood)
        
        # Generate music with mood
        music = generator.generate_music(
            prompt=prompt,
            duration=40,  # 40 seconds for testing
            style="ambient",
            mood=mood,
            tempo=int(120 * mood_params.get('tempo_modifier', 1.0))
        )
        
        if len(music) > 0:
            # Save the result
            filename = f"examples/mood_{mood}.wav"
            generator.save_music(music, filename)
            print(f"✓ Saved {mood} mood to: {filename}")
        else:
            print(f"✗ Failed to generate {mood} mood")
    
    print()

def example_music_processing():
    """Example: Music processing and enhancement"""
    print("🔧 Example 5: Music Processing and Enhancement")
    print("=" * 50)
    
    # Initialize components
    generator = MusicGenerator()
    processor = MusicProcessor()
    
    # Generate base music
    prompt = "A dynamic electronic track with multiple layers"
    base_music = generator.generate_music(
        prompt=prompt,
        duration=60,
        style="electronic"
    )
    
    if len(base_music) > 0:
        # Test different processing options
        processing_types = ["light", "medium", "full"]
        
        for processing_type in processing_types:
            print(f"Applying {processing_type} processing...")
            
            # Apply processing
            processed_music = processor.enhance_music(
                base_music,
                enhancement_type=processing_type
            )
            
            # Save the result
            filename = f"examples/processed_{processing_type}.wav"
            generator.save_music(processed_music, filename)
            print(f"✓ Saved {processing_type} processing to: {filename}")
        
        # Test mastering presets
        print("Testing mastering presets...")
        mastering_presets = ['streaming', 'club', 'radio', 'cinematic']
        
        for preset in mastering_presets:
            print(f"Applying {preset} mastering...")
            
            # Apply mastering
            mastered_music = processor.master_music(base_music, preset)
            
            # Save the result
            filename = f"examples/mastered_{preset}.wav"
            generator.save_music(mastered_music, filename)
            print(f"✓ Saved {preset} mastering to: {filename}")
        
        # Test effect chains
        print("Testing effect chains...")
        effect_chains = ['electronic', 'ambient', 'rock', 'jazz']
        
        for chain in effect_chains:
            print(f"Applying {chain} effect chain...")
            
            # Apply effect chain
            effected_music = processor.apply_effect_chain(base_music, chain)
            
            # Save the result
            filename = f"examples/effects_{chain}.wav"
            generator.save_music(effected_music, filename)
            print(f"✓ Saved {chain} effects to: {filename}")
        
        # Analyze quality
        quality_metrics = processor.analyze_music_quality(base_music)
        print(f"✓ Base music quality: {quality_metrics}")
        
    else:
        print("✗ Failed to generate base music")
    
    print()

def example_custom_style_creation():
    """Example: Creating custom styles"""
    print("✨ Example 6: Custom Style Creation")
    print("=" * 50)
    
    # Initialize style controller
    style_controller = MusicStyleController()
    
    # Create custom style
    custom_style_name = "dreamy_synthwave"
    custom_params = {
        'tempo_range': (80, 100),
        'key_signatures': ['C', 'G', 'D'],
        'time_signatures': ['4/4'],
        'instruments': ['synth', 'pad', 'bass', 'drums'],
        'dynamics': 'atmospheric',
        'harmony': 'modal',
        'rhythm': 'steady',
        'effects': ['reverb', 'delay', 'chorus', 'lowpass'],
        'structure': 'verse_chorus',
        'characteristics': ['nostalgic', 'synthetic', 'atmospheric', 'retro']
    }
    
    success = style_controller.create_custom_style(custom_style_name, custom_params)
    
    if success:
        print(f"✓ Created custom style: {custom_style_name}")
        
        # Test the custom style
        generator = MusicGenerator()
        music = generator.generate_music(
            prompt="A dreamy synthwave track with nostalgic vibes",
            duration=50,
            style=custom_style_name
        )
        
        if len(music) > 0:
            generator.save_music(music, f"examples/custom_{custom_style_name}.wav")
            print(f"✓ Custom style test saved to: examples/custom_{custom_style_name}.wav")
        else:
            print("✗ Failed to generate music with custom style")
    else:
        print("✗ Failed to create custom style")
    
    print()

def example_advanced_workflow():
    """Example: Advanced workflow combining multiple features"""
    print("🚀 Example 7: Advanced Workflow")
    print("=" * 50)
    
    # Initialize all components
    generator = MusicGenerator()
    style_controller = MusicStyleController()
    processor = MusicProcessor()
    
    # Step 1: Generate base music with specific style and mood
    prompt = "An epic cinematic orchestral piece with emotional depth and dramatic crescendos"
    print("Step 1: Generating base music...")
    
    base_music = generator.generate_music(
        prompt=prompt,
        duration=90,  # 1.5 minutes
        style="cinematic",
        mood="dramatic",
        tempo=90,
        key="C",
        instruments=['strings', 'brass', 'percussion', 'choir']
    )
    
    if len(base_music) > 0:
        print("✓ Base music generated")
        
        # Step 2: Apply style-specific processing
        print("Step 2: Applying cinematic processing...")
        processed_music = processor.apply_effect_chain(base_music, "cinematic")
        print("✓ Cinematic effects applied")
        
        # Step 3: Enhance the music
        print("Step 3: Enhancing music quality...")
        enhanced_music = processor.enhance_music(
            processed_music,
            enhancement_type="full",
            noise_reduction=True,
            dynamic_range=True,
            stereo_imaging=True,
            harmonic_enhancement=True
        )
        print("✓ Music enhancement applied")
        
        # Step 4: Master the final track
        print("Step 4: Mastering final track...")
        mastered_music = processor.master_music(enhanced_music, "cinematic")
        print("✓ Mastering applied")
        
        # Step 5: Save the final result
        print("Step 5: Saving final track...")
        generator.save_music(mastered_music, "examples/advanced_workflow.wav")
        print("✓ Advanced workflow completed! Saved to: examples/advanced_workflow.wav")
        
        # Step 6: Analyze the final music
        print("Step 6: Analyzing final music quality...")
        quality_metrics = processor.analyze_music_quality(mastered_music)
        print(f"✓ Final music quality: {quality_metrics}")
        
        # Step 7: Create variations
        print("Step 7: Creating style variations...")
        
        # Create ambient variation
        ambient_variation = processor.apply_effect_chain(mastered_music, "ambient")
        generator.save_music(ambient_variation, "examples/advanced_ambient_variation.wav")
        print("✓ Ambient variation saved")
        
        # Create electronic variation
        electronic_variation = processor.apply_effect_chain(mastered_music, "electronic")
        generator.save_music(electronic_variation, "examples/advanced_electronic_variation.wav")
        print("✓ Electronic variation saved")
        
    else:
        print("✗ Failed to generate base music")
    
    print()

def example_batch_generation():
    """Example: Batch music generation"""
    print("📦 Example 8: Batch Music Generation")
    print("=" * 50)
    
    # Initialize music generator
    generator = MusicGenerator()
    
    # List of prompts for batch generation
    prompts = [
        "A peaceful morning meditation track",
        "An energetic workout playlist opener",
        "A romantic dinner background music",
        "A mysterious thriller movie soundtrack",
        "A happy children's song melody"
    ]
    
    print(f"Generating {len(prompts)} tracks in batch...")
    
    # Generate all tracks
    results = []
    for i, prompt in enumerate(prompts):
        print(f"Processing track {i+1}/{len(prompts)}: {prompt[:50]}...")
        
        # Generate music
        music = generator.generate_music(
            prompt=prompt,
            duration=60,  # 1 minute each
            style="ambient" if "peaceful" in prompt else "electronic" if "energetic" in prompt else "cinematic"
        )
        
        if len(music) > 0:
            # Save the result
            filename = f"examples/batch_track_{i+1:02d}.wav"
            generator.save_music(music, filename)
            results.append(filename)
            print(f"✓ Saved track {i+1}: {filename}")
        else:
            print(f"✗ Failed to process track {i+1}")
    
    print(f"✓ Batch generation completed! Generated {len(results)} tracks.")
    print()

def example_music_analysis():
    """Example: Music analysis and style detection"""
    print("📊 Example 9: Music Analysis")
    print("=" * 50)
    
    # Initialize components
    generator = MusicGenerator()
    style_controller = MusicStyleController()
    processor = MusicProcessor()
    
    # Generate different styles for analysis
    styles_to_analyze = ['ambient', 'electronic', 'rock', 'jazz']
    
    for style in styles_to_analyze:
        print(f"Analyzing {style} style...")
        
        # Generate music
        music = generator.generate_music(
            prompt=f"A {style} track for analysis",
            duration=30,
            style=style
        )
        
        if len(music) > 0:
            # Analyze the music
            quality_metrics = processor.analyze_music_quality(music)
            print(f"✓ {style} quality metrics: {quality_metrics}")
            
            # Save for analysis
            generator.save_music(music, f"examples/analysis_{style}.wav")
        else:
            print(f"✗ Failed to generate {style} for analysis")
    
    print()

def create_music_examples_directory():
    """Create music examples directory if it doesn't exist"""
    examples_dir = Path("examples")
    examples_dir.mkdir(exist_ok=True)
    print(f"📁 Created music examples directory: {examples_dir}")

def main():
    """Run all music generation examples"""
    print("🎵 Hydax AI Music Generator - Examples and Usage Guide")
    print("=" * 60)
    print()
    
    # Create examples directory
    create_music_examples_directory()
    
    try:
        # Run all examples
        example_basic_music_generation()
        example_style_control()
        example_genre_specific()
        example_mood_control()
        example_music_processing()
        example_custom_style_creation()
        example_advanced_workflow()
        example_batch_generation()
        example_music_analysis()
        
        print("🎉 All music generation examples completed successfully!")
        print()
        print("📁 Check the 'examples' directory for generated music files.")
        print("🌐 Run 'python music_web_interface.py' to launch the music web interface.")
        print("📚 See README.md for more detailed documentation.")
        
    except Exception as e:
        print(f"❌ Error running music examples: {e}")
        print("💡 Make sure all dependencies are installed: pip install -r requirements.txt")

if __name__ == "__main__":
    main()
