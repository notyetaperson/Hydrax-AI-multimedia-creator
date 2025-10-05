"""
Comprehensive Examples for Hydax AI Video Generator
"""

import os
import time
from pathlib import Path
from typing import List, Dict, Optional

# Import our video generation modules
from video_generator import VideoGenerator
from video_styles import VideoStyleController
from video_processing import VideoProcessor

class VideoExamples:
    """
    Comprehensive examples for the Hydax AI Video Generator
    """
    
    def __init__(self):
        """Initialize the video examples"""
        self.video_generator = VideoGenerator()
        self.style_controller = VideoStyleController()
        self.video_processor = VideoProcessor()
        
        # Create examples directory
        self.examples_dir = Path("video_examples")
        self.examples_dir.mkdir(exist_ok=True)
        
        # Example prompts and configurations
        self.example_configs = self._create_example_configs()
    
    def _create_example_configs(self) -> Dict:
        """Create example configurations for different use cases"""
        return {
            'nature_scenes': {
                'prompts': [
                    "A peaceful forest with tall trees and dappled sunlight filtering through the leaves",
                    "A serene mountain lake reflecting snow-capped peaks at sunset",
                    "A gentle waterfall cascading over moss-covered rocks in a hidden grove",
                    "A field of wildflowers swaying in a gentle breeze under a blue sky",
                    "A misty morning in a bamboo forest with rays of golden light"
                ],
                'style': 'cinematic',
                'scene_type': 'nature',
                'mood': 'peaceful',
                'duration': 120
            },
            'urban_life': {
                'prompts': [
                    "A bustling city street at night with neon lights and moving traffic",
                    "A busy subway station with people rushing to catch their trains",
                    "A rooftop view of a city skyline at sunset with lights beginning to twinkle",
                    "A street market with vendors selling colorful goods and people shopping",
                    "A modern office building with glass windows reflecting the city below"
                ],
                'style': 'documentary',
                'scene_type': 'urban',
                'mood': 'energetic',
                'duration': 90
            },
            'abstract_art': {
                'prompts': [
                    "Flowing geometric patterns in vibrant colors that morph and transform",
                    "Abstract shapes floating in a cosmic space with swirling particles",
                    "Colorful liquid forms that blend and separate in mesmerizing patterns",
                    "Geometric mandalas that rotate and change colors rhythmically",
                    "Fractal patterns that zoom in infinitely revealing new details"
                ],
                'style': 'artistic',
                'scene_type': 'abstract',
                'mood': 'creative',
                'duration': 150
            },
            'technology_futuristic': {
                'prompts': [
                    "A futuristic city with flying cars and holographic displays",
                    "A high-tech laboratory with glowing screens and robotic equipment",
                    "A cyberpunk street scene with neon signs and digital rain",
                    "A space station interior with advanced technology and starfield views",
                    "A virtual reality environment with floating data streams and interfaces"
                ],
                'style': 'futuristic',
                'scene_type': 'technology',
                'mood': 'futuristic',
                'duration': 180
            },
            'corporate_presentation': {
                'prompts': [
                    "A clean, modern office space with professional lighting and minimal design",
                    "A conference room with a large screen displaying business charts and graphs",
                    "A sleek corporate lobby with glass walls and contemporary furniture",
                    "A professional workspace with people collaborating on projects",
                    "A modern building exterior with clean lines and corporate branding"
                ],
                'style': 'corporate',
                'scene_type': 'urban',
                'mood': 'professional',
                'duration': 60
            },
            'social_media_content': {
                'prompts': [
                    "A trendy coffee shop with people working on laptops and socializing",
                    "A vibrant street art mural with people taking photos and selfies",
                    "A modern gym with people exercising and using high-tech equipment",
                    "A food market with colorful displays and people enjoying street food",
                    "A music festival scene with crowds dancing and colorful stage lights"
                ],
                'style': 'social_media',
                'scene_type': 'urban',
                'mood': 'energetic',
                'duration': 30
            }
        }
    
    def run_basic_examples(self) -> List[str]:
        """Run basic video generation examples"""
        print("🎬 Running Basic Video Generation Examples...")
        print("=" * 60)
        
        generated_videos = []
        
        # Example 1: Simple nature scene
        print("\n1. 🌲 Simple Nature Scene")
        print("   Prompt: A peaceful forest with tall trees and sunlight")
        
        try:
            video_path = self.video_generator.generate_video(
                prompt="A peaceful forest with tall trees and sunlight filtering through the leaves",
                duration=60,  # 1 minute for quick testing
                style="cinematic",
                scene_type="nature",
                mood="peaceful"
            )
            
            if video_path and os.path.exists(video_path):
                generated_videos.append(video_path)
                print(f"   ✓ Generated: {video_path}")
            else:
                print("   ✗ Failed to generate video")
                
        except Exception as e:
            print(f"   ✗ Error: {e}")
        
        # Example 2: Urban scene
        print("\n2. 🏙️ Urban Scene")
        print("   Prompt: A bustling city street with traffic and lights")
        
        try:
            video_path = self.video_generator.generate_video(
                prompt="A bustling city street with traffic and lights",
                duration=60,
                style="documentary",
                scene_type="urban",
                mood="energetic"
            )
            
            if video_path and os.path.exists(video_path):
                generated_videos.append(video_path)
                print(f"   ✓ Generated: {video_path}")
            else:
                print("   ✗ Failed to generate video")
                
        except Exception as e:
            print(f"   ✗ Error: {e}")
        
        # Example 3: Abstract art
        print("\n3. 🎨 Abstract Art")
        print("   Prompt: Flowing geometric patterns in vibrant colors")
        
        try:
            video_path = self.video_generator.generate_video(
                prompt="Flowing geometric patterns in vibrant colors that morph and transform",
                duration=60,
                style="artistic",
                scene_type="abstract",
                mood="creative"
            )
            
            if video_path and os.path.exists(video_path):
                generated_videos.append(video_path)
                print(f"   ✓ Generated: {video_path}")
            else:
                print("   ✗ Failed to generate video")
                
        except Exception as e:
            print(f"   ✗ Error: {e}")
        
        print(f"\n✓ Basic examples completed! Generated {len(generated_videos)} videos.")
        return generated_videos
    
    def run_style_comparison_examples(self) -> List[str]:
        """Run style comparison examples"""
        print("\n🎨 Running Style Comparison Examples...")
        print("=" * 60)
        
        generated_videos = []
        base_prompt = "A peaceful nature scene with trees and a gentle stream"
        
        styles_to_compare = ['cinematic', 'documentary', 'artistic', 'corporate']
        
        for style in styles_to_compare:
            print(f"\n📹 Testing {style} style...")
            
            try:
                video_path = self.video_generator.generate_video(
                    prompt=base_prompt,
                    duration=45,  # Shorter for comparison
                    style=style,
                    scene_type="nature",
                    mood="peaceful"
                )
                
                if video_path and os.path.exists(video_path):
                    generated_videos.append(video_path)
                    print(f"   ✓ Generated {style} style video: {video_path}")
                else:
                    print(f"   ✗ Failed to generate {style} style video")
                    
            except Exception as e:
                print(f"   ✗ Error with {style} style: {e}")
        
        print(f"\n✓ Style comparison completed! Generated {len(generated_videos)} videos.")
        return generated_videos
    
    def run_scene_type_examples(self) -> List[str]:
        """Run scene type examples"""
        print("\n🌍 Running Scene Type Examples...")
        print("=" * 60)
        
        generated_videos = []
        scene_configs = [
            {
                'scene': 'nature',
                'prompt': 'A serene mountain landscape with snow-capped peaks',
                'style': 'cinematic'
            },
            {
                'scene': 'urban',
                'prompt': 'A busy city intersection with cars and pedestrians',
                'style': 'documentary'
            },
            {
                'scene': 'abstract',
                'prompt': 'Swirling colors and geometric shapes in motion',
                'style': 'artistic'
            },
            {
                'scene': 'technology',
                'prompt': 'A futuristic laboratory with glowing screens and holograms',
                'style': 'futuristic'
            }
        ]
        
        for config in scene_configs:
            print(f"\n🎬 Testing {config['scene']} scene type...")
            
            try:
                video_path = self.video_generator.generate_video(
                    prompt=config['prompt'],
                    duration=60,
                    style=config['style'],
                    scene_type=config['scene'],
                    mood="neutral"
                )
                
                if video_path and os.path.exists(video_path):
                    generated_videos.append(video_path)
                    print(f"   ✓ Generated {config['scene']} scene: {video_path}")
                else:
                    print(f"   ✗ Failed to generate {config['scene']} scene")
                    
            except Exception as e:
                print(f"   ✗ Error with {config['scene']} scene: {e}")
        
        print(f"\n✓ Scene type examples completed! Generated {len(generated_videos)} videos.")
        return generated_videos
    
    def run_mood_examples(self) -> List[str]:
        """Run mood examples"""
        print("\n😊 Running Mood Examples...")
        print("=" * 60)
        
        generated_videos = []
        base_prompt = "A forest scene with trees and natural elements"
        
        moods_to_test = ['peaceful', 'energetic', 'mysterious', 'romantic', 'dramatic']
        
        for mood in moods_to_test:
            print(f"\n🎭 Testing {mood} mood...")
            
            try:
                video_path = self.video_generator.generate_video(
                    prompt=base_prompt,
                    duration=45,
                    style="cinematic",
                    scene_type="nature",
                    mood=mood
                )
                
                if video_path and os.path.exists(video_path):
                    generated_videos.append(video_path)
                    print(f"   ✓ Generated {mood} mood video: {video_path}")
                else:
                    print(f"   ✗ Failed to generate {mood} mood video")
                    
            except Exception as e:
                print(f"   ✗ Error with {mood} mood: {e}")
        
        print(f"\n✓ Mood examples completed! Generated {len(generated_videos)} videos.")
        return generated_videos
    
    def run_processing_examples(self, input_videos: List[str]) -> List[str]:
        """Run video processing examples"""
        print("\n🔧 Running Video Processing Examples...")
        print("=" * 60)
        
        if not input_videos:
            print("No input videos provided for processing examples.")
            return []
        
        processed_videos = []
        processing_styles = ['cinematic', 'documentary', 'artistic', 'corporate']
        
        for i, input_video in enumerate(input_videos[:2]):  # Process first 2 videos
            if not os.path.exists(input_video):
                continue
                
            print(f"\n🎬 Processing video {i+1}: {os.path.basename(input_video)}")
            
            for style in processing_styles:
                print(f"   🔧 Applying {style} processing...")
                
                try:
                    processed_path = self.video_processor.process_video(
                        video_path=input_video,
                        style=style
                    )
                    
                    if processed_path and os.path.exists(processed_path):
                        processed_videos.append(processed_path)
                        print(f"   ✓ Processed with {style} style: {processed_path}")
                    else:
                        print(f"   ✗ Failed to process with {style} style")
                        
                except Exception as e:
                    print(f"   ✗ Error processing with {style} style: {e}")
        
        print(f"\n✓ Processing examples completed! Generated {len(processed_videos)} processed videos.")
        return processed_videos
    
    def run_duration_examples(self) -> List[str]:
        """Run duration examples"""
        print("\n⏱️ Running Duration Examples...")
        print("=" * 60)
        
        generated_videos = []
        base_prompt = "A beautiful sunset over the ocean with waves gently lapping the shore"
        durations = [30, 60, 120, 180, 240]  # 30s to 4 minutes
        
        for duration in durations:
            print(f"\n⏰ Testing {duration} seconds duration...")
            
            try:
                video_path = self.video_generator.generate_video(
                    prompt=base_prompt,
                    duration=duration,
                    style="cinematic",
                    scene_type="nature",
                    mood="peaceful"
                )
                
                if video_path and os.path.exists(video_path):
                    generated_videos.append(video_path)
                    print(f"   ✓ Generated {duration}s video: {video_path}")
                else:
                    print(f"   ✗ Failed to generate {duration}s video")
                    
            except Exception as e:
                print(f"   ✗ Error with {duration}s duration: {e}")
        
        print(f"\n✓ Duration examples completed! Generated {len(generated_videos)} videos.")
        return generated_videos
    
    def run_resolution_examples(self) -> List[str]:
        """Run resolution examples"""
        print("\n📺 Running Resolution Examples...")
        print("=" * 60)
        
        generated_videos = []
        base_prompt = "A modern city skyline at night with twinkling lights"
        resolutions = [(1920, 1080), (1280, 720), (854, 480), (640, 360)]
        
        for resolution in resolutions:
            width, height = resolution
            print(f"\n📱 Testing {width}x{height} resolution...")
            
            try:
                video_path = self.video_generator.generate_video(
                    prompt=base_prompt,
                    duration=60,
                    style="documentary",
                    scene_type="urban",
                    mood="energetic",
                    resolution=resolution
                )
                
                if video_path and os.path.exists(video_path):
                    generated_videos.append(video_path)
                    print(f"   ✓ Generated {width}x{height} video: {video_path}")
                else:
                    print(f"   ✗ Failed to generate {width}x{height} video")
                    
            except Exception as e:
                print(f"   ✗ Error with {width}x{height} resolution: {e}")
        
        print(f"\n✓ Resolution examples completed! Generated {len(generated_videos)} videos.")
        return generated_videos
    
    def run_text_overlay_examples(self) -> List[str]:
        """Run text overlay examples"""
        print("\n📝 Running Text Overlay Examples...")
        print("=" * 60)
        
        generated_videos = []
        base_prompt = "A peaceful garden with flowers and butterflies"
        
        text_overlays = [
            "Welcome to Our Garden",
            "Nature's Beauty",
            "Peaceful Moments",
            "Garden Paradise"
        ]
        
        for i, text in enumerate(text_overlays):
            print(f"\n📝 Testing text overlay: '{text}'")
            
            try:
                video_path = self.video_generator.generate_video(
                    prompt=base_prompt,
                    duration=60,
                    style="cinematic",
                    scene_type="nature",
                    mood="peaceful",
                    text_overlay=text
                )
                
                if video_path and os.path.exists(video_path):
                    generated_videos.append(video_path)
                    print(f"   ✓ Generated video with text: {video_path}")
                else:
                    print(f"   ✗ Failed to generate video with text")
                    
            except Exception as e:
                print(f"   ✗ Error with text overlay: {e}")
        
        print(f"\n✓ Text overlay examples completed! Generated {len(generated_videos)} videos.")
        return generated_videos
    
    def run_batch_generation_example(self) -> List[str]:
        """Run batch generation example"""
        print("\n📦 Running Batch Generation Example...")
        print("=" * 60)
        
        # Get prompts from nature scenes config
        nature_config = self.example_configs['nature_scenes']
        prompts = nature_config['prompts'][:3]  # Use first 3 prompts
        
        print(f"Generating {len(prompts)} videos in batch...")
        
        generated_videos = []
        
        for i, prompt in enumerate(prompts):
            print(f"\n🎬 Batch item {i+1}/{len(prompts)}: {prompt[:50]}...")
            
            try:
                video_path = self.video_generator.generate_video(
                    prompt=prompt,
                    duration=90,
                    style=nature_config['style'],
                    scene_type=nature_config['scene_type'],
                    mood=nature_config['mood']
                )
                
                if video_path and os.path.exists(video_path):
                    generated_videos.append(video_path)
                    print(f"   ✓ Generated batch video {i+1}: {video_path}")
                else:
                    print(f"   ✗ Failed to generate batch video {i+1}")
                    
            except Exception as e:
                print(f"   ✗ Error with batch video {i+1}: {e}")
        
        print(f"\n✓ Batch generation completed! Generated {len(generated_videos)} videos.")
        return generated_videos
    
    def run_custom_style_examples(self) -> List[str]:
        """Run custom style examples"""
        print("\n✨ Running Custom Style Examples...")
        print("=" * 60)
        
        generated_videos = []
        
        # Create custom styles
        custom_styles = [
            {
                'name': 'vintage_warm',
                'params': {
                    'color_palette': 'vintage',
                    'lighting': 'soft',
                    'camera_movement': 'gentle',
                    'transitions': 'fade',
                    'effects': ['film_grain', 'vignette'],
                    'aspect_ratio': '4:3',
                    'framing': 'classic',
                    'characteristics': ['vintage', 'warm', 'nostalgic']
                }
            },
            {
                'name': 'neon_cyberpunk',
                'params': {
                    'color_palette': 'neon',
                    'lighting': 'harsh',
                    'camera_movement': 'robotic',
                    'transitions': 'digital',
                    'effects': ['glitch', 'hologram', 'neon_glow'],
                    'aspect_ratio': '16:9',
                    'framing': 'geometric',
                    'characteristics': ['futuristic', 'high_tech', 'cyberpunk']
                }
            }
        ]
        
        for style_config in custom_styles:
            print(f"\n🎨 Testing custom style: {style_config['name']}")
            
            # Create the custom style
            success = self.style_controller.create_custom_style(
                style_config['name'], 
                style_config['params']
            )
            
            if success:
                print(f"   ✓ Created custom style: {style_config['name']}")
                
                # Generate video with custom style
                try:
                    video_path = self.video_generator.generate_video(
                        prompt="A futuristic cityscape with neon lights and flying vehicles",
                        duration=60,
                        style=style_config['name'],
                        scene_type="technology",
                        mood="futuristic"
                    )
                    
                    if video_path and os.path.exists(video_path):
                        generated_videos.append(video_path)
                        print(f"   ✓ Generated video with custom style: {video_path}")
                    else:
                        print(f"   ✗ Failed to generate video with custom style")
                        
                except Exception as e:
                    print(f"   ✗ Error with custom style: {e}")
            else:
                print(f"   ✗ Failed to create custom style: {style_config['name']}")
        
        print(f"\n✓ Custom style examples completed! Generated {len(generated_videos)} videos.")
        return generated_videos
    
    def run_all_examples(self) -> Dict[str, List[str]]:
        """Run all video generation examples"""
        print("🎬 Running All Hydax AI Video Generator Examples")
        print("=" * 80)
        print("This will generate multiple videos to demonstrate all features.")
        print("The process may take several minutes depending on your system.")
        print("=" * 80)
        
        start_time = time.time()
        results = {}
        
        # Run all example categories
        try:
            results['basic'] = self.run_basic_examples()
            results['style_comparison'] = self.run_style_comparison_examples()
            results['scene_types'] = self.run_scene_type_examples()
            results['moods'] = self.run_mood_examples()
            results['durations'] = self.run_duration_examples()
            results['resolutions'] = self.run_resolution_examples()
            results['text_overlays'] = self.run_text_overlay_examples()
            results['batch_generation'] = self.run_batch_generation_example()
            results['custom_styles'] = self.run_custom_style_examples()
            
            # Run processing examples on some generated videos
            all_videos = []
            for video_list in results.values():
                all_videos.extend(video_list)
            
            if all_videos:
                results['processing'] = self.run_processing_examples(all_videos[:2])
            
        except Exception as e:
            print(f"Error running examples: {e}")
        
        # Calculate total time and results
        end_time = time.time()
        total_time = end_time - start_time
        
        # Count total videos generated
        total_videos = sum(len(video_list) for video_list in results.values())
        
        print("\n" + "=" * 80)
        print("🎉 ALL EXAMPLES COMPLETED!")
        print("=" * 80)
        print(f"⏱️  Total time: {total_time:.1f} seconds")
        print(f"🎬 Total videos generated: {total_videos}")
        print(f"📁 Output directory: {self.examples_dir}")
        
        # Print summary by category
        print("\n📊 Results Summary:")
        for category, videos in results.items():
            print(f"   {category}: {len(videos)} videos")
        
        # Print all generated video paths
        print(f"\n📹 Generated Videos:")
        for category, videos in results.items():
            if videos:
                print(f"\n{category.upper()}:")
                for video in videos:
                    print(f"   {video}")
        
        return results
    
    def create_example_report(self, results: Dict[str, List[str]]) -> str:
        """Create a detailed report of the examples"""
        try:
            report_path = self.examples_dir / "video_examples_report.md"
            
            with open(report_path, 'w') as f:
                f.write("# Hydax AI Video Generator - Examples Report\n\n")
                f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # Summary
                total_videos = sum(len(video_list) for video_list in results.values())
                f.write(f"## Summary\n\n")
                f.write(f"- Total videos generated: {total_videos}\n")
                f.write(f"- Categories tested: {len(results)}\n\n")
                
                # Results by category
                f.write("## Results by Category\n\n")
                for category, videos in results.items():
                    f.write(f"### {category.replace('_', ' ').title()}\n\n")
                    f.write(f"Videos generated: {len(videos)}\n\n")
                    
                    if videos:
                        f.write("Generated videos:\n")
                        for i, video in enumerate(videos, 1):
                            f.write(f"{i}. {video}\n")
                        f.write("\n")
                
                # Available features
                f.write("## Available Features\n\n")
                f.write("### Styles\n")
                for style in self.style_controller.get_available_styles():
                    f.write(f"- {style}\n")
                
                f.write("\n### Scene Types\n")
                for scene in self.style_controller.get_available_scenes():
                    f.write(f"- {scene}\n")
                
                f.write("\n### Moods\n")
                for mood in self.style_controller.get_available_moods():
                    f.write(f"- {mood}\n")
                
                f.write("\n### Processing Effects\n")
                for effect in self.video_processor.get_available_effects():
                    f.write(f"- {effect}\n")
            
            print(f"📄 Example report created: {report_path}")
            return str(report_path)
            
        except Exception as e:
            print(f"Error creating report: {e}")
            return None


# Main execution
if __name__ == "__main__":
    # Create video examples instance
    examples = VideoExamples()
    
    # Run all examples
    print("Starting comprehensive video generation examples...")
    results = examples.run_all_examples()
    
    # Create report
    report_path = examples.create_example_report(results)
    
    print(f"\n🎉 Video generation examples completed!")
    print(f"📄 Detailed report: {report_path}")
    print(f"📁 All videos saved in: {examples.examples_dir}")
