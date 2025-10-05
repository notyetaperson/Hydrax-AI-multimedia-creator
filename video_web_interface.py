"""
Web Interface for Hydax AI Video Generator
"""

import gradio as gr
import numpy as np
import tempfile
import os
from pathlib import Path
import json
from typing import Optional, Tuple, List

# Import our video generation modules
from video_generator import VideoGenerator
from video_styles import VideoStyleController
from video_processing import VideoProcessor

class VideoWebInterface:
    """
    Web interface for the Hydax AI Video Generator
    """
    
    def __init__(self):
        """Initialize the video web interface"""
        self.video_generator = VideoGenerator()
        self.style_controller = VideoStyleController()
        self.video_processor = VideoProcessor()
        
        # Create output directory
        self.output_dir = Path("video_outputs")
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize interface
        self.interface = self._create_interface()
    
    def _create_interface(self) -> gr.Blocks:
        """Create the Gradio interface"""
        with gr.Blocks(
            title="Hydax AI Video Generator",
            theme=gr.themes.Soft(),
            css="""
            .gradio-container {
                max-width: 1600px !important;
                margin: auto !important;
            }
            .main-header {
                text-align: center;
                margin-bottom: 30px;
            }
            .section-header {
                margin-top: 20px;
                margin-bottom: 10px;
                font-weight: bold;
                color: #2563eb;
            }
            .video-controls {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 20px;
                border-radius: 10px;
                color: white;
                margin-bottom: 20px;
            }
            """
        ) as interface:
            
            gr.HTML("""
            <div class="main-header">
                <h1>🎬 Hydax AI Video Generator</h1>
                <p>Create amazing videos from text prompts with AI-powered generation, style control, and professional effects</p>
            </div>
            """)
            
            with gr.Tabs():
                # Video Generation Tab
                with gr.Tab("🎬 Video Generation"):
                    self._create_video_generation_tab()
                
                # Style Control Tab
                with gr.Tab("🎨 Style & Scene Control"):
                    self._create_style_control_tab()
                
                # Video Processing Tab
                with gr.Tab("🔧 Video Processing"):
                    self._create_video_processing_tab()
                
                # Slideshow Creation Tab
                with gr.Tab("📸 Slideshow Creation"):
                    self._create_slideshow_tab()
                
                # Batch Generation Tab
                with gr.Tab("📦 Batch Generation"):
                    self._create_batch_generation_tab()
                
                # Advanced Settings Tab
                with gr.Tab("⚙️ Advanced Settings"):
                    self._create_advanced_settings_tab()
        
        return interface
    
    def _create_video_generation_tab(self):
        """Create the main video generation tab"""
        with gr.Row():
            with gr.Column(scale=2):
                gr.HTML('<div class="video-controls"><h3>🎬 Video Generation</h3></div>')
                
                # Prompt input
                prompt = gr.Textbox(
                    label="Video Prompt",
                    placeholder="Describe the video you want to generate... (e.g., 'A peaceful nature scene with trees and sunset', 'An energetic city street with people and traffic')",
                    lines=3,
                    max_lines=5
                )
                
                # Duration control
                with gr.Row():
                    duration = gr.Slider(
                        minimum=30,
                        maximum=300,
                        value=120,
                        step=30,
                        label="Duration (seconds)",
                        info="2-5 minutes recommended"
                    )
                    
                    duration_preset = gr.Dropdown(
                        choices=["30s", "60s", "120s", "180s", "240s", "300s"],
                        value="120s",
                        label="Quick Duration"
                    )
                
                # Style and scene selection
                with gr.Row():
                    style = gr.Dropdown(
                        choices=self.style_controller.get_available_styles(),
                        value="cinematic",
                        label="Video Style"
                    )
                    
                    scene_type = gr.Dropdown(
                        choices=self.style_controller.get_available_scenes(),
                        value="nature",
                        label="Scene Type"
                    )
                
                # Mood and resolution
                with gr.Row():
                    mood = gr.Dropdown(
                        choices=self.style_controller.get_available_moods(),
                        value="peaceful",
                        label="Mood"
                    )
                    
                    resolution = gr.Dropdown(
                        choices=["1920x1080", "1280x720", "854x480", "640x360"],
                        value="1920x1080",
                        label="Resolution"
                    )
                
                # Video parameters
                with gr.Accordion("🎬 Video Parameters", open=False):
                    with gr.Row():
                        fps = gr.Slider(
                            minimum=24,
                            maximum=60,
                            value=30,
                            step=6,
                            label="FPS"
                        )
                        
                        include_audio = gr.Checkbox(
                            value=True,
                            label="Include Audio"
                        )
                    
                    background_music = gr.Textbox(
                        label="Background Music Prompt (Optional)",
                        placeholder="Describe background music for the video..."
                    )
                    
                    voice_over = gr.Textbox(
                        label="Voice Over Text (Optional)",
                        placeholder="Text to be spoken over the video..."
                    )
                    
                    text_overlay = gr.Textbox(
                        label="Text Overlay (Optional)",
                        placeholder="Text to display on the video..."
                    )
                
                # Generate button
                generate_btn = gr.Button("🎬 Generate Video", variant="primary", size="lg")
                
            with gr.Column(scale=1):
                gr.Markdown("### 🎥 Output")
                
                # Video output
                video_output = gr.Video(
                    label="Generated Video",
                    format="mp4"
                )
                
                # Download and play controls
                with gr.Row():
                    download_btn = gr.Button("💾 Download", size="sm")
                    play_btn = gr.Button("▶️ Play", size="sm")
                    stop_btn = gr.Button("⏹️ Stop", size="sm")
                
                # Video information
                gr.Markdown("### 📊 Video Info")
                video_info = gr.JSON(label="Video Information")
                
                # Generation status
                status = gr.Textbox(
                    label="Generation Status",
                    interactive=False,
                    value="Ready to generate video..."
                )
        
        # Event handlers
        duration_preset.change(
            fn=self._update_duration_from_preset,
            inputs=[duration_preset],
            outputs=[duration]
        )
        
        generate_btn.click(
            fn=self._generate_video,
            inputs=[prompt, duration, style, scene_type, mood, resolution, fps, include_audio, background_music, voice_over, text_overlay],
            outputs=[video_output, video_info, status]
        )
    
    def _create_style_control_tab(self):
        """Create the style control tab"""
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🎨 Style & Scene Control")
                
                # Style selection
                style = gr.Dropdown(
                    choices=self.style_controller.get_available_styles(),
                    value="cinematic",
                    label="Video Style"
                )
                
                # Scene selection
                scene_type = gr.Dropdown(
                    choices=self.style_controller.get_available_scenes(),
                    value="nature",
                    label="Scene Type"
                )
                
                # Mood selection
                mood = gr.Dropdown(
                    choices=self.style_controller.get_available_moods(),
                    value="peaceful",
                    label="Mood"
                )
                
                # Style parameters display
                gr.Markdown("### 📋 Style Parameters")
                style_params = gr.JSON(label="Current Style Parameters")
                
                # Custom style creation
                gr.Markdown("### ✨ Custom Style")
                
                custom_style_name = gr.Textbox(
                    label="Custom Style Name",
                    placeholder="Enter name for custom style..."
                )
                
                with gr.Accordion("Custom Style Parameters", open=False):
                    custom_color_palette = gr.Dropdown(
                        choices=["warm", "cool", "vibrant", "monochrome", "vintage"],
                        label="Color Palette"
                    )
                    
                    custom_lighting = gr.Dropdown(
                        choices=["natural", "dramatic", "soft", "harsh", "artistic"],
                        label="Lighting"
                    )
                    
                    custom_camera_movement = gr.Dropdown(
                        choices=["smooth", "steady", "dynamic", "minimal", "experimental"],
                        label="Camera Movement"
                    )
                
                create_custom_style_btn = gr.Button("✨ Create Custom Style", variant="secondary")
                
            with gr.Column(scale=2):
                gr.Markdown("### 🎬 Test Style")
                
                # Test prompt
                test_prompt = gr.Textbox(
                    label="Test Prompt",
                    placeholder="Enter a prompt to test the current style settings...",
                    lines=2
                )
                
                # Test duration
                test_duration = gr.Slider(
                    minimum=30,
                    maximum=120,
                    value=60,
                    step=30,
                    label="Test Duration (seconds)"
                )
                
                # Test generation
                test_generate_btn = gr.Button("🎬 Test Style", variant="primary")
                
                # Test output
                test_video_output = gr.Video(
                    label="Style Test Output",
                    format="mp4"
                )
                
                # Style comparison
                gr.Markdown("### 🔄 Style Comparison")
                
                with gr.Row():
                    compare_style1 = gr.Dropdown(
                        choices=self.style_controller.get_available_styles(),
                        value="cinematic",
                        label="Style 1"
                    )
                    
                    compare_style2 = gr.Dropdown(
                        choices=self.style_controller.get_available_styles(),
                        value="documentary",
                        label="Style 2"
                    )
                
                compare_prompt = gr.Textbox(
                    label="Comparison Prompt",
                    placeholder="Same prompt for both styles...",
                    lines=2
                )
                
                compare_btn = gr.Button("🔄 Compare Styles", variant="secondary")
                
                with gr.Row():
                    compare_video1 = gr.Video(label="Style 1 Output", format="mp4")
                    compare_video2 = gr.Video(label="Style 2 Output", format="mp4")
        
        # Event handlers
        style.change(
            fn=self._update_style_params,
            inputs=[style],
            outputs=[style_params]
        )
        
        create_custom_style_btn.click(
            fn=self._create_custom_style,
            inputs=[custom_style_name, custom_color_palette, custom_lighting, custom_camera_movement],
            outputs=[style, style_params]
        )
        
        test_generate_btn.click(
            fn=self._test_style,
            inputs=[test_prompt, test_duration, style, mood],
            outputs=[test_video_output]
        )
        
        compare_btn.click(
            fn=self._compare_styles,
            inputs=[compare_prompt, compare_style1, compare_style2],
            outputs=[compare_video1, compare_video2]
        )
    
    def _create_video_processing_tab(self):
        """Create the video processing tab"""
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🔧 Video Processing")
                
                # Input video
                input_video = gr.Video(
                    label="Input Video",
                    format="mp4",
                    info="Upload video to process"
                )
                
                # Processing options
                gr.Markdown("### 🎛️ Processing Options")
                
                # Style presets
                processing_style = gr.Dropdown(
                    choices=self.video_processor.get_available_effects(),
                    value="cinematic",
                    label="Processing Style"
                )
                
                # Color presets
                color_preset = gr.Dropdown(
                    choices=self.video_processor.get_available_color_presets(),
                    value="warm",
                    label="Color Preset"
                )
                
                # Enhancement options
                with gr.Accordion("Enhancement Options", open=False):
                    color_grading = gr.Checkbox(value=True, label="Color Grading")
                    stabilization = gr.Checkbox(value=True, label="Stabilization")
                    film_grain = gr.Checkbox(value=False, label="Film Grain")
                    vignette = gr.Checkbox(value=False, label="Vignette")
                    sharpening = gr.Checkbox(value=True, label="Sharpening")
                
                # Process button
                process_btn = gr.Button("🔧 Process Video", variant="primary")
                
            with gr.Column(scale=2):
                gr.Markdown("### 🎥 Processed Output")
                
                # Processed video
                processed_video = gr.Video(
                    label="Processed Video",
                    format="mp4"
                )
                
                # Download controls
                with gr.Row():
                    download_processed_btn = gr.Button("💾 Download Processed", size="sm")
                    compare_btn = gr.Button("🔄 Compare", size="sm")
                
                # A/B comparison
                with gr.Row():
                    original_video = gr.Video(label="Original Video", format="mp4")
                    processed_video_compare = gr.Video(label="Processed Video", format="mp4")
        
        # Event handlers
        process_btn.click(
            fn=self._process_video,
            inputs=[input_video, processing_style, color_preset, color_grading, stabilization, film_grain, vignette, sharpening],
            outputs=[processed_video]
        )
        
        compare_btn.click(
            fn=self._compare_video,
            inputs=[input_video, processed_video],
            outputs=[original_video, processed_video_compare]
        )
    
    def _create_slideshow_tab(self):
        """Create the slideshow creation tab"""
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📸 Slideshow Creation")
                
                # Image upload
                image_files = gr.File(
                    label="Upload Images",
                    file_count="multiple",
                    file_types=["image"]
                )
                
                # Slideshow settings
                with gr.Row():
                    duration_per_image = gr.Slider(
                        minimum=1.0,
                        maximum=10.0,
                        value=3.0,
                        step=0.5,
                        label="Duration per Image (seconds)"
                    )
                    
                    transition_type = gr.Dropdown(
                        choices=self.video_processor.get_available_transitions(),
                        value="fade",
                        label="Transition Type"
                    )
                
                # Background music
                background_music = gr.Audio(
                    label="Background Music (Optional)",
                    type="filepath"
                )
                
                # Create slideshow button
                create_slideshow_btn = gr.Button("📸 Create Slideshow", variant="primary")
                
            with gr.Column(scale=2):
                gr.Markdown("### 🎬 Slideshow Output")
                
                # Slideshow output
                slideshow_output = gr.Video(
                    label="Generated Slideshow",
                    format="mp4"
                )
                
                # Download controls
                with gr.Row():
                    download_slideshow_btn = gr.Button("💾 Download Slideshow", size="sm")
                    play_slideshow_btn = gr.Button("▶️ Play Slideshow", size="sm")
        
        # Event handlers
        create_slideshow_btn.click(
            fn=self._create_slideshow,
            inputs=[image_files, duration_per_image, transition_type, background_music],
            outputs=[slideshow_output]
        )
    
    def _create_batch_generation_tab(self):
        """Create the batch generation tab"""
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📦 Batch Video Generation")
                
                # Batch prompts
                batch_prompts = gr.Textbox(
                    label="Video Prompts (one per line)",
                    placeholder="Enter multiple video prompts, one per line...",
                    lines=10
                )
                
                # Batch settings
                with gr.Row():
                    batch_duration = gr.Slider(
                        minimum=30,
                        maximum=300,
                        value=120,
                        step=30,
                        label="Duration per video (seconds)"
                    )
                    
                    batch_style = gr.Dropdown(
                        choices=self.style_controller.get_available_styles(),
                        value="cinematic",
                        label="Style"
                    )
                
                with gr.Row():
                    batch_scene = gr.Dropdown(
                        choices=self.style_controller.get_available_scenes(),
                        value="nature",
                        label="Scene Type"
                    )
                    
                    batch_mood = gr.Dropdown(
                        choices=self.style_controller.get_available_moods(),
                        value="peaceful",
                        label="Mood"
                    )
                
                # Batch processing options
                with gr.Accordion("Batch Processing Options", open=False):
                    batch_processing = gr.Dropdown(
                        choices=self.video_processor.get_available_effects(),
                        value="cinematic",
                        label="Processing Style"
                    )
                    
                    batch_resolution = gr.Dropdown(
                        choices=["1920x1080", "1280x720", "854x480"],
                        value="1280x720",
                        label="Resolution"
                    )
                
                # Batch generation button
                batch_generate_btn = gr.Button("📦 Generate Batch", variant="primary", size="lg")
                
            with gr.Column(scale=2):
                gr.Markdown("### 📊 Batch Progress")
                
                # Progress tracking
                batch_progress = gr.Progress()
                batch_status = gr.Textbox(
                    label="Batch Status",
                    interactive=False,
                    lines=3
                )
                
                # Batch results
                gr.Markdown("### 🎬 Batch Results")
                
                batch_results = gr.File(
                    label="Download Batch Results",
                    file_count="multiple"
                )
                
                # Individual video previews
                gr.Markdown("### 🎥 Video Previews")
                
                with gr.Row():
                    video1 = gr.Video(label="Video 1", format="mp4")
                    video2 = gr.Video(label="Video 2", format="mp4")
                
                with gr.Row():
                    video3 = gr.Video(label="Video 3", format="mp4")
                    video4 = gr.Video(label="Video 4", format="mp4")
        
        # Event handlers
        batch_generate_btn.click(
            fn=self._generate_batch,
            inputs=[batch_prompts, batch_duration, batch_style, batch_scene, batch_mood, batch_processing, batch_resolution],
            outputs=[batch_results, batch_status, video1, video2, video3, video4]
        )
    
    def _create_advanced_settings_tab(self):
        """Create the advanced settings tab"""
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Advanced Settings")
                
                # Model settings
                gr.Markdown("#### 🤖 Model Settings")
                
                device = gr.Dropdown(
                    choices=["auto", "cpu", "cuda"],
                    value="auto",
                    label="Device"
                )
                
                fps = gr.Slider(
                    minimum=24,
                    maximum=60,
                    value=30,
                    step=6,
                    label="Default FPS"
                )
                
                # Generation settings
                gr.Markdown("#### 🎬 Generation Settings")
                
                max_duration = gr.Slider(
                    minimum=60,
                    maximum=600,
                    value=300,
                    step=30,
                    label="Max Duration (seconds)"
                )
                
                generation_quality = gr.Dropdown(
                    choices=["fast", "balanced", "high_quality"],
                    value="balanced",
                    label="Generation Quality"
                )
                
                # Processing settings
                gr.Markdown("#### 🔧 Processing Settings")
                
                enable_processing = gr.Checkbox(value=True, label="Enable Auto Processing")
                enable_effects = gr.Checkbox(value=True, label="Enable Effects")
                
                # Save settings
                save_settings_btn = gr.Button("💾 Save Settings", variant="secondary")
                
            with gr.Column(scale=2):
                gr.Markdown("### 📊 System Information")
                
                # System info
                system_info = gr.JSON(label="System Information")
                
                gr.Markdown("### 🎨 Available Styles & Scenes")
                
                with gr.Row():
                    available_styles = gr.Textbox(
                        label="Available Styles",
                        value=", ".join(self.style_controller.get_available_styles()),
                        interactive=False,
                        lines=3
                    )
                    
                    available_scenes = gr.Textbox(
                        label="Available Scenes",
                        value=", ".join(self.style_controller.get_available_scenes()),
                        interactive=False,
                        lines=3
                    )
                
                gr.Markdown("### 🔧 Available Processing Options")
                
                with gr.Row():
                    processing_effects = gr.Textbox(
                        label="Processing Effects",
                        value=", ".join(self.video_processor.get_available_effects()),
                        interactive=False,
                        lines=2
                    )
                    
                    color_presets = gr.Textbox(
                        label="Color Presets",
                        value=", ".join(self.video_processor.get_available_color_presets()),
                        interactive=False,
                        lines=2
                    )
                
                # Export/Import settings
                gr.Markdown("### 📤 Export/Import")
                
                with gr.Row():
                    export_settings_btn = gr.Button("📤 Export Settings")
                    import_settings_btn = gr.Button("📥 Import Settings")
                
                settings_file = gr.File(label="Settings File")
        
        # Event handlers
        save_settings_btn.click(
            fn=self._save_settings,
            inputs=[device, fps, max_duration, generation_quality, enable_processing, enable_effects],
            outputs=[system_info]
        )
        
        export_settings_btn.click(
            fn=self._export_settings,
            outputs=[settings_file]
        )
        
        import_settings_btn.click(
            fn=self._import_settings,
            inputs=[settings_file],
            outputs=[system_info]
        )
    
    # Event handler methods
    def _update_duration_from_preset(self, preset: str) -> int:
        """Update duration from preset selection"""
        duration_map = {
            "30s": 30,
            "60s": 60,
            "120s": 120,
            "180s": 180,
            "240s": 240,
            "300s": 300
        }
        return duration_map.get(preset, 120)
    
    def _generate_video(self, prompt: str, duration: float, style: str, scene_type: str, 
                       mood: str, resolution: str, fps: int, include_audio: bool, 
                       background_music: str, voice_over: str, text_overlay: str) -> Tuple[str, dict, str]:
        """Generate video from parameters"""
        try:
            if not prompt.strip():
                return None, {"error": "No prompt provided"}, "Error: No prompt provided"
            
            # Parse resolution
            width, height = map(int, resolution.split('x'))
            
            status = f"Generating {duration:.0f}s video with {style} style..."
            
            # Generate video
            video_path = self.video_generator.generate_video(
                prompt=prompt,
                duration=duration,
                style=style,
                scene_type=scene_type,
                resolution=(width, height),
                fps=fps,
                include_audio=include_audio,
                background_music=background_music,
                voice_over=voice_over,
                text_overlay=text_overlay
            )
            
            if video_path and os.path.exists(video_path):
                # Get video info
                info = self.video_generator.get_video_info(video_path)
                status = f"✓ Generated {info.get('duration', 0):.1f}s video successfully!"
                
                return video_path, info, status
            else:
                return None, {"error": "Failed to generate video"}, "Error: Failed to generate video"
                
        except Exception as e:
            return None, {"error": str(e)}, f"Error: {str(e)}"
    
    def _update_style_params(self, style: str) -> dict:
        """Update style parameters display"""
        try:
            params = self.style_controller.get_style_parameters(style)
            return params
        except Exception as e:
            return {"error": str(e)}
    
    def _create_custom_style(self, name: str, color_palette: str, lighting: str, camera_movement: str) -> Tuple[List[str], dict]:
        """Create a custom style"""
        try:
            if not name.strip():
                return gr.update(), {}
            
            # Create custom style parameters
            custom_params = {
                'color_palette': color_palette,
                'lighting': lighting,
                'camera_movement': camera_movement,
                'transitions': 'fade',
                'effects': ['color_grading'],
                'aspect_ratio': '16:9',
                'framing': 'centered',
                'characteristics': ['custom', 'unique', 'personalized']
            }
            
            # Create the custom style
            success = self.style_controller.create_custom_style(name, custom_params)
            
            if success:
                # Update style dropdown
                new_choices = self.style_controller.get_available_styles()
                return gr.update(choices=new_choices, value=name), custom_params
            else:
                return gr.update(), {}
                
        except Exception as e:
            print(f"Custom style creation failed: {e}")
            return gr.update(), {}
    
    def _test_style(self, prompt: str, duration: float, style: str, mood: str) -> str:
        """Test a style with a prompt"""
        try:
            if not prompt.strip():
                return None
            
            video_path = self.video_generator.generate_video(
                prompt=prompt,
                duration=duration,
                style=style,
                scene_type="abstract"
            )
            
            return video_path if video_path and os.path.exists(video_path) else None
                
        except Exception as e:
            print(f"Style test failed: {e}")
            return None
    
    def _compare_styles(self, prompt: str, style1: str, style2: str) -> Tuple[str, str]:
        """Compare two styles with the same prompt"""
        try:
            if not prompt.strip():
                return None, None
            
            # Generate with style 1
            video1 = self.video_generator.generate_video(
                prompt=prompt,
                duration=60,
                style=style1
            )
            
            # Generate with style 2
            video2 = self.video_generator.generate_video(
                prompt=prompt,
                duration=60,
                style=style2
            )
            
            result1 = video1 if video1 and os.path.exists(video1) else None
            result2 = video2 if video2 and os.path.exists(video2) else None
            
            return result1, result2
            
        except Exception as e:
            print(f"Style comparison failed: {e}")
            return None, None
    
    def _process_video(self, video_path: str, processing_style: str, color_preset: str,
                      color_grading: bool, stabilization: bool, film_grain: bool,
                      vignette: bool, sharpening: bool) -> str:
        """Process video with selected options"""
        try:
            if video_path is None:
                return None
            
            # Create custom effects based on checkboxes
            custom_effects = {
                'color_grading': {'enabled': color_grading, 'preset': color_preset},
                'stabilization': {'enabled': stabilization},
                'film_grain': {'enabled': film_grain},
                'vignette': {'enabled': vignette},
                'sharpening': {'enabled': sharpening}
            }
            
            # Process video
            processed_path = self.video_processor.process_video(
                video_path=video_path,
                style=processing_style,
                custom_effects=custom_effects
            )
            
            return processed_path
            
        except Exception as e:
            print(f"Video processing failed: {e}")
            return None
    
    def _compare_video(self, original: str, processed: str) -> Tuple[str, str]:
        """Compare original and processed video"""
        return original, processed
    
    def _create_slideshow(self, image_files: List[str], duration_per_image: float, 
                         transition_type: str, background_music: str) -> str:
        """Create slideshow from images"""
        try:
            if not image_files:
                return None
            
            # Create slideshow
            slideshow_path = self.video_processor.create_slideshow(
                image_paths=image_files,
                duration_per_image=duration_per_image,
                transition_type=transition_type,
                background_music=background_music
            )
            
            return slideshow_path
            
        except Exception as e:
            print(f"Slideshow creation failed: {e}")
            return None
    
    def _generate_batch(self, prompts: str, duration: float, style: str, scene_type: str, 
                       mood: str, processing_style: str, resolution: str) -> Tuple[List[str], str, str, str, str, str]:
        """Generate batch of videos"""
        try:
            if not prompts.strip():
                return [], "No prompts provided", None, None, None, None
            
            prompt_list = [p.strip() for p in prompts.split('\n') if p.strip()]
            
            if not prompt_list:
                return [], "No valid prompts found", None, None, None, None
            
            # Parse resolution
            width, height = map(int, resolution.split('x'))
            
            results = []
            status = f"Processing {len(prompt_list)} videos..."
            
            # Generate first 4 videos for preview
            preview_videos = []
            
            for i, prompt in enumerate(prompt_list[:4]):
                try:
                    # Generate video
                    video_path = self.video_generator.generate_video(
                        prompt=prompt,
                        duration=duration,
                        style=style,
                        scene_type=scene_type,
                        resolution=(width, height)
                    )
                    
                    if video_path and os.path.exists(video_path):
                        # Process the video
                        processed_path = self.video_processor.process_video(video_path, processing_style)
                        
                        # Save to results
                        results.append(processed_path if processed_path else video_path)
                        preview_videos.append(processed_path if processed_path else video_path)
                    else:
                        preview_videos.append(None)
                    
                    status = f"Processed {i+1}/{len(prompt_list)} videos..."
                    
                except Exception as e:
                    print(f"Error processing video {i+1}: {e}")
                    preview_videos.append(None)
                    continue
            
            # Pad preview videos to 4
            while len(preview_videos) < 4:
                preview_videos.append(None)
            
            status = f"Completed! Generated {len(results)} videos successfully."
            
            return results, status, preview_videos[0], preview_videos[1], preview_videos[2], preview_videos[3]
            
        except Exception as e:
            return [], f"Batch generation error: {e}", None, None, None, None
    
    def _save_settings(self, device: str, fps: int, max_duration: int, 
                      generation_quality: str, enable_processing: bool, enable_effects: bool) -> dict:
        """Save advanced settings"""
        try:
            settings = {
                "device": device,
                "fps": fps,
                "max_duration": max_duration,
                "generation_quality": generation_quality,
                "enable_processing": enable_processing,
                "enable_effects": enable_effects,
                "available_styles": len(self.style_controller.get_available_styles()),
                "available_scenes": len(self.style_controller.get_available_scenes()),
                "available_effects": len(self.video_processor.get_available_effects())
            }
            
            return settings
            
        except Exception as e:
            return {"error": str(e)}
    
    def _export_settings(self) -> str:
        """Export settings to file"""
        try:
            settings_file = self.output_dir / "exported_video_settings.json"
            
            settings = {
                "styles": self.style_controller.get_available_styles(),
                "scenes": self.style_controller.get_available_scenes(),
                "moods": self.style_controller.get_available_moods(),
                "effects": self.video_processor.get_available_effects(),
                "color_presets": self.video_processor.get_available_color_presets()
            }
            
            with open(settings_file, 'w') as f:
                json.dump(settings, f, indent=2)
            
            return str(settings_file)
            
        except Exception as e:
            return None
    
    def _import_settings(self, settings_file: str) -> dict:
        """Import settings from file"""
        try:
            if not settings_file:
                return {"error": "No file provided"}
            
            with open(settings_file, 'r') as f:
                settings = json.load(f)
            
            return settings
            
        except Exception as e:
            return {"error": str(e)}
    
    def launch(self, share: bool = False, server_name: str = "127.0.0.1", server_port: int = 7862):
        """Launch the video web interface"""
        print("🎬 Launching Hydax AI Video Generator Web Interface...")
        print(f"🎥 Interface will be available at: http://{server_name}:{server_port}")
        
        self.interface.launch(
            share=share,
            server_name=server_name,
            server_port=server_port,
            show_error=True,
            quiet=False
        )


# Main execution
if __name__ == "__main__":
    # Create and launch the video web interface
    video_web_interface = VideoWebInterface()
    video_web_interface.launch(share=False, server_port=7862)
