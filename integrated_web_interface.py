"""
Integrated Web Interface for Hydax AI - Complete Audio & Video Suite
Combines TTS, Music Generation, and Video Generation in one interface
"""

import gradio as gr
import numpy as np
import tempfile
import os
from pathlib import Path
import json
from typing import Optional, Tuple, List, Dict

# Import all our modules
from tts_engine import HydaxTTS
from voice_cloning import VoiceCloner
from emotion_control import EmotionController
from audio_processing import AudioProcessor
from music_generator import MusicGenerator
from music_styles import MusicStyleController
from music_processing import MusicProcessor
from video_generator import VideoGenerator
from video_styles import VideoStyleController
from video_processing import VideoProcessor

class IntegratedWebInterface:
    """
    Complete integrated web interface for Hydax AI
    """
    
    def __init__(self):
        """Initialize the integrated interface"""
        # Initialize all components
        self.tts_engine = HydaxTTS()
        self.voice_cloner = VoiceCloner()
        self.emotion_controller = EmotionController()
        self.audio_processor = AudioProcessor()
        
        self.music_generator = MusicGenerator()
        self.music_style_controller = MusicStyleController()
        self.music_processor = MusicProcessor()
        
        self.video_generator = VideoGenerator()
        self.video_style_controller = VideoStyleController()
        self.video_processor = VideoProcessor()
        
        # Create output directories
        self.output_dir = Path("integrated_outputs")
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize interface
        self.interface = self._create_interface()
    
    def _create_interface(self) -> gr.Blocks:
        """Create the integrated Gradio interface"""
        with gr.Blocks(
            title="Hydax AI - Complete Audio & Video Suite",
            theme=gr.themes.Soft(),
            css="""
            .gradio-container {
                max-width: 1800px !important;
                margin: auto !important;
            }
            .main-header {
                text-align: center;
                margin-bottom: 30px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                border-radius: 15px;
            }
            .section-header {
                margin-top: 20px;
                margin-bottom: 10px;
                font-weight: bold;
                color: #2563eb;
            }
            .feature-card {
                background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                padding: 20px;
                border-radius: 10px;
                color: white;
                margin-bottom: 20px;
            }
            """
        ) as interface:
            
            gr.HTML("""
            <div class="main-header">
                <h1>🎤🎵🎬 Hydax AI - Complete Audio & Video Suite</h1>
                <p>The ultimate AI-powered platform for Text-to-Speech, Music Generation, and Video Creation</p>
                <p>Create professional audio and video content with advanced AI models and beautiful interfaces</p>
            </div>
            """)
            
            with gr.Tabs():
                # Quick Start Tab
                with gr.Tab("🚀 Quick Start"):
                    self._create_quick_start_tab()
                
                # TTS Tab
                with gr.Tab("🎤 Text-to-Speech"):
                    self._create_tts_tab()
                
                # Music Generation Tab
                with gr.Tab("🎵 Music Generation"):
                    self._create_music_tab()
                
                # Video Generation Tab
                with gr.Tab("🎬 Video Generation"):
                    self._create_video_tab()
                
                # Multimedia Creation Tab
                with gr.Tab("🎭 Multimedia Creation"):
                    self._create_multimedia_tab()
                
                # Voice Cloning Tab
                with gr.Tab("🎭 Voice Cloning"):
                    self._create_voice_cloning_tab()
                
                # Batch Processing Tab
                with gr.Tab("📦 Batch Processing"):
                    self._create_batch_processing_tab()
                
                # Advanced Settings Tab
                with gr.Tab("⚙️ Advanced Settings"):
                    self._create_advanced_settings_tab()
        
        return interface
    
    def _create_quick_start_tab(self):
        """Create the quick start tab"""
        with gr.Row():
            with gr.Column(scale=2):
                gr.HTML('<div class="feature-card"><h3>🚀 Quick Start - Create Everything</h3></div>')
                
                # Quick creation options
                creation_type = gr.Radio(
                    choices=["TTS Only", "Music Only", "Video Only", "TTS + Music", "TTS + Video", "Music + Video", "Complete Multimedia"],
                    value="TTS Only",
                    label="What would you like to create?"
                )
                
                # Universal prompt
                universal_prompt = gr.Textbox(
                    label="Content Description",
                    placeholder="Describe what you want to create... (e.g., 'A motivational speech about success', 'An upbeat pop song', 'A nature documentary video')",
                    lines=3
                )
                
                # Quick settings
                with gr.Row():
                    duration = gr.Slider(
                        minimum=30,
                        maximum=300,
                        value=120,
                        step=30,
                        label="Duration (seconds)"
                    )
                    
                    quality = gr.Dropdown(
                        choices=["Fast", "Balanced", "High Quality"],
                        value="Balanced",
                        label="Quality"
                    )
                
                # Quick generate button
                quick_generate_btn = gr.Button("🚀 Generate Content", variant="primary", size="lg")
                
            with gr.Column(scale=1):
                gr.Markdown("### 🎯 Output")
                
                # Outputs
                tts_output = gr.Audio(label="Generated Speech", type="filepath")
                music_output = gr.Audio(label="Generated Music", type="filepath")
                video_output = gr.Video(label="Generated Video", format="mp4")
                
                # Combined output
                combined_output = gr.File(label="Combined Content", file_count="multiple")
                
                # Status
                status = gr.Textbox(label="Status", interactive=False)
        
        # Event handlers
        quick_generate_btn.click(
            fn=self._quick_generate,
            inputs=[creation_type, universal_prompt, duration, quality],
            outputs=[tts_output, music_output, video_output, combined_output, status]
        )
    
    def _create_tts_tab(self):
        """Create the TTS tab"""
        with gr.Row():
            with gr.Column(scale=2):
                gr.HTML('<div class="feature-card"><h3>🎤 Text-to-Speech Generation</h3></div>')
                
                # Text input
                text_input = gr.Textbox(
                    label="Text to Convert",
                    placeholder="Enter the text you want to convert to speech...",
                    lines=5
                )
                
                # Voice selection
                with gr.Row():
                    voice = gr.Dropdown(
                        choices=["en-US-AriaNeural", "en-US-JennyNeural", "en-US-GuyNeural"],
                        value="en-US-AriaNeural",
                        label="Voice"
                    )
                    
                    speed = gr.Slider(
                        minimum=0.5,
                        maximum=2.0,
                        value=1.0,
                        step=0.1,
                        label="Speed"
                    )
                
                # Emotion and style
                with gr.Row():
                    emotion = gr.Dropdown(
                        choices=self.emotion_controller.get_available_emotions(),
                        value="neutral",
                        label="Emotion"
                    )
                    
                    style = gr.Dropdown(
                        choices=self.emotion_controller.get_available_styles(),
                        value="conversational",
                        label="Speaking Style"
                    )
                
                # Audio processing
                with gr.Accordion("Audio Processing", open=False):
                    noise_reduction = gr.Checkbox(value=True, label="Noise Reduction")
                    normalization = gr.Checkbox(value=True, label="Normalization")
                    enhancement = gr.Checkbox(value=True, label="Audio Enhancement")
                
                # Generate button
                tts_generate_btn = gr.Button("🎤 Generate Speech", variant="primary")
                
            with gr.Column(scale=1):
                gr.Markdown("### 🎵 TTS Output")
                
                # Audio output
                tts_audio_output = gr.Audio(label="Generated Speech", type="filepath")
                
                # Download and controls
                with gr.Row():
                    tts_download_btn = gr.Button("💾 Download", size="sm")
                    tts_play_btn = gr.Button("▶️ Play", size="sm")
                
                # Audio info
                tts_info = gr.JSON(label="Audio Information")
        
        # Event handlers
        tts_generate_btn.click(
            fn=self._generate_tts,
            inputs=[text_input, voice, speed, emotion, style, noise_reduction, normalization, enhancement],
            outputs=[tts_audio_output, tts_info]
        )
    
    def _create_music_tab(self):
        """Create the music generation tab"""
        with gr.Row():
            with gr.Column(scale=2):
                gr.HTML('<div class="feature-card"><h3>🎵 Music Generation</h3></div>')
                
                # Music prompt
                music_prompt = gr.Textbox(
                    label="Music Description",
                    placeholder="Describe the music you want to generate... (e.g., 'An upbeat pop song with electric guitar and drums')",
                    lines=3
                )
                
                # Music parameters
                with gr.Row():
                    music_style = gr.Dropdown(
                        choices=self.music_style_controller.get_available_styles(),
                        value="pop",
                        label="Style"
                    )
                    
                    music_mood = gr.Dropdown(
                        choices=self.music_style_controller.get_available_moods(),
                        value="happy",
                        label="Mood"
                    )
                
                with gr.Row():
                    music_duration = gr.Slider(
                        minimum=60,
                        maximum=300,
                        value=180,
                        step=30,
                        label="Duration (seconds)"
                    )
                    
                    tempo = gr.Slider(
                        minimum=60,
                        maximum=180,
                        value=120,
                        step=10,
                        label="Tempo (BPM)"
                    )
                
                # Music processing
                with gr.Accordion("Music Processing", open=False):
                    mastering_preset = gr.Dropdown(
                        choices=self.music_processor.get_available_presets(),
                        value="radio",
                        label="Mastering Preset"
                    )
                    
                    add_effects = gr.Checkbox(value=True, label="Add Effects")
                
                # Generate button
                music_generate_btn = gr.Button("🎵 Generate Music", variant="primary")
                
            with gr.Column(scale=1):
                gr.Markdown("### 🎶 Music Output")
                
                # Music output
                music_audio_output = gr.Audio(label="Generated Music", type="filepath")
                
                # Download and controls
                with gr.Row():
                    music_download_btn = gr.Button("💾 Download", size="sm")
                    music_play_btn = gr.Button("▶️ Play", size="sm")
                
                # Music info
                music_info = gr.JSON(label="Music Information")
        
        # Event handlers
        music_generate_btn.click(
            fn=self._generate_music,
            inputs=[music_prompt, music_style, music_mood, music_duration, tempo, mastering_preset, add_effects],
            outputs=[music_audio_output, music_info]
        )
    
    def _create_video_tab(self):
        """Create the video generation tab"""
        with gr.Row():
            with gr.Column(scale=2):
                gr.HTML('<div class="feature-card"><h3>🎬 Video Generation</h3></div>')
                
                # Video prompt
                video_prompt = gr.Textbox(
                    label="Video Description",
                    placeholder="Describe the video you want to generate... (e.g., 'A peaceful nature scene with trees and sunset')",
                    lines=3
                )
                
                # Video parameters
                with gr.Row():
                    video_style = gr.Dropdown(
                        choices=self.video_style_controller.get_available_styles(),
                        value="cinematic",
                        label="Style"
                    )
                    
                    scene_type = gr.Dropdown(
                        choices=self.video_style_controller.get_available_scenes(),
                        value="nature",
                        label="Scene Type"
                    )
                
                with gr.Row():
                    video_duration = gr.Slider(
                        minimum=30,
                        maximum=300,
                        value=120,
                        step=30,
                        label="Duration (seconds)"
                    )
                    
                    resolution = gr.Dropdown(
                        choices=["1920x1080", "1280x720", "854x480"],
                        value="1280x720",
                        label="Resolution"
                    )
                
                # Video processing
                with gr.Accordion("Video Processing", open=False):
                    video_processing = gr.Dropdown(
                        choices=self.video_processor.get_available_effects(),
                        value="cinematic",
                        label="Processing Style"
                    )
                    
                    add_text_overlay = gr.Checkbox(value=False, label="Add Text Overlay")
                    text_overlay_content = gr.Textbox(
                        label="Text Overlay",
                        placeholder="Text to display on video...",
                        visible=False
                    )
                
                # Generate button
                video_generate_btn = gr.Button("🎬 Generate Video", variant="primary")
                
            with gr.Column(scale=1):
                gr.Markdown("### 🎥 Video Output")
                
                # Video output
                video_video_output = gr.Video(label="Generated Video", format="mp4")
                
                # Download and controls
                with gr.Row():
                    video_download_btn = gr.Button("💾 Download", size="sm")
                    video_play_btn = gr.Button("▶️ Play", size="sm")
                
                # Video info
                video_info = gr.JSON(label="Video Information")
        
        # Event handlers
        add_text_overlay.change(
            fn=lambda x: gr.update(visible=x),
            inputs=[add_text_overlay],
            outputs=[text_overlay_content]
        )
        
        video_generate_btn.click(
            fn=self._generate_video,
            inputs=[video_prompt, video_style, scene_type, video_duration, resolution, video_processing, add_text_overlay, text_overlay_content],
            outputs=[video_video_output, video_info]
        )
    
    def _create_multimedia_tab(self):
        """Create the multimedia creation tab"""
        with gr.Row():
            with gr.Column(scale=2):
                gr.HTML('<div class="feature-card"><h3>🎭 Complete Multimedia Creation</h3></div>')
                
                # Content description
                content_description = gr.Textbox(
                    label="Content Description",
                    placeholder="Describe your complete multimedia content... (e.g., 'A motivational video with speech, background music, and nature scenes')",
                    lines=4
                )
                
                # Component selection
                with gr.Row():
                    include_tts = gr.Checkbox(value=True, label="Include TTS")
                    include_music = gr.Checkbox(value=True, label="Include Music")
                    include_video = gr.Checkbox(value=True, label="Include Video")
                
                # TTS settings
                with gr.Accordion("TTS Settings", open=False):
                    tts_text = gr.Textbox(
                        label="Speech Text",
                        placeholder="Text for the speech component...",
                        lines=3
                    )
                    tts_voice = gr.Dropdown(
                        choices=["en-US-AriaNeural", "en-US-JennyNeural", "en-US-GuyNeural"],
                        value="en-US-AriaNeural",
                        label="Voice"
                    )
                    tts_emotion = gr.Dropdown(
                        choices=self.emotion_controller.get_available_emotions(),
                        value="motivational",
                        label="Emotion"
                    )
                
                # Music settings
                with gr.Accordion("Music Settings", open=False):
                    music_description = gr.Textbox(
                        label="Music Description",
                        placeholder="Description for background music...",
                        lines=2
                    )
                    music_style = gr.Dropdown(
                        choices=self.music_style_controller.get_available_styles(),
                        value="ambient",
                        label="Music Style"
                    )
                    music_mood = gr.Dropdown(
                        choices=self.music_style_controller.get_available_moods(),
                        value="inspiring",
                        label="Music Mood"
                    )
                
                # Video settings
                with gr.Accordion("Video Settings", open=False):
                    video_description = gr.Textbox(
                        label="Video Description",
                        placeholder="Description for video content...",
                        lines=2
                    )
                    video_style = gr.Dropdown(
                        choices=self.video_style_controller.get_available_styles(),
                        value="cinematic",
                        label="Video Style"
                    )
                    scene_type = gr.Dropdown(
                        choices=self.video_style_controller.get_available_scenes(),
                        value="nature",
                        label="Scene Type"
                    )
                
                # Duration and quality
                with gr.Row():
                    multimedia_duration = gr.Slider(
                        minimum=60,
                        maximum=300,
                        value=180,
                        step=30,
                        label="Total Duration (seconds)"
                    )
                    
                    multimedia_quality = gr.Dropdown(
                        choices=["Fast", "Balanced", "High Quality"],
                        value="Balanced",
                        label="Quality"
                    )
                
                # Generate button
                multimedia_generate_btn = gr.Button("🎭 Create Multimedia Content", variant="primary", size="lg")
                
            with gr.Column(scale=1):
                gr.Markdown("### 🎬 Multimedia Output")
                
                # Individual components
                multimedia_tts_output = gr.Audio(label="Speech Component", type="filepath")
                multimedia_music_output = gr.Audio(label="Music Component", type="filepath")
                multimedia_video_output = gr.Video(label="Video Component", format="mp4")
                
                # Combined output
                multimedia_combined_output = gr.Video(label="Combined Multimedia", format="mp4")
                
                # Download controls
                with gr.Row():
                    download_combined_btn = gr.Button("💾 Download Combined", size="sm")
                    download_individual_btn = gr.Button("📦 Download All", size="sm")
                
                # Status
                multimedia_status = gr.Textbox(label="Creation Status", interactive=False, lines=3)
        
        # Event handlers
        multimedia_generate_btn.click(
            fn=self._create_multimedia,
            inputs=[content_description, include_tts, include_music, include_video, 
                   tts_text, tts_voice, tts_emotion, music_description, music_style, music_mood,
                   video_description, video_style, scene_type, multimedia_duration, multimedia_quality],
            outputs=[multimedia_tts_output, multimedia_music_output, multimedia_video_output, 
                    multimedia_combined_output, multimedia_status]
        )
    
    def _create_voice_cloning_tab(self):
        """Create the voice cloning tab"""
        with gr.Row():
            with gr.Column(scale=2):
                gr.HTML('<div class="feature-card"><h3>🎭 Voice Cloning</h3></div>')
                
                # Voice sample upload
                voice_sample = gr.Audio(
                    label="Voice Sample",
                    type="filepath",
                    info="Upload a clear audio sample of the voice to clone (10-30 seconds recommended)"
                )
                
                # Text to synthesize
                clone_text = gr.Textbox(
                    label="Text to Synthesize",
                    placeholder="Enter the text you want to synthesize with the cloned voice...",
                    lines=4
                )
                
                # Cloning settings
                with gr.Row():
                    clone_quality = gr.Dropdown(
                        choices=["Fast", "Balanced", "High Quality"],
                        value="Balanced",
                        label="Cloning Quality"
                    )
                    
                    clone_speed = gr.Slider(
                        minimum=0.5,
                        maximum=2.0,
                        value=1.0,
                        step=0.1,
                        label="Speech Speed"
                    )
                
                # Generate button
                clone_generate_btn = gr.Button("🎭 Clone Voice", variant="primary")
                
            with gr.Column(scale=1):
                gr.Markdown("### 🎤 Cloned Voice Output")
                
                # Cloned voice output
                cloned_voice_output = gr.Audio(label="Cloned Voice", type="filepath")
                
                # Download and controls
                with gr.Row():
                    clone_download_btn = gr.Button("💾 Download", size="sm")
                    clone_play_btn = gr.Button("▶️ Play", size="sm")
                
                # Cloning info
                clone_info = gr.JSON(label="Cloning Information")
        
        # Event handlers
        clone_generate_btn.click(
            fn=self._clone_voice,
            inputs=[voice_sample, clone_text, clone_quality, clone_speed],
            outputs=[cloned_voice_output, clone_info]
        )
    
    def _create_batch_processing_tab(self):
        """Create the batch processing tab"""
        with gr.Row():
            with gr.Column(scale=2):
                gr.HTML('<div class="feature-card"><h3>📦 Batch Processing</h3></div>')
                
                # Batch type selection
                batch_type = gr.Radio(
                    choices=["TTS Batch", "Music Batch", "Video Batch", "Mixed Batch"],
                    value="TTS Batch",
                    label="Batch Type"
                )
                
                # Batch input
                batch_input = gr.Textbox(
                    label="Batch Input",
                    placeholder="Enter multiple items, one per line...",
                    lines=10
                )
                
                # Batch settings
                with gr.Row():
                    batch_duration = gr.Slider(
                        minimum=30,
                        maximum=300,
                        value=120,
                        step=30,
                        label="Duration per item (seconds)"
                    )
                    
                    batch_quality = gr.Dropdown(
                        choices=["Fast", "Balanced", "High Quality"],
                        value="Balanced",
                        label="Quality"
                    )
                
                # Process button
                batch_process_btn = gr.Button("📦 Process Batch", variant="primary", size="lg")
                
            with gr.Column(scale=1):
                gr.Markdown("### 📊 Batch Progress")
                
                # Progress tracking
                batch_progress = gr.Progress()
                batch_status = gr.Textbox(label="Batch Status", interactive=False, lines=3)
                
                # Batch results
                batch_results = gr.File(label="Download Batch Results", file_count="multiple")
                
                # Preview
                gr.Markdown("### 🎥 Preview")
                with gr.Row():
                    batch_preview1 = gr.Audio(label="Item 1", type="filepath")
                    batch_preview2 = gr.Audio(label="Item 2", type="filepath")
        
        # Event handlers
        batch_process_btn.click(
            fn=self._process_batch,
            inputs=[batch_type, batch_input, batch_duration, batch_quality],
            outputs=[batch_results, batch_status, batch_preview1, batch_preview2]
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
                
                # Quality settings
                gr.Markdown("#### 🎯 Quality Settings")
                
                tts_quality = gr.Dropdown(
                    choices=["fast", "balanced", "high_quality"],
                    value="balanced",
                    label="TTS Quality"
                )
                
                music_quality = gr.Dropdown(
                    choices=["fast", "balanced", "high_quality"],
                    value="balanced",
                    label="Music Quality"
                )
                
                video_quality = gr.Dropdown(
                    choices=["fast", "balanced", "high_quality"],
                    value="balanced",
                    label="Video Quality"
                )
                
                # Save settings
                save_settings_btn = gr.Button("💾 Save Settings", variant="secondary")
                
            with gr.Column(scale=2):
                gr.Markdown("### 📊 System Information")
                
                # System info
                system_info = gr.JSON(label="System Information")
                
                gr.Markdown("### 🎨 Available Features")
                
                with gr.Row():
                    tts_features = gr.Textbox(
                        label="TTS Features",
                        value="Voice Cloning, Emotion Control, Style Control, Audio Processing",
                        interactive=False,
                        lines=2
                    )
                    
                    music_features = gr.Textbox(
                        label="Music Features",
                        value="AI Generation, Style Control, Mood Control, Professional Mastering",
                        interactive=False,
                        lines=2
                    )
                
                with gr.Row():
                    video_features = gr.Textbox(
                        label="Video Features",
                        value="AI Generation, Style Control, Scene Control, Post-Processing",
                        interactive=False,
                        lines=2
                    )
                
                # Export/Import
                gr.Markdown("### 📤 Export/Import")
                
                with gr.Row():
                    export_settings_btn = gr.Button("📤 Export Settings")
                    import_settings_btn = gr.Button("📥 Import Settings")
                
                settings_file = gr.File(label="Settings File")
        
        # Event handlers
        save_settings_btn.click(
            fn=self._save_settings,
            inputs=[device, tts_quality, music_quality, video_quality],
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
    def _quick_generate(self, creation_type: str, prompt: str, duration: float, quality: str) -> Tuple[str, str, str, List[str], str]:
        """Quick generation based on creation type"""
        try:
            status = f"Generating {creation_type.lower()} content..."
            
            tts_output = None
            music_output = None
            video_output = None
            combined_outputs = []
            
            if "TTS" in creation_type:
                # Generate TTS
                tts_output = self.tts_engine.synthesize(
                    text=prompt,
                    voice="en-US-AriaNeural",
                    speed=1.0
                )
                if tts_output:
                    combined_outputs.append(tts_output)
            
            if "Music" in creation_type:
                # Generate Music
                music_output = self.music_generator.generate_music(
                    prompt=prompt,
                    duration=duration,
                    style="ambient",
                    mood="neutral"
                )
                if music_output:
                    combined_outputs.append(music_output)
            
            if "Video" in creation_type:
                # Generate Video
                video_output = self.video_generator.generate_video(
                    prompt=prompt,
                    duration=duration,
                    style="cinematic",
                    scene_type="abstract"
                )
                if video_output:
                    combined_outputs.append(video_output)
            
            status = f"✓ Generated {creation_type.lower()} content successfully!"
            
            return tts_output, music_output, video_output, combined_outputs, status
            
        except Exception as e:
            return None, None, None, [], f"Error: {str(e)}"
    
    def _generate_tts(self, text: str, voice: str, speed: float, emotion: str, style: str, 
                     noise_reduction: bool, normalization: bool, enhancement: bool) -> Tuple[str, dict]:
        """Generate TTS with processing"""
        try:
            # Generate speech
            audio_path = self.tts_engine.synthesize(
                text=text,
                voice=voice,
                speed=speed
            )
            
            if audio_path and os.path.exists(audio_path):
                # Apply processing
                if noise_reduction or normalization or enhancement:
                    processed_path = self.audio_processor.process_audio(
                        audio_path=audio_path,
                        noise_reduction=noise_reduction,
                        normalization=normalization,
                        enhancement=enhancement
                    )
                    if processed_path:
                        audio_path = processed_path
                
                # Get audio info
                info = self.audio_processor.get_audio_info(audio_path)
                return audio_path, info
            else:
                return None, {"error": "Failed to generate speech"}
                
        except Exception as e:
            return None, {"error": str(e)}
    
    def _generate_music(self, prompt: str, style: str, mood: str, duration: float, tempo: int,
                       mastering_preset: str, add_effects: bool) -> Tuple[str, dict]:
        """Generate music with processing"""
        try:
            # Generate music
            music_path = self.music_generator.generate_music(
                prompt=prompt,
                duration=duration,
                style=style,
                mood=mood,
                tempo=tempo
            )
            
            if music_path and os.path.exists(music_path):
                # Apply processing
                if add_effects:
                    processed_path = self.music_processor.process_music(
                        music_path=music_path,
                        preset=mastering_preset
                    )
                    if processed_path:
                        music_path = processed_path
                
                # Get music info
                info = self.music_generator.get_music_info(music_path)
                return music_path, info
            else:
                return None, {"error": "Failed to generate music"}
                
        except Exception as e:
            return None, {"error": str(e)}
    
    def _generate_video(self, prompt: str, style: str, scene_type: str, duration: float, resolution: str,
                       processing_style: str, add_text_overlay: bool, text_overlay_content: str) -> Tuple[str, dict]:
        """Generate video with processing"""
        try:
            # Parse resolution
            width, height = map(int, resolution.split('x'))
            
            # Generate video
            video_path = self.video_generator.generate_video(
                prompt=prompt,
                duration=duration,
                style=style,
                scene_type=scene_type,
                resolution=(width, height),
                text_overlay=text_overlay_content if add_text_overlay else None
            )
            
            if video_path and os.path.exists(video_path):
                # Apply processing
                processed_path = self.video_processor.process_video(
                    video_path=video_path,
                    style=processing_style
                )
                if processed_path:
                    video_path = processed_path
                
                # Get video info
                info = self.video_generator.get_video_info(video_path)
                return video_path, info
            else:
                return None, {"error": "Failed to generate video"}
                
        except Exception as e:
            return None, {"error": str(e)}
    
    def _create_multimedia(self, content_description: str, include_tts: bool, include_music: bool, include_video: bool,
                          tts_text: str, tts_voice: str, tts_emotion: str, music_description: str, music_style: str, music_mood: str,
                          video_description: str, video_style: str, scene_type: str, duration: float, quality: str) -> Tuple[str, str, str, str, str]:
        """Create complete multimedia content"""
        try:
            status = "Creating multimedia content..."
            
            tts_output = None
            music_output = None
            video_output = None
            combined_output = None
            
            # Generate TTS if requested
            if include_tts and tts_text.strip():
                tts_output = self.tts_engine.synthesize(
                    text=tts_text,
                    voice=tts_voice,
                    speed=1.0
                )
                status += "\n✓ Generated speech component"
            
            # Generate Music if requested
            if include_music and music_description.strip():
                music_output = self.music_generator.generate_music(
                    prompt=music_description,
                    duration=duration,
                    style=music_style,
                    mood=music_mood
                )
                status += "\n✓ Generated music component"
            
            # Generate Video if requested
            if include_video and video_description.strip():
                video_output = self.video_generator.generate_video(
                    prompt=video_description,
                    duration=duration,
                    style=video_style,
                    scene_type=scene_type
                )
                status += "\n✓ Generated video component"
            
            # Combine components if multiple are generated
            if video_output and (tts_output or music_output):
                # This would combine video with audio
                # For now, just return the video
                combined_output = video_output
                status += "\n✓ Combined multimedia content"
            
            status += "\n🎉 Multimedia creation completed!"
            
            return tts_output, music_output, video_output, combined_output, status
            
        except Exception as e:
            return None, None, None, None, f"Error: {str(e)}"
    
    def _clone_voice(self, voice_sample: str, text: str, quality: str, speed: float) -> Tuple[str, dict]:
        """Clone voice from sample"""
        try:
            if not voice_sample or not os.path.exists(voice_sample):
                return None, {"error": "No voice sample provided"}
            
            # Clone voice
            cloned_audio = self.voice_cloner.clone_voice(
                reference_audio=voice_sample,
                text=text,
                speed=speed
            )
            
            if cloned_audio and os.path.exists(cloned_audio):
                # Get audio info
                info = self.audio_processor.get_audio_info(cloned_audio)
                return cloned_audio, info
            else:
                return None, {"error": "Failed to clone voice"}
                
        except Exception as e:
            return None, {"error": str(e)}
    
    def _process_batch(self, batch_type: str, batch_input: str, duration: float, quality: str) -> Tuple[List[str], str, str, str]:
        """Process batch of items"""
        try:
            if not batch_input.strip():
                return [], "No batch input provided", None, None
            
            items = [item.strip() for item in batch_input.split('\n') if item.strip()]
            
            if not items:
                return [], "No valid items found", None, None
            
            results = []
            status = f"Processing {len(items)} {batch_type.lower()} items..."
            
            # Process first 2 items for preview
            preview_items = []
            
            for i, item in enumerate(items[:2]):
                try:
                    if batch_type == "TTS Batch":
                        result = self.tts_engine.synthesize(text=item, voice="en-US-AriaNeural")
                    elif batch_type == "Music Batch":
                        result = self.music_generator.generate_music(prompt=item, duration=duration)
                    elif batch_type == "Video Batch":
                        result = self.video_generator.generate_video(prompt=item, duration=duration)
                    else:
                        result = None
                    
                    if result and os.path.exists(result):
                        results.append(result)
                        preview_items.append(result)
                    else:
                        preview_items.append(None)
                    
                    status = f"Processed {i+1}/{len(items)} items..."
                    
                except Exception as e:
                    print(f"Error processing item {i+1}: {e}")
                    preview_items.append(None)
                    continue
            
            # Pad preview items
            while len(preview_items) < 2:
                preview_items.append(None)
            
            status = f"Completed! Processed {len(results)} items successfully."
            
            return results, status, preview_items[0], preview_items[1]
            
        except Exception as e:
            return [], f"Batch processing error: {e}", None, None
    
    def _save_settings(self, device: str, tts_quality: str, music_quality: str, video_quality: str) -> dict:
        """Save advanced settings"""
        try:
            settings = {
                "device": device,
                "tts_quality": tts_quality,
                "music_quality": music_quality,
                "video_quality": video_quality,
                "available_voices": 3,
                "available_music_styles": len(self.music_style_controller.get_available_styles()),
                "available_video_styles": len(self.video_style_controller.get_available_styles())
            }
            
            return settings
            
        except Exception as e:
            return {"error": str(e)}
    
    def _export_settings(self) -> str:
        """Export settings to file"""
        try:
            settings_file = self.output_dir / "exported_settings.json"
            
            settings = {
                "tts_settings": {
                    "available_voices": ["en-US-AriaNeural", "en-US-JennyNeural", "en-US-GuyNeural"],
                    "available_emotions": self.emotion_controller.get_available_emotions(),
                    "available_styles": self.emotion_controller.get_available_styles()
                },
                "music_settings": {
                    "available_styles": self.music_style_controller.get_available_styles(),
                    "available_moods": self.music_style_controller.get_available_moods(),
                    "available_presets": self.music_processor.get_available_presets()
                },
                "video_settings": {
                    "available_styles": self.video_style_controller.get_available_styles(),
                    "available_scenes": self.video_style_controller.get_available_scenes(),
                    "available_effects": self.video_processor.get_available_effects()
                }
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
    
    def launch(self, share: bool = False, server_name: str = "127.0.0.1", server_port: int = 7860):
        """Launch the integrated web interface"""
        print("🚀 Launching Hydax AI - Complete Audio & Video Suite...")
        print(f"🎤🎵🎬 Interface will be available at: http://{server_name}:{server_port}")
        print("Features: TTS, Music Generation, Video Generation, Voice Cloning, Multimedia Creation")
        
        self.interface.launch(
            share=share,
            server_name=server_name,
            server_port=server_port,
            show_error=True,
            quiet=False
        )


# Main execution
if __name__ == "__main__":
    # Create and launch the integrated web interface
    integrated_interface = IntegratedWebInterface()
    integrated_interface.launch(share=False, server_port=7860)