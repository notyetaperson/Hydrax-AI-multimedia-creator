"""
Web Interface for Hydax AI Music Generator
"""

import gradio as gr
import numpy as np
import tempfile
import os
from pathlib import Path
import json
from typing import Optional, Tuple, List

# Import our music generation modules
from music_generator import MusicGenerator
from music_styles import MusicStyleController
from music_processing import MusicProcessor

class MusicWebInterface:
    """
    Web interface for the Hydax AI Music Generator
    """
    
    def __init__(self):
        """Initialize the music web interface"""
        self.music_generator = MusicGenerator()
        self.style_controller = MusicStyleController()
        self.music_processor = MusicProcessor()
        
        # Create output directory
        self.output_dir = Path("music_outputs")
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize interface
        self.interface = self._create_interface()
    
    def _create_interface(self) -> gr.Blocks:
        """Create the Gradio interface"""
        with gr.Blocks(
            title="Hydax AI Music Generator",
            theme=gr.themes.Soft(),
            css="""
            .gradio-container {
                max-width: 1400px !important;
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
            .music-controls {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 20px;
                border-radius: 10px;
                color: white;
            }
            """
        ) as interface:
            
            gr.HTML("""
            <div class="main-header">
                <h1>🎵 Hydax AI Music Generator</h1>
                <p>Create amazing music from text prompts with AI-powered generation, style control, and professional mastering</p>
            </div>
            """)
            
            with gr.Tabs():
                # Music Generation Tab
                with gr.Tab("🎼 Music Generation"):
                    self._create_music_generation_tab()
                
                # Style Control Tab
                with gr.Tab("🎨 Style & Genre Control"):
                    self._create_style_control_tab()
                
                # Music Processing Tab
                with gr.Tab("🔧 Music Processing"):
                    self._create_music_processing_tab()
                
                # Batch Generation Tab
                with gr.Tab("📦 Batch Generation"):
                    self._create_batch_generation_tab()
                
                # Advanced Settings Tab
                with gr.Tab("⚙️ Advanced Settings"):
                    self._create_advanced_settings_tab()
        
        return interface
    
    def _create_music_generation_tab(self):
        """Create the main music generation tab"""
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### 🎵 Music Generation")
                
                # Prompt input
                prompt = gr.Textbox(
                    label="Music Prompt",
                    placeholder="Describe the music you want to generate... (e.g., 'A peaceful ambient track with soft piano and strings', 'An energetic electronic dance track with heavy bass')",
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
                
                # Style and genre selection
                with gr.Row():
                    style = gr.Dropdown(
                        choices=self.style_controller.get_available_styles(),
                        value="ambient",
                        label="Music Style"
                    )
                    
                    genre = gr.Dropdown(
                        choices=self.style_controller.get_available_genres(),
                        value=None,
                        label="Genre (Optional)"
                    )
                
                # Mood and complexity
                with gr.Row():
                    mood = gr.Dropdown(
                        choices=self.style_controller.get_available_moods(),
                        value="neutral",
                        label="Mood"
                    )
                    
                    complexity = gr.Dropdown(
                        choices=["simple", "medium", "complex"],
                        value="medium",
                        label="Complexity"
                    )
                
                # Musical parameters
                with gr.Accordion("🎼 Musical Parameters", open=False):
                    with gr.Row():
                        tempo = gr.Slider(
                            minimum=60,
                            maximum=200,
                            value=120,
                            step=5,
                            label="Tempo (BPM)"
                        )
                        
                        key = gr.Dropdown(
                            choices=['C', 'G', 'D', 'A', 'E', 'B', 'F#', 'C#', 'F', 'Bb', 'Eb', 'Ab'],
                            value='C',
                            label="Key"
                        )
                    
                    time_signature = gr.Dropdown(
                        choices=['4/4', '3/4', '2/4', '6/8', '12/8'],
                        value='4/4',
                        label="Time Signature"
                    )
                    
                    instruments = gr.CheckboxGroup(
                        choices=['piano', 'guitar', 'bass', 'drums', 'strings', 'synth', 'brass', 'woodwinds', 'vocals', 'percussion'],
                        value=['piano', 'strings', 'bass', 'drums'],
                        label="Instruments"
                    )
                
                # Generate button
                generate_btn = gr.Button("🎵 Generate Music", variant="primary", size="lg")
                
            with gr.Column(scale=1):
                gr.Markdown("### 🎧 Output")
                
                # Audio output
                audio_output = gr.Audio(
                    label="Generated Music",
                    type="numpy",
                    format="wav"
                )
                
                # Download and play controls
                with gr.Row():
                    download_btn = gr.Button("💾 Download", size="sm")
                    play_btn = gr.Button("▶️ Play", size="sm")
                    stop_btn = gr.Button("⏹️ Stop", size="sm")
                
                # Music information
                gr.Markdown("### 📊 Music Info")
                music_info = gr.JSON(label="Music Information")
                
                # Generation status
                status = gr.Textbox(
                    label="Generation Status",
                    interactive=False,
                    value="Ready to generate music..."
                )
        
        # Event handlers
        duration_preset.change(
            fn=self._update_duration_from_preset,
            inputs=[duration_preset],
            outputs=[duration]
        )
        
        generate_btn.click(
            fn=self._generate_music,
            inputs=[prompt, duration, style, genre, mood, complexity, tempo, key, time_signature, instruments],
            outputs=[audio_output, music_info, status]
        )
    
    def _create_style_control_tab(self):
        """Create the style control tab"""
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🎨 Style & Genre Control")
                
                # Style selection
                style = gr.Dropdown(
                    choices=self.style_controller.get_available_styles(),
                    value="ambient",
                    label="Music Style"
                )
                
                # Genre selection
                genre = gr.Dropdown(
                    choices=self.style_controller.get_available_genres(),
                    value=None,
                    label="Genre"
                )
                
                # Mood selection
                mood = gr.Dropdown(
                    choices=self.style_controller.get_available_moods(),
                    value="neutral",
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
                    custom_tempo_range = gr.Textbox(
                        label="Tempo Range (min-max)",
                        placeholder="60-120"
                    )
                    
                    custom_instruments = gr.CheckboxGroup(
                        choices=['piano', 'guitar', 'bass', 'drums', 'strings', 'synth', 'brass', 'woodwinds', 'vocals', 'percussion'],
                        label="Instruments"
                    )
                    
                    custom_effects = gr.CheckboxGroup(
                        choices=['reverb', 'delay', 'distortion', 'compression', 'eq', 'chorus', 'filter'],
                        label="Effects"
                    )
                
                create_custom_style_btn = gr.Button("✨ Create Custom Style", variant="secondary")
                
            with gr.Column(scale=2):
                gr.Markdown("### 🎵 Test Style")
                
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
                test_generate_btn = gr.Button("🎵 Test Style", variant="primary")
                
                # Test output
                test_audio_output = gr.Audio(
                    label="Style Test Output",
                    type="numpy"
                )
                
                # Style comparison
                gr.Markdown("### 🔄 Style Comparison")
                
                with gr.Row():
                    compare_style1 = gr.Dropdown(
                        choices=self.style_controller.get_available_styles(),
                        value="ambient",
                        label="Style 1"
                    )
                    
                    compare_style2 = gr.Dropdown(
                        choices=self.style_controller.get_available_styles(),
                        value="electronic",
                        label="Style 2"
                    )
                
                compare_prompt = gr.Textbox(
                    label="Comparison Prompt",
                    placeholder="Same prompt for both styles...",
                    lines=2
                )
                
                compare_btn = gr.Button("🔄 Compare Styles", variant="secondary")
                
                with gr.Row():
                    compare_audio1 = gr.Audio(label="Style 1 Output", type="numpy")
                    compare_audio2 = gr.Audio(label="Style 2 Output", type="numpy")
        
        # Event handlers
        style.change(
            fn=self._update_style_params,
            inputs=[style],
            outputs=[style_params]
        )
        
        create_custom_style_btn.click(
            fn=self._create_custom_style,
            inputs=[custom_style_name, custom_tempo_range, custom_instruments, custom_effects],
            outputs=[style, style_params]
        )
        
        test_generate_btn.click(
            fn=self._test_style,
            inputs=[test_prompt, test_duration, style, mood],
            outputs=[test_audio_output]
        )
        
        compare_btn.click(
            fn=self._compare_styles,
            inputs=[compare_prompt, compare_style1, compare_style2],
            outputs=[compare_audio1, compare_audio2]
        )
    
    def _create_music_processing_tab(self):
        """Create the music processing tab"""
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🔧 Music Processing")
                
                # Input audio
                input_audio = gr.Audio(
                    label="Input Audio",
                    type="numpy",
                    info="Upload audio to process"
                )
                
                # Processing options
                gr.Markdown("### 🎛️ Processing Options")
                
                # Mastering presets
                mastering_preset = gr.Dropdown(
                    choices=self.music_processor.get_available_mastering_presets(),
                    value="streaming",
                    label="Mastering Preset"
                )
                
                # Effect chains
                effect_chain = gr.Dropdown(
                    choices=self.music_processor.get_available_effect_chains(),
                    value="electronic",
                    label="Effect Chain"
                )
                
                # Enhancement options
                with gr.Accordion("Enhancement Options", open=False):
                    enhancement_type = gr.Dropdown(
                        choices=["light", "medium", "full", "custom"],
                        value="medium",
                        label="Enhancement Type"
                    )
                    
                    noise_reduction = gr.Checkbox(value=True, label="Noise Reduction")
                    dynamic_range = gr.Checkbox(value=True, label="Dynamic Range Enhancement")
                    stereo_imaging = gr.Checkbox(value=True, label="Stereo Imaging")
                    harmonic_enhancement = gr.Checkbox(value=True, label="Harmonic Enhancement")
                
                # Process button
                process_btn = gr.Button("🔧 Process Audio", variant="primary")
                
            with gr.Column(scale=2):
                gr.Markdown("### 🎧 Processed Output")
                
                # Processed audio
                processed_audio = gr.Audio(
                    label="Processed Audio",
                    type="numpy"
                )
                
                # Download controls
                with gr.Row():
                    download_processed_btn = gr.Button("💾 Download Processed", size="sm")
                    play_processed_btn = gr.Button("▶️ Play Processed", size="sm")
                
                # Quality analysis
                gr.Markdown("### 📊 Quality Analysis")
                
                with gr.Row():
                    analyze_btn = gr.Button("📊 Analyze Quality", variant="secondary")
                    compare_btn = gr.Button("🔄 Compare Original", variant="secondary")
                
                # Analysis results
                quality_analysis = gr.JSON(label="Quality Metrics")
                
                # A/B comparison
                with gr.Row():
                    original_audio = gr.Audio(label="Original Audio", type="numpy")
                    processed_audio_compare = gr.Audio(label="Processed Audio", type="numpy")
        
        # Event handlers
        process_btn.click(
            fn=self._process_audio,
            inputs=[input_audio, mastering_preset, effect_chain, enhancement_type, noise_reduction, dynamic_range, stereo_imaging, harmonic_enhancement],
            outputs=[processed_audio, quality_analysis]
        )
        
        analyze_btn.click(
            fn=self._analyze_audio_quality,
            inputs=[input_audio],
            outputs=[quality_analysis]
        )
        
        compare_btn.click(
            fn=self._compare_audio,
            inputs=[input_audio, processed_audio],
            outputs=[original_audio, processed_audio_compare]
        )
    
    def _create_batch_generation_tab(self):
        """Create the batch generation tab"""
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📦 Batch Music Generation")
                
                # Batch prompts
                batch_prompts = gr.Textbox(
                    label="Music Prompts (one per line)",
                    placeholder="Enter multiple music prompts, one per line...",
                    lines=10
                )
                
                # Batch settings
                with gr.Row():
                    batch_duration = gr.Slider(
                        minimum=30,
                        maximum=300,
                        value=120,
                        step=30,
                        label="Duration per track (seconds)"
                    )
                    
                    batch_style = gr.Dropdown(
                        choices=self.style_controller.get_available_styles(),
                        value="ambient",
                        label="Style"
                    )
                
                with gr.Row():
                    batch_mood = gr.Dropdown(
                        choices=self.style_controller.get_available_moods(),
                        value="neutral",
                        label="Mood"
                    )
                    
                    batch_complexity = gr.Dropdown(
                        choices=["simple", "medium", "complex"],
                        value="medium",
                        label="Complexity"
                    )
                
                # Batch processing options
                with gr.Accordion("Batch Processing Options", open=False):
                    batch_mastering = gr.Dropdown(
                        choices=self.music_processor.get_available_mastering_presets(),
                        value="streaming",
                        label="Mastering Preset"
                    )
                    
                    batch_enhancement = gr.Dropdown(
                        choices=["light", "medium", "full"],
                        value="medium",
                        label="Enhancement"
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
                gr.Markdown("### 🎵 Batch Results")
                
                batch_results = gr.File(
                    label="Download Batch Results",
                    file_count="multiple"
                )
                
                # Individual track previews
                gr.Markdown("### 🎧 Track Previews")
                
                with gr.Row():
                    track1_audio = gr.Audio(label="Track 1", type="numpy")
                    track2_audio = gr.Audio(label="Track 2", type="numpy")
                
                with gr.Row():
                    track3_audio = gr.Audio(label="Track 3", type="numpy")
                    track4_audio = gr.Audio(label="Track 4", type="numpy")
        
        # Event handlers
        batch_generate_btn.click(
            fn=self._generate_batch,
            inputs=[batch_prompts, batch_duration, batch_style, batch_mood, batch_complexity, batch_mastering, batch_enhancement],
            outputs=[batch_results, batch_status, track1_audio, track2_audio, track3_audio, track4_audio]
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
                
                sample_rate = gr.Slider(
                    minimum=22050,
                    maximum=48000,
                    value=44100,
                    step=1000,
                    label="Sample Rate"
                )
                
                # Generation settings
                gr.Markdown("#### 🎵 Generation Settings")
                
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
                
                enable_mastering = gr.Checkbox(value=True, label="Enable Mastering")
                enable_enhancement = gr.Checkbox(value=True, label="Enable Enhancement")
                
                # Save settings
                save_settings_btn = gr.Button("💾 Save Settings", variant="secondary")
                
            with gr.Column(scale=2):
                gr.Markdown("### 📊 System Information")
                
                # System info
                system_info = gr.JSON(label="System Information")
                
                gr.Markdown("### 🎨 Available Styles & Genres")
                
                with gr.Row():
                    available_styles = gr.Textbox(
                        label="Available Styles",
                        value=", ".join(self.style_controller.get_available_styles()),
                        interactive=False,
                        lines=3
                    )
                    
                    available_genres = gr.Textbox(
                        label="Available Genres",
                        value=", ".join(self.style_controller.get_available_genres()),
                        interactive=False,
                        lines=3
                    )
                
                gr.Markdown("### 🔧 Available Processing Options")
                
                with gr.Row():
                    mastering_presets = gr.Textbox(
                        label="Mastering Presets",
                        value=", ".join(self.music_processor.get_available_mastering_presets()),
                        interactive=False,
                        lines=2
                    )
                    
                    effect_chains = gr.Textbox(
                        label="Effect Chains",
                        value=", ".join(self.music_processor.get_available_effect_chains()),
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
            inputs=[device, sample_rate, max_duration, generation_quality, enable_mastering, enable_enhancement],
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
    
    def _generate_music(self, prompt: str, duration: float, style: str, genre: str, 
                       mood: str, complexity: str, tempo: int, key: str, 
                       time_signature: str, instruments: List[str]) -> Tuple[np.ndarray, dict, str]:
        """Generate music from parameters"""
        try:
            if not prompt.strip():
                return None, {"error": "No prompt provided"}, "Error: No prompt provided"
            
            status = f"Generating {duration:.0f}s of {style} music..."
            
            # Generate music
            music = self.music_generator.generate_music(
                prompt=prompt,
                duration=duration,
                style=style,
                genre=genre,
                tempo=tempo,
                key=key,
                time_signature=time_signature,
                instruments=instruments,
                mood=mood,
                complexity=complexity
            )
            
            if len(music) > 0:
                # Get music info
                info = self.music_generator.get_music_info(music)
                status = f"✓ Generated {info['duration']:.1f}s of music successfully!"
                
                return (self.music_generator.sample_rate, music), info, status
            else:
                return None, {"error": "Failed to generate music"}, "Error: Failed to generate music"
                
        except Exception as e:
            return None, {"error": str(e)}, f"Error: {str(e)}"
    
    def _update_style_params(self, style: str) -> dict:
        """Update style parameters display"""
        try:
            params = self.style_controller.get_style_parameters(style)
            return params
        except Exception as e:
            return {"error": str(e)}
    
    def _create_custom_style(self, name: str, tempo_range: str, instruments: List[str], effects: List[str]) -> Tuple[List[str], dict]:
        """Create a custom style"""
        try:
            if not name.strip():
                return gr.update(), {}
            
            # Parse tempo range
            if '-' in tempo_range:
                min_tempo, max_tempo = map(int, tempo_range.split('-'))
            else:
                min_tempo, max_tempo = 60, 120
            
            # Create custom style parameters
            custom_params = {
                'tempo_range': (min_tempo, max_tempo),
                'instruments': instruments,
                'effects': effects,
                'dynamics': 'custom',
                'harmony': 'custom',
                'rhythm': 'custom'
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
    
    def _test_style(self, prompt: str, duration: float, style: str, mood: str) -> Tuple[int, np.ndarray]:
        """Test a style with a prompt"""
        try:
            if not prompt.strip():
                return None
            
            music = self.music_generator.generate_music(
                prompt=prompt,
                duration=duration,
                style=style,
                mood=mood
            )
            
            if len(music) > 0:
                return (self.music_generator.sample_rate, music)
            else:
                return None
                
        except Exception as e:
            print(f"Style test failed: {e}")
            return None
    
    def _compare_styles(self, prompt: str, style1: str, style2: str) -> Tuple[Tuple[int, np.ndarray], Tuple[int, np.ndarray]]:
        """Compare two styles with the same prompt"""
        try:
            if not prompt.strip():
                return None, None
            
            # Generate with style 1
            music1 = self.music_generator.generate_music(
                prompt=prompt,
                duration=60,
                style=style1
            )
            
            # Generate with style 2
            music2 = self.music_generator.generate_music(
                prompt=prompt,
                duration=60,
                style=style2
            )
            
            result1 = (self.music_generator.sample_rate, music1) if len(music1) > 0 else None
            result2 = (self.music_generator.sample_rate, music2) if len(music2) > 0 else None
            
            return result1, result2
            
        except Exception as e:
            print(f"Style comparison failed: {e}")
            return None, None
    
    def _process_audio(self, audio: Tuple[int, np.ndarray], mastering_preset: str, 
                      effect_chain: str, enhancement_type: str, noise_reduction: bool,
                      dynamic_range: bool, stereo_imaging: bool, harmonic_enhancement: bool) -> Tuple[Tuple[int, np.ndarray], dict]:
        """Process audio with selected options"""
        try:
            if audio is None:
                return None, {"error": "No audio provided"}
            
            sample_rate, audio_data = audio
            
            # Apply effect chain
            processed_audio = self.music_processor.apply_effect_chain(audio_data, effect_chain)
            
            # Apply enhancement
            enhanced_audio = self.music_processor.enhance_music(
                processed_audio,
                enhancement_type=enhancement_type,
                noise_reduction=noise_reduction,
                dynamic_range=dynamic_range,
                stereo_imaging=stereo_imaging,
                harmonic_enhancement=harmonic_enhancement
            )
            
            # Apply mastering
            mastered_audio = self.music_processor.master_music(enhanced_audio, mastering_preset)
            
            # Analyze quality
            quality_metrics = self.music_processor.analyze_music_quality(mastered_audio)
            
            return (sample_rate, mastered_audio), quality_metrics
            
        except Exception as e:
            return None, {"error": str(e)}
    
    def _analyze_audio_quality(self, audio: Tuple[int, np.ndarray]) -> dict:
        """Analyze audio quality"""
        try:
            if audio is None:
                return {"error": "No audio provided"}
            
            sample_rate, audio_data = audio
            return self.music_processor.analyze_music_quality(audio_data)
            
        except Exception as e:
            return {"error": str(e)}
    
    def _compare_audio(self, original: Tuple[int, np.ndarray], processed: Tuple[int, np.ndarray]) -> Tuple[Tuple[int, np.ndarray], Tuple[int, np.ndarray]]:
        """Compare original and processed audio"""
        return original, processed
    
    def _generate_batch(self, prompts: str, duration: float, style: str, mood: str, 
                       complexity: str, mastering_preset: str, enhancement: str) -> Tuple[List[str], str, Tuple[int, np.ndarray], Tuple[int, np.ndarray], Tuple[int, np.ndarray], Tuple[int, np.ndarray]]:
        """Generate batch of music tracks"""
        try:
            if not prompts.strip():
                return [], "No prompts provided", None, None, None, None
            
            prompt_list = [p.strip() for p in prompts.split('\n') if p.strip()]
            
            if not prompt_list:
                return [], "No valid prompts found", None, None, None, None
            
            results = []
            status = f"Processing {len(prompt_list)} tracks..."
            
            # Generate first 4 tracks for preview
            preview_tracks = []
            
            for i, prompt in enumerate(prompt_list[:4]):
                try:
                    # Generate music
                    music = self.music_generator.generate_music(
                        prompt=prompt,
                        duration=duration,
                        style=style,
                        mood=mood,
                        complexity=complexity
                    )
                    
                    if len(music) > 0:
                        # Process the music
                        processed_music = self.music_processor.master_music(music, mastering_preset)
                        
                        # Save to file
                        filename = f"batch_track_{i+1:03d}.wav"
                        filepath = self.output_dir / filename
                        
                        if self.music_generator.save_music(processed_music, str(filepath)):
                            results.append(str(filepath))
                            preview_tracks.append((self.music_generator.sample_rate, processed_music))
                        else:
                            preview_tracks.append(None)
                    else:
                        preview_tracks.append(None)
                    
                    status = f"Processed {i+1}/{len(prompt_list)} tracks..."
                    
                except Exception as e:
                    print(f"Error processing track {i+1}: {e}")
                    preview_tracks.append(None)
                    continue
            
            # Pad preview tracks to 4
            while len(preview_tracks) < 4:
                preview_tracks.append(None)
            
            status = f"Completed! Generated {len(results)} tracks successfully."
            
            return results, status, preview_tracks[0], preview_tracks[1], preview_tracks[2], preview_tracks[3]
            
        except Exception as e:
            return [], f"Batch generation error: {e}", None, None, None, None
    
    def _save_settings(self, device: str, sample_rate: int, max_duration: int, 
                      generation_quality: str, enable_mastering: bool, enable_enhancement: bool) -> dict:
        """Save advanced settings"""
        try:
            settings = {
                "device": device,
                "sample_rate": sample_rate,
                "max_duration": max_duration,
                "generation_quality": generation_quality,
                "enable_mastering": enable_mastering,
                "enable_enhancement": enable_enhancement,
                "available_styles": len(self.style_controller.get_available_styles()),
                "available_genres": len(self.style_controller.get_available_genres()),
                "available_mastering_presets": len(self.music_processor.get_available_mastering_presets()),
                "available_effect_chains": len(self.music_processor.get_available_effect_chains())
            }
            
            # Save to file
            settings_file = self.output_dir / "settings.json"
            with open(settings_file, 'w') as f:
                json.dump(settings, f, indent=2)
            
            return settings
            
        except Exception as e:
            return {"error": str(e)}
    
    def _export_settings(self) -> str:
        """Export settings to file"""
        try:
            settings_file = self.output_dir / "exported_settings.json"
            
            settings = {
                "styles": self.style_controller.get_available_styles(),
                "genres": self.style_controller.get_available_genres(),
                "moods": self.style_controller.get_available_moods(),
                "mastering_presets": self.music_processor.get_available_mastering_presets(),
                "effect_chains": self.music_processor.get_available_effect_chains()
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
    
    def launch(self, share: bool = False, server_name: str = "127.0.0.1", server_port: int = 7861):
        """Launch the music web interface"""
        print("🎵 Launching Hydax AI Music Generator Web Interface...")
        print(f"🎧 Interface will be available at: http://{server_name}:{server_port}")
        
        self.interface.launch(
            share=share,
            server_name=server_name,
            server_port=server_port,
            show_error=True,
            quiet=False
        )


# Main execution
if __name__ == "__main__":
    # Create and launch the music web interface
    music_web_interface = MusicWebInterface()
    music_web_interface.launch(share=False, server_port=7861)
