"""
Web Interface for Hydax AI TTS Engine
"""

import gradio as gr
import numpy as np
import tempfile
import os
from pathlib import Path
import json
from typing import Optional, Tuple, List

# Import our TTS modules
from tts_engine import HydaxTTS
from voice_cloning import VoiceCloner
from emotion_control import EmotionController
from audio_processing import AudioProcessor

class TTSWebInterface:
    """
    Web interface for the Hydax AI TTS Engine
    """
    
    def __init__(self):
        """Initialize the web interface"""
        self.tts_engine = HydaxTTS()
        self.voice_cloner = VoiceCloner()
        self.emotion_controller = EmotionController()
        self.audio_processor = AudioProcessor()
        
        # Create output directory
        self.output_dir = Path("outputs")
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize interface
        self.interface = self._create_interface()
    
    def _create_interface(self) -> gr.Blocks:
        """Create the Gradio interface"""
        with gr.Blocks(
            title="Hydax AI TTS Engine",
            theme=gr.themes.Soft(),
            css="""
            .gradio-container {
                max-width: 1200px !important;
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
            """
        ) as interface:
            
            gr.HTML("""
            <div class="main-header">
                <h1>🎤 Hydax AI TTS Engine</h1>
                <p>Powerful, realistic AI-powered Text-to-Speech with voice cloning and emotion control</p>
            </div>
            """)
            
            with gr.Tabs():
                # Basic TTS Tab
                with gr.Tab("🎯 Basic TTS"):
                    self._create_basic_tts_tab()
                
                # Voice Cloning Tab
                with gr.Tab("🎭 Voice Cloning"):
                    self._create_voice_cloning_tab()
                
                # Emotion Control Tab
                with gr.Tab("😊 Emotion Control"):
                    self._create_emotion_control_tab()
                
                # Audio Processing Tab
                with gr.Tab("🔧 Audio Processing"):
                    self._create_audio_processing_tab()
                
                # Batch Processing Tab
                with gr.Tab("📦 Batch Processing"):
                    self._create_batch_processing_tab()
                
                # Settings Tab
                with gr.Tab("⚙️ Settings"):
                    self._create_settings_tab()
        
        return interface
    
    def _create_basic_tts_tab(self):
        """Create the basic TTS tab"""
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### Text Input")
                text_input = gr.Textbox(
                    label="Text to synthesize",
                    placeholder="Enter the text you want to convert to speech...",
                    lines=5,
                    max_lines=10
                )
                
                with gr.Row():
                    model_type = gr.Dropdown(
                        choices=["coqui", "pyttsx3"],
                        value="coqui",
                        label="TTS Model"
                    )
                    
                    language = gr.Dropdown(
                        choices=["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko"],
                        value="en",
                        label="Language"
                    )
                
                with gr.Row():
                    speed = gr.Slider(
                        minimum=0.5,
                        maximum=2.0,
                        value=1.0,
                        step=0.1,
                        label="Speed"
                    )
                    
                    pitch = gr.Slider(
                        minimum=0.5,
                        maximum=2.0,
                        value=1.0,
                        step=0.1,
                        label="Pitch"
                    )
                
                synthesize_btn = gr.Button("🎤 Synthesize Speech", variant="primary")
                
            with gr.Column(scale=1):
                gr.Markdown("### Output")
                audio_output = gr.Audio(label="Generated Speech", type="numpy")
                
                with gr.Row():
                    download_btn = gr.Button("💾 Download Audio")
                    play_btn = gr.Button("▶️ Play Audio")
                
                gr.Markdown("### Audio Info")
                audio_info = gr.JSON(label="Audio Information")
        
        # Event handlers
        synthesize_btn.click(
            fn=self._synthesize_speech,
            inputs=[text_input, model_type, language, speed, pitch],
            outputs=[audio_output, audio_info]
        )
    
    def _create_voice_cloning_tab(self):
        """Create the voice cloning tab"""
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Voice Cloning")
                
                voice_name = gr.Textbox(
                    label="Voice Name",
                    placeholder="Enter a name for your cloned voice..."
                )
                
                voice_audio = gr.Audio(
                    label="Reference Audio",
                    type="filepath",
                    info="Upload an audio file (3-30 seconds) of the voice you want to clone"
                )
                
                reference_text = gr.Textbox(
                    label="Reference Text (Optional)",
                    placeholder="Enter the text spoken in the reference audio (leave empty for auto-transcription)...",
                    lines=3
                )
                
                clone_btn = gr.Button("🎭 Clone Voice", variant="primary")
                
                gr.Markdown("### Cloned Voices")
                cloned_voices = gr.Dropdown(
                    choices=[],
                    label="Select Cloned Voice",
                    interactive=True
                )
                
                refresh_voices_btn = gr.Button("🔄 Refresh Voices")
                
            with gr.Column(scale=2):
                gr.Markdown("### Test Cloned Voice")
                
                test_text = gr.Textbox(
                    label="Text to synthesize",
                    placeholder="Enter text to synthesize with the cloned voice...",
                    lines=4
                )
                
                with gr.Row():
                    test_language = gr.Dropdown(
                        choices=["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko"],
                        value="en",
                        label="Language"
                    )
                    
                    test_speed = gr.Slider(
                        minimum=0.5,
                        maximum=2.0,
                        value=1.0,
                        step=0.1,
                        label="Speed"
                    )
                
                test_synthesize_btn = gr.Button("🎤 Synthesize with Cloned Voice", variant="primary")
                
                test_audio_output = gr.Audio(label="Generated Speech", type="numpy")
                
                with gr.Row():
                    test_download_btn = gr.Button("💾 Download Audio")
                    delete_voice_btn = gr.Button("🗑️ Delete Voice", variant="stop")
        
        # Event handlers
        clone_btn.click(
            fn=self._clone_voice,
            inputs=[voice_name, voice_audio, reference_text],
            outputs=[cloned_voices]
        )
        
        refresh_voices_btn.click(
            fn=self._refresh_cloned_voices,
            outputs=[cloned_voices]
        )
        
        test_synthesize_btn.click(
            fn=self._synthesize_with_cloned_voice,
            inputs=[test_text, cloned_voices, test_language, test_speed],
            outputs=[test_audio_output]
        )
        
        delete_voice_btn.click(
            fn=self._delete_voice,
            inputs=[cloned_voices],
            outputs=[cloned_voices]
        )
    
    def _create_emotion_control_tab(self):
        """Create the emotion control tab"""
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Emotion & Style Control")
                
                emotion = gr.Dropdown(
                    choices=self.emotion_controller.get_available_emotions(),
                    value="neutral",
                    label="Emotion"
                )
                
                emotion_intensity = gr.Slider(
                    minimum=0.0,
                    maximum=2.0,
                    value=1.0,
                    step=0.1,
                    label="Emotion Intensity"
                )
                
                style = gr.Dropdown(
                    choices=self.emotion_controller.get_available_styles(),
                    value="conversational",
                    label="Speaking Style"
                )
                
                style_intensity = gr.Slider(
                    minimum=0.0,
                    maximum=2.0,
                    value=1.0,
                    step=0.1,
                    label="Style Intensity"
                )
                
                gr.Markdown("### Custom Emotion")
                
                custom_emotion_name = gr.Textbox(
                    label="Custom Emotion Name",
                    placeholder="Enter name for custom emotion..."
                )
                
                with gr.Row():
                    custom_pitch = gr.Slider(
                        minimum=-5,
                        maximum=5,
                        value=0,
                        step=0.5,
                        label="Pitch Shift"
                    )
                    
                    custom_speed = gr.Slider(
                        minimum=0.5,
                        maximum=2.0,
                        value=1.0,
                        step=0.1,
                        label="Speed"
                    )
                
                with gr.Row():
                    custom_energy = gr.Slider(
                        minimum=0.1,
                        maximum=2.0,
                        value=1.0,
                        step=0.1,
                        label="Energy"
                    )
                    
                    custom_prosody = gr.Dropdown(
                        choices=["normal", "upward", "downward", "sharp", "smooth", "sudden"],
                        value="normal",
                        label="Prosody"
                    )
                
                create_custom_btn = gr.Button("✨ Create Custom Emotion", variant="secondary")
                
            with gr.Column(scale=2):
                gr.Markdown("### Test Emotion & Style")
                
                emotion_text = gr.Textbox(
                    label="Text to synthesize",
                    placeholder="Enter text to synthesize with emotion and style...",
                    lines=4
                )
                
                emotion_synthesize_btn = gr.Button("🎭 Synthesize with Emotion", variant="primary")
                
                emotion_audio_output = gr.Audio(label="Generated Speech", type="numpy")
                
                with gr.Row():
                    emotion_download_btn = gr.Button("💾 Download Audio")
                    emotion_play_btn = gr.Button("▶️ Play Audio")
        
        # Event handlers
        create_custom_btn.click(
            fn=self._create_custom_emotion,
            inputs=[custom_emotion_name, custom_pitch, custom_speed, custom_energy, custom_prosody],
            outputs=[emotion]
        )
        
        emotion_synthesize_btn.click(
            fn=self._synthesize_with_emotion,
            inputs=[emotion_text, emotion, emotion_intensity, style, style_intensity],
            outputs=[emotion_audio_output]
        )
    
    def _create_audio_processing_tab(self):
        """Create the audio processing tab"""
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Audio Enhancement")
                
                enhancement_type = gr.Dropdown(
                    choices=["light", "medium", "full", "custom"],
                    value="medium",
                    label="Enhancement Type"
                )
                
                with gr.Accordion("Custom Enhancement Options", open=False):
                    noise_reduction = gr.Checkbox(value=True, label="Noise Reduction")
                    normalize_audio = gr.Checkbox(value=True, label="Normalize Audio")
                    remove_silence = gr.Checkbox(value=True, label="Remove Silence")
                    enhance_clarity = gr.Checkbox(value=True, label="Enhance Clarity")
                
                gr.Markdown("### Audio Effects")
                
                reverb_type = gr.Dropdown(
                    choices=["none", "room", "hall", "plate", "spring"],
                    value="none",
                    label="Reverb Type"
                )
                
                reverb_intensity = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.3,
                    step=0.1,
                    label="Reverb Intensity"
                )
                
                eq_type = gr.Dropdown(
                    choices=["none", "speech", "music", "bright", "warm"],
                    value="speech",
                    label="Equalizer"
                )
                
                process_btn = gr.Button("🔧 Process Audio", variant="primary")
                
            with gr.Column(scale=2):
                gr.Markdown("### Audio Input & Output")
                
                input_audio = gr.Audio(
                    label="Input Audio",
                    type="numpy",
                    info="Upload an audio file to process"
                )
                
                processed_audio = gr.Audio(label="Processed Audio", type="numpy")
                
                with gr.Row():
                    process_download_btn = gr.Button("💾 Download Processed Audio")
                    analyze_btn = gr.Button("📊 Analyze Audio Quality")
                
                gr.Markdown("### Audio Analysis")
                audio_analysis = gr.JSON(label="Audio Quality Metrics")
        
        # Event handlers
        process_btn.click(
            fn=self._process_audio,
            inputs=[input_audio, enhancement_type, noise_reduction, normalize_audio, remove_silence, enhance_clarity, reverb_type, reverb_intensity, eq_type],
            outputs=[processed_audio, audio_analysis]
        )
        
        analyze_btn.click(
            fn=self._analyze_audio,
            inputs=[input_audio],
            outputs=[audio_analysis]
        )
    
    def _create_batch_processing_tab(self):
        """Create the batch processing tab"""
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Batch Processing")
                
                batch_texts = gr.Textbox(
                    label="Texts to process (one per line)",
                    placeholder="Enter multiple texts, one per line...",
                    lines=10
                )
                
                with gr.Row():
                    batch_model = gr.Dropdown(
                        choices=["coqui", "pyttsx3"],
                        value="coqui",
                        label="TTS Model"
                    )
                    
                    batch_language = gr.Dropdown(
                        choices=["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko"],
                        value="en",
                        label="Language"
                    )
                
                with gr.Row():
                    batch_speed = gr.Slider(
                        minimum=0.5,
                        maximum=2.0,
                        value=1.0,
                        step=0.1,
                        label="Speed"
                    )
                    
                    batch_pitch = gr.Slider(
                        minimum=0.5,
                        maximum=2.0,
                        value=1.0,
                        step=0.1,
                        label="Pitch"
                    )
                
                batch_emotion = gr.Dropdown(
                    choices=self.emotion_controller.get_available_emotions(),
                    value="neutral",
                    label="Emotion"
                )
                
                batch_cloned_voice = gr.Dropdown(
                    choices=[],
                    label="Cloned Voice (Optional)",
                    value=None
                )
                
                batch_process_btn = gr.Button("📦 Process Batch", variant="primary")
                
                refresh_batch_voices_btn = gr.Button("🔄 Refresh Voices")
                
            with gr.Column(scale=2):
                gr.Markdown("### Batch Results")
                
                batch_progress = gr.Progress()
                batch_status = gr.Textbox(label="Processing Status", interactive=False)
                
                batch_results = gr.File(
                    label="Download Batch Results",
                    file_count="multiple"
                )
        
        # Event handlers
        batch_process_btn.click(
            fn=self._process_batch,
            inputs=[batch_texts, batch_model, batch_language, batch_speed, batch_pitch, batch_emotion, batch_cloned_voice],
            outputs=[batch_results, batch_status]
        )
        
        refresh_batch_voices_btn.click(
            fn=self._refresh_cloned_voices,
            outputs=[batch_cloned_voice]
        )
    
    def _create_settings_tab(self):
        """Create the settings tab"""
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### TTS Engine Settings")
                
                device = gr.Dropdown(
                    choices=["auto", "cpu", "cuda"],
                    value="auto",
                    label="Device"
                )
                
                sample_rate = gr.Slider(
                    minimum=16000,
                    maximum=48000,
                    value=22050,
                    step=1000,
                    label="Sample Rate"
                )
                
                gr.Markdown("### Available Models")
                available_models = gr.Dropdown(
                    choices=[],
                    label="Available TTS Models",
                    interactive=False
                )
                
                refresh_models_btn = gr.Button("🔄 Refresh Models")
                
                gr.Markdown("### Voice Management")
                
                with gr.Row():
                    export_voices_btn = gr.Button("📤 Export Voices")
                    import_voices_btn = gr.Button("📥 Import Voices")
                
                voices_file = gr.File(
                    label="Voices File",
                    file_count="single"
                )
                
            with gr.Column(scale=2):
                gr.Markdown("### System Information")
                
                system_info = gr.JSON(label="System Information")
                
                gr.Markdown("### Audio Cache")
                
                with gr.Row():
                    clear_cache_btn = gr.Button("🗑️ Clear Cache")
                    cache_info_btn = gr.Button("📊 Cache Info")
                
                cache_info = gr.Textbox(label="Cache Information", interactive=False)
        
        # Event handlers
        refresh_models_btn.click(
            fn=self._refresh_models,
            outputs=[available_models]
        )
        
        export_voices_btn.click(
            fn=self._export_voices,
            outputs=[voices_file]
        )
        
        import_voices_btn.click(
            fn=self._import_voices,
            inputs=[voices_file],
            outputs=[voices_file]
        )
        
        clear_cache_btn.click(
            fn=self._clear_cache,
            outputs=[cache_info]
        )
        
        cache_info_btn.click(
            fn=self._get_cache_info,
            outputs=[cache_info]
        )
    
    # Event handler methods
    def _synthesize_speech(self, text: str, model_type: str, language: str, speed: float, pitch: float) -> Tuple[np.ndarray, dict]:
        """Synthesize speech from text"""
        try:
            if not text.strip():
                return None, {"error": "No text provided"}
            
            audio = self.tts_engine.synthesize(
                text=text,
                language=language,
                speed=speed,
                pitch=pitch,
                model_type=model_type
            )
            
            if len(audio) > 0:
                info = self.tts_engine.get_audio_info(audio)
                return (self.sample_rate, audio), info
            else:
                return None, {"error": "Failed to synthesize audio"}
                
        except Exception as e:
            return None, {"error": str(e)}
    
    def _clone_voice(self, voice_name: str, audio_file: str, reference_text: str) -> List[str]:
        """Clone a voice from audio file"""
        try:
            if not voice_name or not audio_file:
                return self._refresh_cloned_voices()
            
            success = self.voice_cloner.clone_voice_from_audio(
                voice_name=voice_name,
                audio_file=audio_file,
                reference_text=reference_text if reference_text.strip() else None
            )
            
            if success:
                return self._refresh_cloned_voices()
            else:
                return self._refresh_cloned_voices()
                
        except Exception as e:
            print(f"Voice cloning error: {e}")
            return self._refresh_cloned_voices()
    
    def _refresh_cloned_voices(self) -> List[str]:
        """Refresh the list of cloned voices"""
        try:
            voices = self.voice_cloner.get_cloned_voices()
            return gr.update(choices=voices, value=voices[0] if voices else None)
        except Exception as e:
            print(f"Error refreshing voices: {e}")
            return gr.update(choices=[], value=None)
    
    def _synthesize_with_cloned_voice(self, text: str, voice_name: str, language: str, speed: float) -> Tuple[int, np.ndarray]:
        """Synthesize speech with cloned voice"""
        try:
            if not text.strip() or not voice_name:
                return None
            
            audio = self.voice_cloner.synthesize_with_cloned_voice(
                text=text,
                voice_name=voice_name,
                language=language,
                speed=speed
            )
            
            if len(audio) > 0:
                return (self.sample_rate, audio)
            else:
                return None
                
        except Exception as e:
            print(f"Cloned voice synthesis error: {e}")
            return None
    
    def _delete_voice(self, voice_name: str) -> List[str]:
        """Delete a cloned voice"""
        try:
            if voice_name:
                self.voice_cloner.delete_voice(voice_name)
            return self._refresh_cloned_voices()
        except Exception as e:
            print(f"Error deleting voice: {e}")
            return self._refresh_cloned_voices()
    
    def _create_custom_emotion(self, name: str, pitch: float, speed: float, energy: float, prosody: str) -> List[str]:
        """Create a custom emotion"""
        try:
            if name:
                success = self.emotion_controller.create_custom_emotion(
                    name=name,
                    pitch_shift=pitch,
                    speed=speed,
                    energy=energy,
                    prosody=prosody
                )
                if success:
                    return gr.update(choices=self.emotion_controller.get_available_emotions(), value=name)
            
            return gr.update(choices=self.emotion_controller.get_available_emotions())
        except Exception as e:
            print(f"Error creating custom emotion: {e}")
            return gr.update(choices=self.emotion_controller.get_available_emotions())
    
    def _synthesize_with_emotion(self, text: str, emotion: str, emotion_intensity: float, style: str, style_intensity: float) -> Tuple[int, np.ndarray]:
        """Synthesize speech with emotion and style"""
        try:
            if not text.strip():
                return None
            
            # First synthesize basic audio
            audio = self.tts_engine.synthesize(text=text)
            
            if len(audio) > 0:
                # Apply emotion
                if emotion != "neutral":
                    audio = self.emotion_controller.apply_emotion(audio, emotion, emotion_intensity)
                
                # Apply style
                if style != "conversational":
                    audio = self.emotion_controller.apply_style(audio, style, style_intensity)
                
                return (self.sample_rate, audio)
            else:
                return None
                
        except Exception as e:
            print(f"Emotion synthesis error: {e}")
            return None
    
    def _process_audio(self, audio: Tuple[int, np.ndarray], enhancement_type: str, noise_reduction: bool, 
                      normalize_audio: bool, remove_silence: bool, enhance_clarity: bool,
                      reverb_type: str, reverb_intensity: float, eq_type: str) -> Tuple[Tuple[int, np.ndarray], dict]:
        """Process audio with enhancements"""
        try:
            if audio is None:
                return None, {"error": "No audio provided"}
            
            sample_rate, audio_data = audio
            
            # Apply enhancements
            processed_audio = self.audio_processor.enhance_audio(
                audio=audio_data,
                enhancement_type=enhancement_type,
                noise_reduction=noise_reduction,
                normalize_audio=normalize_audio,
                remove_silence=remove_silence,
                enhance_clarity=enhance_clarity
            )
            
            # Apply reverb
            if reverb_type != "none":
                processed_audio = self.audio_processor.add_reverb(processed_audio, reverb_type, reverb_intensity)
            
            # Apply EQ
            if eq_type != "none":
                processed_audio = self.audio_processor.apply_eq(processed_audio, eq_type)
            
            # Analyze processed audio
            analysis = self.audio_processor.analyze_audio_quality(processed_audio)
            
            return (sample_rate, processed_audio), analysis
            
        except Exception as e:
            return None, {"error": str(e)}
    
    def _analyze_audio(self, audio: Tuple[int, np.ndarray]) -> dict:
        """Analyze audio quality"""
        try:
            if audio is None:
                return {"error": "No audio provided"}
            
            sample_rate, audio_data = audio
            return self.audio_processor.analyze_audio_quality(audio_data)
            
        except Exception as e:
            return {"error": str(e)}
    
    def _process_batch(self, texts: str, model_type: str, language: str, speed: float, 
                      pitch: float, emotion: str, cloned_voice: str) -> Tuple[List[str], str]:
        """Process multiple texts in batch"""
        try:
            if not texts.strip():
                return [], "No texts provided"
            
            text_list = [text.strip() for text in texts.split('\n') if text.strip()]
            
            if not text_list:
                return [], "No valid texts found"
            
            results = []
            status = f"Processing {len(text_list)} texts..."
            
            for i, text in enumerate(text_list):
                try:
                    # Synthesize audio
                    if cloned_voice:
                        audio = self.voice_cloner.synthesize_with_cloned_voice(
                            text=text,
                            voice_name=cloned_voice,
                            language=language,
                            speed=speed
                        )
                    else:
                        audio = self.tts_engine.synthesize(
                            text=text,
                            language=language,
                            speed=speed,
                            pitch=pitch,
                            model_type=model_type
                        )
                    
                    if len(audio) > 0:
                        # Apply emotion if specified
                        if emotion != "neutral":
                            audio = self.emotion_controller.apply_emotion(audio, emotion)
                        
                        # Save audio file
                        filename = f"batch_{i+1:03d}.wav"
                        filepath = self.output_dir / filename
                        
                        if self.tts_engine.save_audio(audio, str(filepath)):
                            results.append(str(filepath))
                    
                    status = f"Processed {i+1}/{len(text_list)} texts..."
                    
                except Exception as e:
                    print(f"Error processing text {i+1}: {e}")
                    continue
            
            status = f"Completed! Generated {len(results)} audio files."
            return results, status
            
        except Exception as e:
            return [], f"Batch processing error: {e}"
    
    def _refresh_models(self) -> List[str]:
        """Refresh available TTS models"""
        try:
            models = self.tts_engine.list_available_models()
            return gr.update(choices=models)
        except Exception as e:
            print(f"Error refreshing models: {e}")
            return gr.update(choices=[])
    
    def _export_voices(self) -> str:
        """Export voice embeddings"""
        try:
            filepath = self.output_dir / "voice_embeddings.pkl"
            if self.voice_cloner.save_voice_embeddings(str(filepath)):
                return str(filepath)
            else:
                return None
        except Exception as e:
            print(f"Error exporting voices: {e}")
            return None
    
    def _import_voices(self, file_path: str) -> str:
        """Import voice embeddings"""
        try:
            if file_path and self.voice_cloner.load_voice_embeddings(file_path):
                return "Voices imported successfully!"
            else:
                return "Failed to import voices"
        except Exception as e:
            return f"Import error: {e}"
    
    def _clear_cache(self) -> str:
        """Clear audio cache"""
        try:
            self.tts_engine.clear_cache()
            return "Cache cleared successfully!"
        except Exception as e:
            return f"Cache clear error: {e}"
    
    def _get_cache_info(self) -> str:
        """Get cache information"""
        try:
            cache_size = len(self.tts_engine.audio_cache)
            return f"Cache contains {cache_size} items"
        except Exception as e:
            return f"Cache info error: {e}"
    
    def launch(self, share: bool = False, server_name: str = "127.0.0.1", server_port: int = 7860):
        """Launch the web interface"""
        print("🚀 Launching Hydax AI TTS Engine Web Interface...")
        print(f"📱 Interface will be available at: http://{server_name}:{server_port}")
        
        self.interface.launch(
            share=share,
            server_name=server_name,
            server_port=server_port,
            show_error=True,
            quiet=False
        )


# Main execution
if __name__ == "__main__":
    # Create and launch the web interface
    web_interface = TTSWebInterface()
    web_interface.launch(share=False, server_port=7860)
