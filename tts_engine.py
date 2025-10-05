"""
Hydax AI TTS Engine - A powerful, realistic AI-powered Text-to-Speech engine
"""

import os
import torch
import numpy as np
import soundfile as sf
import librosa
from typing import Optional, Dict, List, Union, Tuple
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

try:
    from TTS.api import TTS
    from TTS.utils.manage import ModelManager
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    print("Warning: Coqui TTS not available. Install with: pip install TTS")

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False
    print("Warning: pyttsx3 not available. Install with: pip install pyttsx3")

class HydaxTTS:
    """
    Main TTS Engine class supporting multiple AI models and voice cloning
    """
    
    def __init__(self, 
                 model_name: str = "tts_models/en/ljspeech/tacotron2-DDC",
                 device: str = "auto",
                 use_gpu: bool = True):
        """
        Initialize the Hydax TTS Engine
        
        Args:
            model_name: Name of the TTS model to use
            device: Device to run on ('cpu', 'cuda', or 'auto')
            use_gpu: Whether to use GPU if available
        """
        self.model_name = model_name
        self.device = self._setup_device(device, use_gpu)
        self.models = {}
        self.voice_clones = {}
        self.audio_cache = {}
        
        # Initialize default model
        self._initialize_model()
        
        # Audio processing parameters
        self.sample_rate = 22050
        self.target_length = None
        
    def _setup_device(self, device: str, use_gpu: bool) -> str:
        """Setup the device for inference"""
        if device == "auto":
            if use_gpu and torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        return device
    
    def _initialize_model(self):
        """Initialize the TTS model"""
        if TTS_AVAILABLE:
            try:
                print(f"Loading TTS model: {self.model_name}")
                self.models['coqui'] = TTS(model_name=self.model_name, progress_bar=False)
                print("✓ Coqui TTS model loaded successfully")
            except Exception as e:
                print(f"Failed to load Coqui TTS model: {e}")
                self.models['coqui'] = None
        
        if PYTTSX3_AVAILABLE:
            try:
                self.models['pyttsx3'] = pyttsx3.init()
                # Configure pyttsx3 settings
                voices = self.models['pyttsx3'].getProperty('voices')
                if voices:
                    self.models['pyttsx3'].setProperty('voice', voices[0].id)
                self.models['pyttsx3'].setProperty('rate', 150)  # Speed
                self.models['pyttsx3'].setProperty('volume', 0.9)  # Volume
                print("✓ pyttsx3 engine initialized")
            except Exception as e:
                print(f"Failed to initialize pyttsx3: {e}")
                self.models['pyttsx3'] = None
    
    def list_available_models(self) -> List[str]:
        """List all available TTS models"""
        if TTS_AVAILABLE:
            try:
                manager = ModelManager()
                models = manager.list_tts_models()
                return models
            except:
                return []
        return []
    
    def load_model(self, model_name: str, model_type: str = "coqui") -> bool:
        """
        Load a specific TTS model
        
        Args:
            model_name: Name of the model to load
            model_type: Type of model ('coqui', 'pyttsx3', etc.)
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if model_type == "coqui" and TTS_AVAILABLE:
                self.models['coqui'] = TTS(model_name=model_name, progress_bar=False)
                self.model_name = model_name
                print(f"✓ Loaded model: {model_name}")
                return True
            else:
                print(f"Model type {model_type} not supported or not available")
                return False
        except Exception as e:
            print(f"Failed to load model {model_name}: {e}")
            return False
    
    def synthesize(self, 
                   text: str, 
                   speaker: Optional[str] = None,
                   language: str = "en",
                   emotion: str = "neutral",
                   speed: float = 1.0,
                   pitch: float = 1.0,
                   model_type: str = "coqui") -> np.ndarray:
        """
        Synthesize speech from text
        
        Args:
            text: Text to synthesize
            speaker: Speaker voice to use (for voice cloning)
            language: Language code
            emotion: Emotion/style of speech
            speed: Speech speed multiplier
            pitch: Pitch multiplier
            model_type: Type of model to use
        
        Returns:
            np.ndarray: Audio waveform
        """
        if not text.strip():
            return np.array([])
        
        # Check cache first
        cache_key = f"{text}_{speaker}_{language}_{emotion}_{speed}_{pitch}_{model_type}"
        if cache_key in self.audio_cache:
            return self.audio_cache[cache_key]
        
        audio = None
        
        if model_type == "coqui" and self.models.get('coqui'):
            audio = self._synthesize_coqui(text, speaker, language, emotion, speed, pitch)
        elif model_type == "pyttsx3" and self.models.get('pyttsx3'):
            audio = self._synthesize_pyttsx3(text, speed, pitch)
        
        if audio is None:
            print(f"Failed to synthesize with {model_type}, trying fallback...")
            # Try fallback models
            for fallback_type in ['coqui', 'pyttsx3']:
                if fallback_type != model_type and self.models.get(fallback_type):
                    audio = self.synthesize(text, speaker, language, emotion, speed, pitch, fallback_type)
                    if audio is not None:
                        break
        
        if audio is not None:
            # Apply post-processing
            audio = self._post_process_audio(audio, speed, pitch)
            # Cache the result
            self.audio_cache[cache_key] = audio
        
        return audio if audio is not None else np.array([])
    
    def _synthesize_coqui(self, text: str, speaker: str, language: str, 
                         emotion: str, speed: float, pitch: float) -> np.ndarray:
        """Synthesize using Coqui TTS"""
        try:
            model = self.models['coqui']
            
            # Prepare synthesis parameters
            kwargs = {}
            if speaker and speaker in self.voice_clones:
                kwargs['speaker_wav'] = self.voice_clones[speaker]
            elif speaker:
                kwargs['speaker'] = speaker
            
            # Synthesize
            wav = model.tts(text, **kwargs)
            
            # Convert to numpy array
            if isinstance(wav, torch.Tensor):
                wav = wav.cpu().numpy()
            
            return wav.astype(np.float32)
            
        except Exception as e:
            print(f"Coqui TTS synthesis failed: {e}")
            return None
    
    def _synthesize_pyttsx3(self, text: str, speed: float, pitch: float) -> np.ndarray:
        """Synthesize using pyttsx3 (fallback)"""
        try:
            engine = self.models['pyttsx3']
            
            # Set properties
            engine.setProperty('rate', int(150 * speed))
            
            # Save to temporary file
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                engine.save_to_file(text, tmp_file.name)
                engine.runAndWait()
                
                # Load the generated audio
                audio, sr = librosa.load(tmp_file.name, sr=self.sample_rate)
                os.unlink(tmp_file.name)
                
                return audio.astype(np.float32)
                
        except Exception as e:
            print(f"pyttsx3 synthesis failed: {e}")
            return None
    
    def _post_process_audio(self, audio: np.ndarray, speed: float, pitch: float) -> np.ndarray:
        """Apply post-processing to the audio"""
        if len(audio) == 0:
            return audio
        
        # Speed adjustment
        if speed != 1.0:
            audio = librosa.effects.time_stretch(audio, rate=speed)
        
        # Pitch adjustment
        if pitch != 1.0:
            audio = librosa.effects.pitch_shift(audio, sr=self.sample_rate, n_steps=12 * np.log2(pitch))
        
        # Normalize audio
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio)) * 0.8
        
        return audio
    
    def clone_voice(self, voice_name: str, audio_file: str) -> bool:
        """
        Clone a voice from an audio file
        
        Args:
            voice_name: Name to assign to the cloned voice
            audio_file: Path to the audio file
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not os.path.exists(audio_file):
                print(f"Audio file not found: {audio_file}")
                return False
            
            # Load and preprocess the audio
            audio, sr = librosa.load(audio_file, sr=self.sample_rate)
            
            # Save the processed audio for voice cloning
            self.voice_clones[voice_name] = audio_file
            print(f"✓ Voice '{voice_name}' cloned successfully")
            return True
            
        except Exception as e:
            print(f"Voice cloning failed: {e}")
            return False
    
    def save_audio(self, audio: np.ndarray, filename: str, format: str = "wav") -> bool:
        """
        Save audio to file
        
        Args:
            audio: Audio waveform
            filename: Output filename
            format: Audio format ('wav', 'mp3', 'flac')
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if len(audio) == 0:
                print("No audio to save")
                return False
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
            
            # Save audio
            if format.lower() == "wav":
                sf.write(filename, audio, self.sample_rate)
            elif format.lower() == "mp3":
                from pydub import AudioSegment
                # Convert to pydub format and export as mp3
                audio_int16 = (audio * 32767).astype(np.int16)
                audio_segment = AudioSegment(
                    audio_int16.tobytes(),
                    frame_rate=self.sample_rate,
                    sample_width=2,
                    channels=1
                )
                audio_segment.export(filename, format="mp3")
            else:
                sf.write(filename, audio, self.sample_rate)
            
            print(f"✓ Audio saved to: {filename}")
            return True
            
        except Exception as e:
            print(f"Failed to save audio: {e}")
            return False
    
    def batch_synthesize(self, texts: List[str], **kwargs) -> List[np.ndarray]:
        """
        Synthesize multiple texts in batch
        
        Args:
            texts: List of texts to synthesize
            **kwargs: Additional arguments for synthesis
        
        Returns:
            List[np.ndarray]: List of audio waveforms
        """
        results = []
        for i, text in enumerate(texts):
            print(f"Processing {i+1}/{len(texts)}: {text[:50]}...")
            audio = self.synthesize(text, **kwargs)
            results.append(audio)
        return results
    
    def get_audio_info(self, audio: np.ndarray) -> Dict:
        """Get information about the audio"""
        if len(audio) == 0:
            return {"duration": 0, "sample_rate": 0, "channels": 0}
        
        duration = len(audio) / self.sample_rate
        return {
            "duration": duration,
            "sample_rate": self.sample_rate,
            "channels": 1 if audio.ndim == 1 else audio.shape[1],
            "samples": len(audio),
            "max_amplitude": float(np.max(np.abs(audio)))
        }
    
    def clear_cache(self):
        """Clear the audio cache"""
        self.audio_cache.clear()
        print("Audio cache cleared")
    
    def get_available_voices(self) -> List[str]:
        """Get list of available voices (including cloned voices)"""
        voices = []
        
        # Add cloned voices
        voices.extend(self.voice_clones.keys())
        
        # Add default voices if available
        if self.models.get('coqui'):
            try:
                # This would depend on the specific model
                voices.append("default")
            except:
                pass
        
        return voices


# Example usage and testing
if __name__ == "__main__":
    # Initialize TTS engine
    tts = HydaxTTS()
    
    # Test basic synthesis
    print("Testing basic synthesis...")
    audio = tts.synthesize("Hello, this is a test of the Hydax AI TTS engine. How does it sound?")
    
    if len(audio) > 0:
        # Save the result
        tts.save_audio(audio, "test_output.wav")
        
        # Get audio info
        info = tts.get_audio_info(audio)
        print(f"Generated audio: {info['duration']:.2f}s, {info['samples']} samples")
    else:
        print("Failed to generate audio")
    
    # Test batch processing
    print("\nTesting batch processing...")
    texts = [
        "This is the first test sentence.",
        "This is the second test sentence.",
        "This is the third test sentence."
    ]
    
    batch_results = tts.batch_synthesize(texts)
    for i, audio in enumerate(batch_results):
        if len(audio) > 0:
            tts.save_audio(audio, f"batch_test_{i+1}.wav")
    
    print("TTS Engine test completed!")
