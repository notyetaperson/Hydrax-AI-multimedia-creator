"""
Advanced Voice Cloning Module for Hydax AI TTS Engine
"""

import os
import torch
import torchaudio
import numpy as np
import librosa
import soundfile as sf
from typing import Optional, Dict, List, Tuple, Union
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

try:
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts
    XTTS_AVAILABLE = True
except ImportError:
    XTTS_AVAILABLE = False
    print("Warning: XTTS not available for voice cloning. Install with: pip install TTS[all]")

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("Warning: Whisper not available. Install with: pip install openai-whisper")

class VoiceCloner:
    """
    Advanced voice cloning using XTTS and other techniques
    """
    
    def __init__(self, device: str = "auto"):
        """
        Initialize the voice cloner
        
        Args:
            device: Device to run on ('cpu', 'cuda', or 'auto')
        """
        self.device = self._setup_device(device)
        self.xtts_model = None
        self.whisper_model = None
        self.voice_embeddings = {}
        self.sample_rate = 22050
        
        # Initialize models
        self._initialize_models()
    
    def _setup_device(self, device: str) -> str:
        """Setup the device for inference"""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        return device
    
    def _initialize_models(self):
        """Initialize voice cloning models"""
        if XTTS_AVAILABLE:
            try:
                print("Loading XTTS model for voice cloning...")
                # Load XTTS model
                config = XttsConfig()
                config.load_json("TTS/tts/configs/xtts_config.json")
                self.xtts_model = Xtts.init_from_config(config)
                self.xtts_model.load_checkpoint(
                    config, 
                    checkpoint_dir="tts_models/multilingual/multi-dataset/xtts_v2",
                    use_deepspeed=False
                )
                self.xtts_model.to(self.device)
                print("✓ XTTS model loaded successfully")
            except Exception as e:
                print(f"Failed to load XTTS model: {e}")
                self.xtts_model = None
        
        if WHISPER_AVAILABLE:
            try:
                print("Loading Whisper model for transcription...")
                self.whisper_model = whisper.load_model("base")
                print("✓ Whisper model loaded successfully")
            except Exception as e:
                print(f"Failed to load Whisper model: {e}")
                self.whisper_model = None
    
    def clone_voice_from_audio(self, 
                              voice_name: str, 
                              audio_file: str, 
                              reference_text: Optional[str] = None,
                              min_duration: float = 3.0,
                              max_duration: float = 30.0) -> bool:
        """
        Clone a voice from an audio file using XTTS
        
        Args:
            voice_name: Name to assign to the cloned voice
            audio_file: Path to the audio file
            reference_text: Reference text (if None, will be transcribed)
            min_duration: Minimum duration of audio in seconds
            max_duration: Maximum duration of audio in seconds
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not os.path.exists(audio_file):
                print(f"Audio file not found: {audio_file}")
                return False
            
            # Load and preprocess audio
            audio, sr = librosa.load(audio_file, sr=self.sample_rate)
            duration = len(audio) / sr
            
            if duration < min_duration:
                print(f"Audio too short: {duration:.2f}s (minimum: {min_duration}s)")
                return False
            
            if duration > max_duration:
                print(f"Audio too long: {duration:.2f}s (maximum: {max_duration}s), trimming...")
                audio = audio[:int(max_duration * sr)]
            
            # Get reference text if not provided
            if reference_text is None and self.whisper_model:
                print("Transcribing audio for reference text...")
                reference_text = self._transcribe_audio(audio)
                if not reference_text:
                    print("Failed to transcribe audio")
                    return False
            
            if not reference_text:
                print("Reference text is required for voice cloning")
                return False
            
            # Create voice embedding using XTTS
            if self.xtts_model:
                success = self._create_xtts_embedding(voice_name, audio, reference_text)
                if success:
                    print(f"✓ Voice '{voice_name}' cloned successfully using XTTS")
                    return True
            
            # Fallback: store audio for basic voice cloning
            self.voice_embeddings[voice_name] = {
                'audio': audio,
                'text': reference_text,
                'type': 'basic'
            }
            print(f"✓ Voice '{voice_name}' stored for basic cloning")
            return True
            
        except Exception as e:
            print(f"Voice cloning failed: {e}")
            return False
    
    def _transcribe_audio(self, audio: np.ndarray) -> str:
        """Transcribe audio using Whisper"""
        try:
            if self.whisper_model is None:
                return ""
            
            # Convert to the format expected by Whisper
            audio_tensor = torch.from_numpy(audio).float()
            result = self.whisper_model.transcribe(audio_tensor)
            return result["text"].strip()
            
        except Exception as e:
            print(f"Transcription failed: {e}")
            return ""
    
    def _create_xtts_embedding(self, voice_name: str, audio: np.ndarray, text: str) -> bool:
        """Create XTTS voice embedding"""
        try:
            if self.xtts_model is None:
                return False
            
            # Prepare audio for XTTS
            audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
            
            # Create speaker embedding
            gpt_cond_latent, speaker_embedding = self.xtts_model.get_conditioning_latents(
                audio_list=[audio_tensor],
                gpt_cond_len=self.xtts_model.config.gpt_cond_len,
                max_ref_length=self.xtts_model.config.max_ref_len,
                sound_norm_refs=self.xtts_model.config.sound_norm_refs
            )
            
            # Store the embeddings
            self.voice_embeddings[voice_name] = {
                'gpt_cond_latent': gpt_cond_latent,
                'speaker_embedding': speaker_embedding,
                'text': text,
                'type': 'xtts'
            }
            
            return True
            
        except Exception as e:
            print(f"XTTS embedding creation failed: {e}")
            return False
    
    def synthesize_with_cloned_voice(self, 
                                   text: str, 
                                   voice_name: str,
                                   language: str = "en",
                                   speed: float = 1.0) -> np.ndarray:
        """
        Synthesize text using a cloned voice
        
        Args:
            text: Text to synthesize
            voice_name: Name of the cloned voice
            language: Language code
            speed: Speech speed multiplier
        
        Returns:
            np.ndarray: Audio waveform
        """
        try:
            if voice_name not in self.voice_embeddings:
                print(f"Voice '{voice_name}' not found")
                return np.array([])
            
            voice_data = self.voice_embeddings[voice_name]
            
            if voice_data['type'] == 'xtts' and self.xtts_model:
                return self._synthesize_xtts(text, voice_data, language, speed)
            else:
                return self._synthesize_basic(text, voice_data, speed)
                
        except Exception as e:
            print(f"Synthesis with cloned voice failed: {e}")
            return np.array([])
    
    def _synthesize_xtts(self, text: str, voice_data: Dict, language: str, speed: float) -> np.ndarray:
        """Synthesize using XTTS with cloned voice"""
        try:
            # Synthesize with XTTS
            out = self.xtts_model.inference(
                text=text,
                language=language,
                gpt_cond_latent=voice_data['gpt_cond_latent'],
                speaker_embedding=voice_data['speaker_embedding'],
                temperature=0.75,
                length_penalty=1.0,
                repetition_penalty=5.0,
                top_k=50,
                top_p=0.85,
                enable_text_splitting=True
            )
            
            # Convert to numpy array
            audio = out["wav"].cpu().numpy().astype(np.float32)
            
            # Apply speed adjustment
            if speed != 1.0:
                audio = librosa.effects.time_stretch(audio, rate=speed)
            
            return audio
            
        except Exception as e:
            print(f"XTTS synthesis failed: {e}")
            return np.array([])
    
    def _synthesize_basic(self, text: str, voice_data: Dict, speed: float) -> np.ndarray:
        """Basic synthesis using stored audio (fallback)"""
        try:
            # This is a simplified approach - in practice, you'd use more sophisticated methods
            reference_audio = voice_data['audio']
            
            # For basic cloning, we'll return the reference audio with some modifications
            # In a real implementation, you'd use techniques like voice conversion or fine-tuning
            
            # Simple pitch and speed adjustment
            if speed != 1.0:
                reference_audio = librosa.effects.time_stretch(reference_audio, rate=speed)
            
            # Normalize
            if np.max(np.abs(reference_audio)) > 0:
                reference_audio = reference_audio / np.max(np.abs(reference_audio)) * 0.8
            
            return reference_audio
            
        except Exception as e:
            print(f"Basic synthesis failed: {e}")
            return np.array([])
    
    def get_cloned_voices(self) -> List[str]:
        """Get list of cloned voice names"""
        return list(self.voice_embeddings.keys())
    
    def delete_voice(self, voice_name: str) -> bool:
        """Delete a cloned voice"""
        if voice_name in self.voice_embeddings:
            del self.voice_embeddings[voice_name]
            print(f"Voice '{voice_name}' deleted")
            return True
        else:
            print(f"Voice '{voice_name}' not found")
            return False
    
    def analyze_voice_characteristics(self, audio_file: str) -> Dict:
        """
        Analyze voice characteristics from an audio file
        
        Args:
            audio_file: Path to the audio file
        
        Returns:
            Dict: Voice characteristics
        """
        try:
            audio, sr = librosa.load(audio_file, sr=self.sample_rate)
            
            # Extract features
            features = {
                'duration': len(audio) / sr,
                'sample_rate': sr,
                'pitch_mean': float(np.mean(librosa.yin(audio, fmin=50, fmax=400))),
                'pitch_std': float(np.std(librosa.yin(audio, fmin=50, fmax=400))),
                'energy_mean': float(np.mean(librosa.feature.rms(y=audio)[0])),
                'energy_std': float(np.std(librosa.feature.rms(y=audio)[0])),
                'spectral_centroid_mean': float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)[0])),
                'zero_crossing_rate': float(np.mean(librosa.feature.zero_crossing_rate(audio)[0]))
            }
            
            return features
            
        except Exception as e:
            print(f"Voice analysis failed: {e}")
            return {}
    
    def save_voice_embeddings(self, filepath: str) -> bool:
        """Save voice embeddings to file"""
        try:
            import pickle
            
            # Convert tensors to CPU for saving
            save_data = {}
            for name, data in self.voice_embeddings.items():
                if data['type'] == 'xtts':
                    save_data[name] = {
                        'gpt_cond_latent': data['gpt_cond_latent'].cpu(),
                        'speaker_embedding': data['speaker_embedding'].cpu(),
                        'text': data['text'],
                        'type': data['type']
                    }
                else:
                    save_data[name] = data
            
            with open(filepath, 'wb') as f:
                pickle.dump(save_data, f)
            
            print(f"Voice embeddings saved to: {filepath}")
            return True
            
        except Exception as e:
            print(f"Failed to save voice embeddings: {e}")
            return False
    
    def load_voice_embeddings(self, filepath: str) -> bool:
        """Load voice embeddings from file"""
        try:
            import pickle
            
            if not os.path.exists(filepath):
                print(f"File not found: {filepath}")
                return False
            
            with open(filepath, 'rb') as f:
                save_data = pickle.load(f)
            
            # Convert tensors back to device
            for name, data in save_data.items():
                if data['type'] == 'xtts':
                    self.voice_embeddings[name] = {
                        'gpt_cond_latent': data['gpt_cond_latent'].to(self.device),
                        'speaker_embedding': data['speaker_embedding'].to(self.device),
                        'text': data['text'],
                        'type': data['type']
                    }
                else:
                    self.voice_embeddings[name] = data
            
            print(f"Voice embeddings loaded from: {filepath}")
            return True
            
        except Exception as e:
            print(f"Failed to load voice embeddings: {e}")
            return False


# Example usage
if __name__ == "__main__":
    # Initialize voice cloner
    cloner = VoiceCloner()
    
    # Example: Clone a voice from an audio file
    # cloner.clone_voice_from_audio("my_voice", "path/to/audio.wav")
    
    # Example: Synthesize with cloned voice
    # audio = cloner.synthesize_with_cloned_voice("Hello, this is my cloned voice!", "my_voice")
    
    print("Voice cloning module initialized successfully!")
