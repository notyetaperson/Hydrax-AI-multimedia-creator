"""
Emotion and Style Control Module for Hydax AI TTS Engine
"""

import numpy as np
import librosa
import torch
import torchaudio
from typing import Dict, List, Optional, Tuple, Union
import warnings
warnings.filterwarnings("ignore")

try:
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts
    XTTS_AVAILABLE = True
except ImportError:
    XTTS_AVAILABLE = False

class EmotionController:
    """
    Advanced emotion and style control for TTS synthesis
    """
    
    def __init__(self):
        """Initialize the emotion controller"""
        self.emotion_presets = self._create_emotion_presets()
        self.style_presets = self._create_style_presets()
        self.sample_rate = 22050
    
    def _create_emotion_presets(self) -> Dict:
        """Create predefined emotion presets"""
        return {
            'neutral': {
                'pitch_shift': 0,
                'speed': 1.0,
                'energy': 1.0,
                'prosody': 'normal',
                'breathing': 'normal',
                'emphasis': 'normal'
            },
            'happy': {
                'pitch_shift': 2,
                'speed': 1.1,
                'energy': 1.2,
                'prosody': 'upward',
                'breathing': 'light',
                'emphasis': 'high'
            },
            'sad': {
                'pitch_shift': -2,
                'speed': 0.9,
                'energy': 0.8,
                'prosody': 'downward',
                'breathing': 'heavy',
                'emphasis': 'low'
            },
            'angry': {
                'pitch_shift': 1,
                'speed': 1.2,
                'energy': 1.4,
                'prosody': 'sharp',
                'breathing': 'short',
                'emphasis': 'very_high'
            },
            'excited': {
                'pitch_shift': 3,
                'speed': 1.3,
                'energy': 1.5,
                'prosody': 'upward',
                'breathing': 'fast',
                'emphasis': 'very_high'
            },
            'calm': {
                'pitch_shift': -1,
                'speed': 0.8,
                'energy': 0.7,
                'prosody': 'smooth',
                'breathing': 'deep',
                'emphasis': 'low'
            },
            'surprised': {
                'pitch_shift': 4,
                'speed': 1.1,
                'energy': 1.3,
                'prosody': 'sudden',
                'breathing': 'sharp',
                'emphasis': 'high'
            },
            'whisper': {
                'pitch_shift': -1,
                'speed': 0.9,
                'energy': 0.3,
                'prosody': 'soft',
                'breathing': 'quiet',
                'emphasis': 'very_low'
            },
            'shouting': {
                'pitch_shift': 2,
                'speed': 1.1,
                'energy': 2.0,
                'prosody': 'loud',
                'breathing': 'strong',
                'emphasis': 'maximum'
            }
        }
    
    def _create_style_presets(self) -> Dict:
        """Create predefined style presets"""
        return {
            'conversational': {
                'pauses': 'natural',
                'rhythm': 'irregular',
                'emphasis': 'contextual',
                'breathing': 'natural'
            },
            'formal': {
                'pauses': 'structured',
                'rhythm': 'regular',
                'emphasis': 'balanced',
                'breathing': 'controlled'
            },
            'dramatic': {
                'pauses': 'dramatic',
                'rhythm': 'varied',
                'emphasis': 'strong',
                'breathing': 'expressive'
            },
            'news': {
                'pauses': 'clear',
                'rhythm': 'steady',
                'emphasis': 'neutral',
                'breathing': 'professional'
            },
            'storytelling': {
                'pauses': 'expressive',
                'rhythm': 'flowing',
                'emphasis': 'narrative',
                'breathing': 'engaging'
            }
        }
    
    def apply_emotion(self, 
                     audio: np.ndarray, 
                     emotion: str, 
                     intensity: float = 1.0) -> np.ndarray:
        """
        Apply emotion to audio
        
        Args:
            audio: Input audio waveform
            emotion: Emotion name ('happy', 'sad', 'angry', etc.)
            intensity: Emotion intensity (0.0 to 2.0)
        
        Returns:
            np.ndarray: Emotionally modified audio
        """
        if emotion not in self.emotion_presets:
            print(f"Unknown emotion: {emotion}")
            return audio
        
        if len(audio) == 0:
            return audio
        
        preset = self.emotion_presets[emotion]
        
        # Apply emotion modifications
        modified_audio = audio.copy()
        
        # Pitch modification
        if preset['pitch_shift'] != 0:
            pitch_shift = preset['pitch_shift'] * intensity
            modified_audio = librosa.effects.pitch_shift(
                modified_audio, 
                sr=self.sample_rate, 
                n_steps=pitch_shift
            )
        
        # Speed modification
        if preset['speed'] != 1.0:
            speed_factor = 1.0 + (preset['speed'] - 1.0) * intensity
            modified_audio = librosa.effects.time_stretch(modified_audio, rate=speed_factor)
        
        # Energy modification
        if preset['energy'] != 1.0:
            energy_factor = 1.0 + (preset['energy'] - 1.0) * intensity
            modified_audio = modified_audio * energy_factor
        
        # Apply prosody modifications
        modified_audio = self._apply_prosody(modified_audio, preset['prosody'], intensity)
        
        # Apply breathing modifications
        modified_audio = self._apply_breathing(modified_audio, preset['breathing'], intensity)
        
        # Normalize to prevent clipping
        if np.max(np.abs(modified_audio)) > 0:
            modified_audio = modified_audio / np.max(np.abs(modified_audio)) * 0.8
        
        return modified_audio
    
    def apply_style(self, 
                   audio: np.ndarray, 
                   style: str, 
                   intensity: float = 1.0) -> np.ndarray:
        """
        Apply speaking style to audio
        
        Args:
            audio: Input audio waveform
            style: Style name ('conversational', 'formal', 'dramatic', etc.)
            intensity: Style intensity (0.0 to 2.0)
        
        Returns:
            np.ndarray: Style-modified audio
        """
        if style not in self.style_presets:
            print(f"Unknown style: {style}")
            return audio
        
        if len(audio) == 0:
            return audio
        
        preset = self.style_presets[style]
        modified_audio = audio.copy()
        
        # Apply style-specific modifications
        if preset['rhythm'] == 'irregular':
            modified_audio = self._add_rhythmic_variation(modified_audio, intensity)
        elif preset['rhythm'] == 'regular':
            modified_audio = self._regularize_rhythm(modified_audio, intensity)
        
        if preset['pauses'] == 'dramatic':
            modified_audio = self._add_dramatic_pauses(modified_audio, intensity)
        elif preset['pauses'] == 'structured':
            modified_audio = self._add_structured_pauses(modified_audio, intensity)
        
        return modified_audio
    
    def _apply_prosody(self, audio: np.ndarray, prosody_type: str, intensity: float) -> np.ndarray:
        """Apply prosody modifications"""
        if prosody_type == 'upward':
            # Increase pitch variation
            return self._add_pitch_variation(audio, intensity, direction='up')
        elif prosody_type == 'downward':
            # Decrease pitch variation
            return self._add_pitch_variation(audio, intensity, direction='down')
        elif prosody_type == 'sharp':
            # Add sharp pitch changes
            return self._add_sharp_pitch_changes(audio, intensity)
        elif prosody_type == 'smooth':
            # Smooth pitch changes
            return self._smooth_pitch_changes(audio, intensity)
        elif prosody_type == 'sudden':
            # Add sudden pitch changes
            return self._add_sudden_pitch_changes(audio, intensity)
        else:
            return audio
    
    def _apply_breathing(self, audio: np.ndarray, breathing_type: str, intensity: float) -> np.ndarray:
        """Apply breathing modifications"""
        if breathing_type == 'light':
            return self._add_light_breathing(audio, intensity)
        elif breathing_type == 'heavy':
            return self._add_heavy_breathing(audio, intensity)
        elif breathing_type == 'short':
            return self._add_short_breathing(audio, intensity)
        elif breathing_type == 'deep':
            return self._add_deep_breathing(audio, intensity)
        elif breathing_type == 'fast':
            return self._add_fast_breathing(audio, intensity)
        elif breathing_type == 'quiet':
            return self._add_quiet_breathing(audio, intensity)
        elif breathing_type == 'strong':
            return self._add_strong_breathing(audio, intensity)
        else:
            return audio
    
    def _add_pitch_variation(self, audio: np.ndarray, intensity: float, direction: str) -> np.ndarray:
        """Add pitch variation to audio"""
        try:
            # Extract pitch
            pitches, magnitudes = librosa.piptrack(y=audio, sr=self.sample_rate)
            pitch_values = []
            
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)
            
            if not pitch_values:
                return audio
            
            # Apply variation
            if direction == 'up':
                variation = np.random.normal(0, intensity * 50, len(pitch_values))
                pitch_values = [p + v for p, v in zip(pitch_values, variation)]
            elif direction == 'down':
                variation = np.random.normal(0, -intensity * 30, len(pitch_values))
                pitch_values = [p + v for p, v in zip(pitch_values, variation)]
            
            # Reconstruct audio with modified pitch
            return self._reconstruct_with_pitch(audio, pitch_values)
            
        except Exception as e:
            print(f"Pitch variation failed: {e}")
            return audio
    
    def _add_rhythmic_variation(self, audio: np.ndarray, intensity: float) -> np.ndarray:
        """Add rhythmic variation to audio"""
        try:
            # Add subtle tempo variations
            frame_length = int(0.1 * self.sample_rate)  # 100ms frames
            modified_audio = []
            
            for i in range(0, len(audio), frame_length):
                frame = audio[i:i + frame_length]
                if len(frame) > 0:
                    # Add slight tempo variation
                    tempo_variation = 1.0 + np.random.normal(0, intensity * 0.1)
                    if tempo_variation > 0.5:  # Prevent extreme values
                        frame = librosa.effects.time_stretch(frame, rate=tempo_variation)
                    modified_audio.extend(frame)
            
            return np.array(modified_audio)
            
        except Exception as e:
            print(f"Rhythmic variation failed: {e}")
            return audio
    
    def _add_dramatic_pauses(self, audio: np.ndarray, intensity: float) -> np.ndarray:
        """Add dramatic pauses to audio"""
        try:
            # Find natural pause points (low energy regions)
            frame_length = int(0.025 * self.sample_rate)  # 25ms frames
            hop_length = int(0.01 * self.sample_rate)     # 10ms hop
            
            rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
            threshold = np.mean(rms) * 0.3
            
            # Find low energy regions
            pause_points = []
            for i, energy in enumerate(rms):
                if energy < threshold:
                    pause_points.append(i * hop_length)
            
            # Add longer pauses at these points
            modified_audio = []
            last_end = 0
            
            for pause_point in pause_points[::5]:  # Every 5th pause point
                if pause_point > last_end:
                    # Add audio before pause
                    modified_audio.extend(audio[last_end:pause_point])
                    
                    # Add dramatic pause (silence with slight noise)
                    pause_length = int(0.3 * intensity * self.sample_rate)
                    pause = np.random.normal(0, 0.001, pause_length)
                    modified_audio.extend(pause)
                    
                    last_end = pause_point
            
            # Add remaining audio
            modified_audio.extend(audio[last_end:])
            
            return np.array(modified_audio)
            
        except Exception as e:
            print(f"Dramatic pauses failed: {e}")
            return audio
    
    def _add_light_breathing(self, audio: np.ndarray, intensity: float) -> np.ndarray:
        """Add light breathing sounds"""
        try:
            # Add subtle breathing sounds at natural pause points
            breathing_sound = self._generate_breathing_sound('light', intensity)
            
            # Find pause points and add breathing
            modified_audio = self._insert_breathing_at_pauses(audio, breathing_sound)
            
            return modified_audio
            
        except Exception as e:
            print(f"Light breathing failed: {e}")
            return audio
    
    def _generate_breathing_sound(self, breathing_type: str, intensity: float) -> np.ndarray:
        """Generate breathing sound"""
        duration = 0.5 * intensity  # Breathing duration
        samples = int(duration * self.sample_rate)
        
        if breathing_type == 'light':
            # Light breathing: low frequency noise
            breathing = np.random.normal(0, 0.01 * intensity, samples)
            # Apply low-pass filter effect
            breathing = np.convolve(breathing, np.ones(10)/10, mode='same')
        elif breathing_type == 'heavy':
            # Heavy breathing: more prominent noise
            breathing = np.random.normal(0, 0.05 * intensity, samples)
            breathing = np.convolve(breathing, np.ones(20)/20, mode='same')
        else:
            breathing = np.zeros(samples)
        
        return breathing
    
    def _insert_breathing_at_pauses(self, audio: np.ndarray, breathing_sound: np.ndarray) -> np.ndarray:
        """Insert breathing sounds at natural pause points"""
        try:
            # Find pause points
            frame_length = int(0.025 * self.sample_rate)
            hop_length = int(0.01 * self.sample_rate)
            
            rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
            threshold = np.mean(rms) * 0.2
            
            pause_points = []
            for i, energy in enumerate(rms):
                if energy < threshold and i > 10:  # Avoid very early pauses
                    pause_points.append(i * hop_length)
            
            # Insert breathing at selected pause points
            modified_audio = []
            last_end = 0
            
            for pause_point in pause_points[::10]:  # Every 10th pause
                if pause_point > last_end and pause_point < len(audio) - len(breathing_sound):
                    # Add audio before pause
                    modified_audio.extend(audio[last_end:pause_point])
                    
                    # Add breathing
                    modified_audio.extend(breathing_sound)
                    
                    last_end = pause_point
            
            # Add remaining audio
            modified_audio.extend(audio[last_end:])
            
            return np.array(modified_audio)
            
        except Exception as e:
            print(f"Breathing insertion failed: {e}")
            return audio
    
    def _reconstruct_with_pitch(self, audio: np.ndarray, pitch_values: List[float]) -> np.ndarray:
        """Reconstruct audio with modified pitch values"""
        # This is a simplified implementation
        # In practice, you'd use more sophisticated pitch modification techniques
        try:
            # Apply pitch shifting based on the pitch values
            if pitch_values:
                avg_pitch_shift = np.mean(pitch_values) - 440  # Relative to A4
                n_steps = 12 * np.log2(avg_pitch_shift / 440) if avg_pitch_shift > 0 else 0
                
                if abs(n_steps) > 0.1:  # Only apply if significant change
                    return librosa.effects.pitch_shift(audio, sr=self.sample_rate, n_steps=n_steps)
            
            return audio
            
        except Exception as e:
            print(f"Pitch reconstruction failed: {e}")
            return audio
    
    def get_available_emotions(self) -> List[str]:
        """Get list of available emotions"""
        return list(self.emotion_presets.keys())
    
    def get_available_styles(self) -> List[str]:
        """Get list of available styles"""
        return list(self.style_presets.keys())
    
    def create_custom_emotion(self, 
                            name: str, 
                            pitch_shift: float = 0,
                            speed: float = 1.0,
                            energy: float = 1.0,
                            prosody: str = 'normal',
                            breathing: str = 'normal',
                            emphasis: str = 'normal') -> bool:
        """
        Create a custom emotion preset
        
        Args:
            name: Name of the custom emotion
            pitch_shift: Pitch shift in semitones
            speed: Speed multiplier
            energy: Energy multiplier
            prosody: Prosody type
            breathing: Breathing type
            emphasis: Emphasis level
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.emotion_presets[name] = {
                'pitch_shift': pitch_shift,
                'speed': speed,
                'energy': energy,
                'prosody': prosody,
                'breathing': breathing,
                'emphasis': emphasis
            }
            print(f"Custom emotion '{name}' created successfully")
            return True
            
        except Exception as e:
            print(f"Failed to create custom emotion: {e}")
            return False


# Example usage
if __name__ == "__main__":
    # Initialize emotion controller
    emotion_controller = EmotionController()
    
    # Test with sample audio
    sample_audio = np.random.normal(0, 0.1, 22050)  # 1 second of noise
    
    # Apply different emotions
    emotions = ['happy', 'sad', 'angry', 'excited', 'calm']
    
    for emotion in emotions:
        modified_audio = emotion_controller.apply_emotion(sample_audio, emotion, intensity=1.0)
        print(f"Applied {emotion} emotion: {len(modified_audio)} samples")
    
    print("Emotion control module initialized successfully!")
