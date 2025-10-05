"""
Advanced Music Post-Processing and Enhancement for Hydax AI Music Generator
"""

import numpy as np
import librosa
import soundfile as sf
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings("ignore")

try:
    from scipy import signal
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available. Install with: pip install scipy")

try:
    from pydub import AudioSegment
    from pydub.effects import normalize, compress_dynamic_range
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    print("Warning: pydub not available. Install with: pip install pydub")

try:
    import noisereduce as nr
    NOISE_REDUCE_AVAILABLE = True
except ImportError:
    NOISE_REDUCE_AVAILABLE = False
    print("Warning: noisereduce not available. Install with: pip install noisereduce")

class MusicProcessor:
    """
    Advanced music post-processing and enhancement engine
    """
    
    def __init__(self, sample_rate: int = 44100):
        """
        Initialize the music processor
        
        Args:
            sample_rate: Audio sample rate
        """
        self.sample_rate = sample_rate
        self.mastering_presets = self._create_mastering_presets()
        self.effect_chains = self._create_effect_chains()
        
    def _create_mastering_presets(self) -> Dict:
        """Create mastering presets for different music styles"""
        return {
            'club': {
                'compression': {'ratio': 4.0, 'threshold': -12, 'attack': 0.003, 'release': 0.1},
                'eq': {'low_shelf': {'freq': 100, 'gain': 2}, 'high_shelf': {'freq': 10000, 'gain': 1}},
                'limiter': {'threshold': -0.3, 'release': 0.01},
                'stereo_width': 1.2,
                'loudness': -14  # LUFS
            },
            'radio': {
                'compression': {'ratio': 3.0, 'threshold': -8, 'attack': 0.01, 'release': 0.1},
                'eq': {'low_shelf': {'freq': 80, 'gain': 1}, 'high_shelf': {'freq': 12000, 'gain': 2}},
                'limiter': {'threshold': -1.0, 'release': 0.05},
                'stereo_width': 1.0,
                'loudness': -16  # LUFS
            },
            'streaming': {
                'compression': {'ratio': 2.5, 'threshold': -6, 'attack': 0.005, 'release': 0.08},
                'eq': {'low_shelf': {'freq': 60, 'gain': 1.5}, 'high_shelf': {'freq': 15000, 'gain': 1.5}},
                'limiter': {'threshold': -1.5, 'release': 0.03},
                'stereo_width': 1.1,
                'loudness': -14  # LUFS
            },
            'cinematic': {
                'compression': {'ratio': 2.0, 'threshold': -4, 'attack': 0.02, 'release': 0.2},
                'eq': {'low_shelf': {'freq': 40, 'gain': 3}, 'high_shelf': {'freq': 8000, 'gain': 1}},
                'limiter': {'threshold': -3.0, 'release': 0.1},
                'stereo_width': 1.5,
                'loudness': -20  # LUFS
            },
            'ambient': {
                'compression': {'ratio': 1.5, 'threshold': -2, 'attack': 0.05, 'release': 0.5},
                'eq': {'low_shelf': {'freq': 20, 'gain': 2}, 'high_shelf': {'freq': 5000, 'gain': 0.5}},
                'limiter': {'threshold': -6.0, 'release': 0.2},
                'stereo_width': 2.0,
                'loudness': -18  # LUFS
            }
        }
    
    def _create_effect_chains(self) -> Dict:
        """Create effect chains for different music styles"""
        return {
            'electronic': [
                {'type': 'compression', 'params': {'ratio': 3.0, 'threshold': -8}},
                {'type': 'eq', 'params': {'low_shelf': {'freq': 100, 'gain': 2}}},
                {'type': 'distortion', 'params': {'drive': 0.3, 'tone': 0.5}},
                {'type': 'reverb', 'params': {'room_size': 0.3, 'damping': 0.5}},
                {'type': 'stereo_width', 'params': {'width': 1.2}}
            ],
            'ambient': [
                {'type': 'reverb', 'params': {'room_size': 0.8, 'damping': 0.2}},
                {'type': 'delay', 'params': {'time': 0.5, 'feedback': 0.4}},
                {'type': 'lowpass', 'params': {'cutoff': 0.7, 'resonance': 0.3}},
                {'type': 'compression', 'params': {'ratio': 1.5, 'threshold': -2}},
                {'type': 'stereo_width', 'params': {'width': 2.0}}
            ],
            'rock': [
                {'type': 'distortion', 'params': {'drive': 0.7, 'tone': 0.4}},
                {'type': 'compression', 'params': {'ratio': 4.0, 'threshold': -12}},
                {'type': 'eq', 'params': {'low_shelf': {'freq': 80, 'gain': 2}}},
                {'type': 'reverb', 'params': {'room_size': 0.4, 'damping': 0.6}},
                {'type': 'stereo_width', 'params': {'width': 1.0}}
            ],
            'jazz': [
                {'type': 'compression', 'params': {'ratio': 2.0, 'threshold': -4}},
                {'type': 'eq', 'params': {'low_shelf': {'freq': 60, 'gain': 1}}},
                {'type': 'reverb', 'params': {'room_size': 0.6, 'damping': 0.4}},
                {'type': 'stereo_width', 'params': {'width': 1.1}}
            ],
            'cinematic': [
                {'type': 'reverb', 'params': {'room_size': 1.0, 'damping': 0.1}},
                {'type': 'eq', 'params': {'low_shelf': {'freq': 40, 'gain': 3}}},
                {'type': 'compression', 'params': {'ratio': 2.5, 'threshold': -6}},
                {'type': 'stereo_width', 'params': {'width': 1.5}},
                {'type': 'limiter', 'params': {'threshold': -3.0}}
            ]
        }
    
    def master_music(self, 
                    audio: np.ndarray, 
                    preset: str = "streaming",
                    custom_params: Optional[Dict] = None) -> np.ndarray:
        """
        Master music using predefined or custom presets
        
        Args:
            audio: Input audio waveform
            preset: Mastering preset ('club', 'radio', 'streaming', 'cinematic', 'ambient')
            custom_params: Custom mastering parameters
        
        Returns:
            np.ndarray: Mastered audio
        """
        try:
            if len(audio) == 0:
                return audio
            
            # Get mastering parameters
            if custom_params:
                params = custom_params
            else:
                params = self.mastering_presets.get(preset, self.mastering_presets['streaming'])
            
            mastered_audio = audio.copy()
            
            # Apply compression
            if 'compression' in params:
                mastered_audio = self._apply_compression(
                    mastered_audio, 
                    params['compression']
                )
            
            # Apply EQ
            if 'eq' in params:
                mastered_audio = self._apply_eq(
                    mastered_audio, 
                    params['eq']
                )
            
            # Apply stereo width
            if 'stereo_width' in params:
                mastered_audio = self._apply_stereo_width(
                    mastered_audio, 
                    params['stereo_width']
                )
            
            # Apply limiter
            if 'limiter' in params:
                mastered_audio = self._apply_limiter(
                    mastered_audio, 
                    params['limiter']
                )
            
            # Normalize to target loudness
            if 'loudness' in params:
                mastered_audio = self._normalize_loudness(
                    mastered_audio, 
                    params['loudness']
                )
            
            return mastered_audio
            
        except Exception as e:
            print(f"Mastering failed: {e}")
            return audio
    
    def apply_effect_chain(self, 
                          audio: np.ndarray, 
                          style: str = "electronic",
                          custom_chain: Optional[List[Dict]] = None) -> np.ndarray:
        """
        Apply a chain of effects to audio
        
        Args:
            audio: Input audio waveform
            style: Effect chain style
            custom_chain: Custom effect chain
        
        Returns:
            np.ndarray: Processed audio
        """
        try:
            if len(audio) == 0:
                return audio
            
            # Get effect chain
            if custom_chain:
                chain = custom_chain
            else:
                chain = self.effect_chains.get(style, self.effect_chains['electronic'])
            
            processed_audio = audio.copy()
            
            # Apply each effect in the chain
            for effect in chain:
                effect_type = effect['type']
                effect_params = effect['params']
                
                if effect_type == 'compression':
                    processed_audio = self._apply_compression(processed_audio, effect_params)
                elif effect_type == 'eq':
                    processed_audio = self._apply_eq(processed_audio, effect_params)
                elif effect_type == 'reverb':
                    processed_audio = self._apply_reverb(processed_audio, effect_params)
                elif effect_type == 'delay':
                    processed_audio = self._apply_delay(processed_audio, effect_params)
                elif effect_type == 'distortion':
                    processed_audio = self._apply_distortion(processed_audio, effect_params)
                elif effect_type == 'lowpass':
                    processed_audio = self._apply_lowpass(processed_audio, effect_params)
                elif effect_type == 'stereo_width':
                    processed_audio = self._apply_stereo_width(processed_audio, effect_params)
                elif effect_type == 'limiter':
                    processed_audio = self._apply_limiter(processed_audio, effect_params)
            
            return processed_audio
            
        except Exception as e:
            print(f"Effect chain application failed: {e}")
            return audio
    
    def enhance_music(self, 
                     audio: np.ndarray,
                     enhancement_type: str = "full",
                     noise_reduction: bool = True,
                     dynamic_range: bool = True,
                     stereo_imaging: bool = True,
                     harmonic_enhancement: bool = True) -> np.ndarray:
        """
        Apply comprehensive music enhancement
        
        Args:
            audio: Input audio waveform
            enhancement_type: Type of enhancement ('light', 'medium', 'full', 'custom')
            noise_reduction: Whether to apply noise reduction
            dynamic_range: Whether to enhance dynamic range
            stereo_imaging: Whether to enhance stereo imaging
            harmonic_enhancement: Whether to enhance harmonics
        
        Returns:
            np.ndarray: Enhanced audio
        """
        try:
            if len(audio) == 0:
                return audio
            
            enhanced_audio = audio.copy()
            
            # Apply enhancement based on type
            if enhancement_type == "light":
                enhanced_audio = self._light_enhancement(enhanced_audio)
            elif enhancement_type == "medium":
                enhanced_audio = self._medium_enhancement(enhanced_audio)
            elif enhancement_type == "full":
                enhanced_audio = self._full_enhancement(
                    enhanced_audio, noise_reduction, dynamic_range, 
                    stereo_imaging, harmonic_enhancement
                )
            elif enhancement_type == "custom":
                enhanced_audio = self._custom_enhancement(
                    enhanced_audio, noise_reduction, dynamic_range, 
                    stereo_imaging, harmonic_enhancement
                )
            
            return enhanced_audio
            
        except Exception as e:
            print(f"Music enhancement failed: {e}")
            return audio
    
    def _light_enhancement(self, audio: np.ndarray) -> np.ndarray:
        """Apply light enhancement"""
        # Basic normalization
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio)) * 0.8
        
        # Light compression
        audio = self._apply_compression(audio, {'ratio': 2.0, 'threshold': -6})
        
        return audio
    
    def _medium_enhancement(self, audio: np.ndarray) -> np.ndarray:
        """Apply medium enhancement"""
        # Compression
        audio = self._apply_compression(audio, {'ratio': 3.0, 'threshold': -8})
        
        # EQ
        audio = self._apply_eq(audio, {
            'low_shelf': {'freq': 80, 'gain': 1.5},
            'high_shelf': {'freq': 12000, 'gain': 1.5}
        })
        
        # Stereo width
        audio = self._apply_stereo_width(audio, 1.1)
        
        return audio
    
    def _full_enhancement(self, audio: np.ndarray, noise_reduction: bool, 
                         dynamic_range: bool, stereo_imaging: bool, 
                         harmonic_enhancement: bool) -> np.ndarray:
        """Apply full enhancement"""
        if noise_reduction:
            audio = self._reduce_noise(audio)
        
        if dynamic_range:
            audio = self._enhance_dynamic_range(audio)
        
        if stereo_imaging:
            audio = self._enhance_stereo_imaging(audio)
        
        if harmonic_enhancement:
            audio = self._enhance_harmonics(audio)
        
        # Final mastering
        audio = self._apply_compression(audio, {'ratio': 4.0, 'threshold': -12})
        audio = self._apply_limiter(audio, {'threshold': -1.0})
        
        return audio
    
    def _custom_enhancement(self, audio: np.ndarray, noise_reduction: bool,
                           dynamic_range: bool, stereo_imaging: bool,
                           harmonic_enhancement: bool) -> np.ndarray:
        """Apply custom enhancement based on parameters"""
        if noise_reduction:
            audio = self._reduce_noise(audio)
        
        if dynamic_range:
            audio = self._enhance_dynamic_range(audio)
        
        if stereo_imaging:
            audio = self._enhance_stereo_imaging(audio)
        
        if harmonic_enhancement:
            audio = self._enhance_harmonics(audio)
        
        return audio
    
    # Individual effect methods
    def _apply_compression(self, audio: np.ndarray, params: Dict) -> np.ndarray:
        """Apply compression to audio"""
        try:
            ratio = params.get('ratio', 3.0)
            threshold = params.get('threshold', -8)
            attack = params.get('attack', 0.01)
            release = params.get('release', 0.1)
            
            # Convert threshold from dB to linear
            threshold_linear = 10 ** (threshold / 20)
            
            compressed = np.copy(audio)
            
            # Simple compression algorithm
            for i in range(len(audio)):
                if np.abs(audio[i]) > threshold_linear:
                    # Calculate compression ratio
                    excess = np.abs(audio[i]) - threshold_linear
                    compressed_excess = excess / ratio
                    new_level = threshold_linear + compressed_excess
                    
                    # Apply with attack/release envelope
                    compressed[i] = np.sign(audio[i]) * new_level
            
            return compressed
            
        except Exception as e:
            print(f"Compression failed: {e}")
            return audio
    
    def _apply_eq(self, audio: np.ndarray, params: Dict) -> np.ndarray:
        """Apply equalization to audio"""
        try:
            if not SCIPY_AVAILABLE:
                return audio
            
            eq_audio = audio.copy()
            
            # Apply low shelf
            if 'low_shelf' in params:
                low_params = params['low_shelf']
                freq = low_params['freq']
                gain = low_params['gain']
                
                # Design low shelf filter
                nyquist = self.sample_rate / 2
                normalized_freq = freq / nyquist
                b, a = signal.butter(1, normalized_freq, btype='low')
                
                # Apply gain
                filtered = signal.filtfilt(b, a, eq_audio)
                eq_audio = eq_audio + (filtered * (gain - 1))
            
            # Apply high shelf
            if 'high_shelf' in params:
                high_params = params['high_shelf']
                freq = high_params['freq']
                gain = high_params['gain']
                
                # Design high shelf filter
                nyquist = self.sample_rate / 2
                normalized_freq = freq / nyquist
                b, a = signal.butter(1, normalized_freq, btype='high')
                
                # Apply gain
                filtered = signal.filtfilt(b, a, eq_audio)
                eq_audio = eq_audio + (filtered * (gain - 1))
            
            return eq_audio
            
        except Exception as e:
            print(f"EQ application failed: {e}")
            return audio
    
    def _apply_reverb(self, audio: np.ndarray, params: Dict) -> np.ndarray:
        """Apply reverb effect"""
        try:
            if not SCIPY_AVAILABLE:
                return audio
            
            room_size = params.get('room_size', 0.5)
            damping = params.get('damping', 0.5)
            
            # Generate impulse response
            length = int(room_size * self.sample_rate)
            impulse = np.random.normal(0, 0.1, length)
            
            # Apply damping
            decay = np.exp(-np.arange(length) / (length * (1 - damping)))
            impulse *= decay
            
            # Apply convolution
            reverb_audio = signal.convolve(audio, impulse, mode='same')
            
            # Mix with original
            wet = params.get('wet', 0.3)
            return (1 - wet) * audio + wet * reverb_audio
            
        except Exception as e:
            print(f"Reverb application failed: {e}")
            return audio
    
    def _apply_delay(self, audio: np.ndarray, params: Dict) -> np.ndarray:
        """Apply delay effect"""
        try:
            delay_time = params.get('time', 0.25)  # seconds
            feedback = params.get('feedback', 0.4)
            wet = params.get('wet', 0.3)
            
            delay_samples = int(delay_time * self.sample_rate)
            delayed_audio = np.zeros_like(audio)
            
            # Apply delay
            for i in range(delay_samples, len(audio)):
                delayed_audio[i] = audio[i] + feedback * delayed_audio[i - delay_samples]
            
            # Mix with original
            return (1 - wet) * audio + wet * delayed_audio
            
        except Exception as e:
            print(f"Delay application failed: {e}")
            return audio
    
    def _apply_distortion(self, audio: np.ndarray, params: Dict) -> np.ndarray:
        """Apply distortion effect"""
        try:
            drive = params.get('drive', 0.5)
            tone = params.get('tone', 0.5)
            level = params.get('level', 0.8)
            
            # Apply drive
            distorted = audio * (1 + drive)
            
            # Apply soft clipping
            distorted = np.tanh(distorted)
            
            # Apply tone control (simple high-pass filter)
            if SCIPY_AVAILABLE:
                nyquist = self.sample_rate / 2
                cutoff = 1000 * (1 - tone) / nyquist
                b, a = signal.butter(1, cutoff, btype='high')
                distorted = signal.filtfilt(b, a, distorted)
            
            # Apply level
            distorted *= level
            
            return distorted
            
        except Exception as e:
            print(f"Distortion application failed: {e}")
            return audio
    
    def _apply_lowpass(self, audio: np.ndarray, params: Dict) -> np.ndarray:
        """Apply low-pass filter"""
        try:
            if not SCIPY_AVAILABLE:
                return audio
            
            cutoff = params.get('cutoff', 0.5)
            resonance = params.get('resonance', 0.5)
            
            # Design low-pass filter
            nyquist = self.sample_rate / 2
            normalized_cutoff = cutoff * nyquist / nyquist
            b, a = signal.butter(2, normalized_cutoff, btype='low')
            
            # Apply filter
            filtered = signal.filtfilt(b, a, audio)
            
            return filtered
            
        except Exception as e:
            print(f"Low-pass filter failed: {e}")
            return audio
    
    def _apply_stereo_width(self, audio: np.ndarray, width: float) -> np.ndarray:
        """Apply stereo width enhancement"""
        try:
            if audio.ndim == 1:
                # Convert mono to stereo
                stereo_audio = np.column_stack([audio, audio])
            else:
                stereo_audio = audio.copy()
            
            if stereo_audio.shape[1] != 2:
                return audio
            
            # Apply stereo width
            left = stereo_audio[:, 0]
            right = stereo_audio[:, 1]
            
            # Mid-side processing
            mid = (left + right) / 2
            side = (left - right) / 2
            
            # Apply width
            side *= width
            
            # Convert back to left-right
            new_left = mid + side
            new_right = mid - side
            
            return np.column_stack([new_left, new_right])
            
        except Exception as e:
            print(f"Stereo width application failed: {e}")
            return audio
    
    def _apply_limiter(self, audio: np.ndarray, params: Dict) -> np.ndarray:
        """Apply limiter to audio"""
        try:
            threshold = params.get('threshold', -1.0)
            release = params.get('release', 0.01)
            
            # Convert threshold from dB to linear
            threshold_linear = 10 ** (threshold / 20)
            
            limited = np.copy(audio)
            
            # Simple limiter
            for i in range(len(audio)):
                if np.abs(audio[i]) > threshold_linear:
                    limited[i] = np.sign(audio[i]) * threshold_linear
            
            return limited
            
        except Exception as e:
            print(f"Limiter application failed: {e}")
            return audio
    
    def _normalize_loudness(self, audio: np.ndarray, target_lufs: float) -> np.ndarray:
        """Normalize audio to target LUFS"""
        try:
            # Simple loudness normalization (in practice, you'd use proper LUFS measurement)
            current_rms = np.sqrt(np.mean(audio**2))
            target_rms = 10 ** (target_lufs / 20)
            
            if current_rms > 0:
                gain = target_rms / current_rms
                return audio * gain
            else:
                return audio
                
        except Exception as e:
            print(f"Loudness normalization failed: {e}")
            return audio
    
    def _reduce_noise(self, audio: np.ndarray) -> np.ndarray:
        """Reduce noise in audio"""
        try:
            if NOISE_REDUCE_AVAILABLE:
                return nr.reduce_noise(y=audio, sr=self.sample_rate, stationary=True)
            else:
                # Simple noise reduction using spectral subtraction
                return self._spectral_subtraction(audio)
                
        except Exception as e:
            print(f"Noise reduction failed: {e}")
            return audio
    
    def _spectral_subtraction(self, audio: np.ndarray) -> np.ndarray:
        """Simple spectral subtraction for noise reduction"""
        try:
            # Apply FFT
            fft = np.fft.fft(audio)
            magnitude = np.abs(fft)
            phase = np.angle(fft)
            
            # Estimate noise floor
            noise_floor = np.percentile(magnitude, 10)
            
            # Apply spectral subtraction
            enhanced_magnitude = magnitude - noise_floor
            enhanced_magnitude = np.maximum(enhanced_magnitude, 0.1 * magnitude)
            
            # Reconstruct signal
            enhanced_fft = enhanced_magnitude * np.exp(1j * phase)
            enhanced_audio = np.real(np.fft.ifft(enhanced_fft))
            
            return enhanced_audio
            
        except Exception as e:
            print(f"Spectral subtraction failed: {e}")
            return audio
    
    def _enhance_dynamic_range(self, audio: np.ndarray) -> np.ndarray:
        """Enhance dynamic range"""
        try:
            # Apply multiband compression
            enhanced = audio.copy()
            
            # Split into frequency bands
            if SCIPY_AVAILABLE:
                # Low frequencies
                b_low, a_low = signal.butter(2, 0.3, btype='low')
                low_band = signal.filtfilt(b_low, a_low, enhanced)
                
                # High frequencies
                b_high, a_high = signal.butter(2, 0.7, btype='high')
                high_band = signal.filtfilt(b_high, a_high, enhanced)
                
                # Apply different compression to each band
                low_compressed = self._apply_compression(low_band, {'ratio': 2.0, 'threshold': -6})
                high_compressed = self._apply_compression(high_band, {'ratio': 3.0, 'threshold': -8})
                
                # Recombine
                enhanced = low_compressed + high_compressed
            
            return enhanced
            
        except Exception as e:
            print(f"Dynamic range enhancement failed: {e}")
            return audio
    
    def _enhance_stereo_imaging(self, audio: np.ndarray) -> np.ndarray:
        """Enhance stereo imaging"""
        try:
            if audio.ndim == 1:
                return audio
            
            if audio.shape[1] != 2:
                return audio
            
            # Apply stereo enhancement
            enhanced = self._apply_stereo_width(audio, 1.2)
            
            # Apply mid-side EQ
            left = enhanced[:, 0]
            right = enhanced[:, 1]
            
            mid = (left + right) / 2
            side = (left - right) / 2
            
            # Enhance side signal
            if SCIPY_AVAILABLE:
                b, a = signal.butter(2, 0.5, btype='high')
                side_enhanced = signal.filtfilt(b, a, side) * 1.5
            else:
                side_enhanced = side * 1.5
            
            # Recombine
            new_left = mid + side_enhanced
            new_right = mid - side_enhanced
            
            return np.column_stack([new_left, new_right])
            
        except Exception as e:
            print(f"Stereo imaging enhancement failed: {e}")
            return audio
    
    def _enhance_harmonics(self, audio: np.ndarray) -> np.ndarray:
        """Enhance harmonics"""
        try:
            # Apply harmonic enhancement using saturation
            enhanced = np.tanh(audio * 1.2) * 0.8
            
            # Add subtle harmonic distortion
            harmonic = np.sin(audio * 2 * np.pi) * 0.1
            enhanced += harmonic
            
            return enhanced
            
        except Exception as e:
            print(f"Harmonic enhancement failed: {e}")
            return audio
    
    def analyze_music_quality(self, audio: np.ndarray) -> Dict:
        """Analyze music quality metrics"""
        try:
            if len(audio) == 0:
                return {"error": "Empty audio"}
            
            # Calculate various quality metrics
            rms = np.sqrt(np.mean(audio**2))
            peak = np.max(np.abs(audio))
            snr = 20 * np.log10(peak / (rms + 1e-10))
            
            # Spectral analysis
            if len(audio) > 1024:
                stft = librosa.stft(audio)
                magnitude = np.abs(stft)
                spectral_centroid = np.mean(librosa.feature.spectral_centroid(S=magnitude))
                spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(S=magnitude))
                zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(audio))
            else:
                spectral_centroid = 0
                spectral_rolloff = 0
                zero_crossing_rate = 0
            
            # Dynamic range
            dynamic_range = 20 * np.log10(peak / (rms + 1e-10))
            
            # Stereo analysis
            if audio.ndim == 2 and audio.shape[1] == 2:
                left = audio[:, 0]
                right = audio[:, 1]
                stereo_width = np.mean(np.abs(left - right)) / np.mean(np.abs(left + right))
            else:
                stereo_width = 0
            
            quality_metrics = {
                "rms_level": float(rms),
                "peak_level": float(peak),
                "snr_db": float(snr),
                "spectral_centroid": float(spectral_centroid),
                "spectral_rolloff": float(spectral_rolloff),
                "zero_crossing_rate": float(zero_crossing_rate),
                "dynamic_range_db": float(dynamic_range),
                "stereo_width": float(stereo_width),
                "duration_seconds": len(audio) / self.sample_rate,
                "sample_rate": self.sample_rate,
                "channels": 1 if audio.ndim == 1 else audio.shape[1]
            }
            
            return quality_metrics
            
        except Exception as e:
            print(f"Music quality analysis failed: {e}")
            return {"error": str(e)}
    
    def get_available_mastering_presets(self) -> List[str]:
        """Get list of available mastering presets"""
        return list(self.mastering_presets.keys())
    
    def get_available_effect_chains(self) -> List[str]:
        """Get list of available effect chains"""
        return list(self.effect_chains.keys())
    
    def create_custom_mastering_preset(self, name: str, parameters: Dict) -> bool:
        """Create a custom mastering preset"""
        try:
            self.mastering_presets[name] = parameters
            return True
        except Exception as e:
            print(f"Failed to create custom mastering preset: {e}")
            return False
    
    def create_custom_effect_chain(self, name: str, chain: List[Dict]) -> bool:
        """Create a custom effect chain"""
        try:
            self.effect_chains[name] = chain
            return True
        except Exception as e:
            print(f"Failed to create custom effect chain: {e}")
            return False


# Example usage
if __name__ == "__main__":
    # Initialize music processor
    processor = MusicProcessor()
    
    # Test with sample audio
    sample_audio = np.random.normal(0, 0.1, 44100)  # 1 second of noise
    
    # Test mastering
    mastered_audio = processor.master_music(sample_audio, preset="streaming")
    print(f"Mastered audio: {len(mastered_audio)} samples")
    
    # Test effect chain
    processed_audio = processor.apply_effect_chain(sample_audio, style="electronic")
    print(f"Processed audio: {len(processed_audio)} samples")
    
    # Test enhancement
    enhanced_audio = processor.enhance_music(sample_audio, enhancement_type="full")
    print(f"Enhanced audio: {len(enhanced_audio)} samples")
    
    # Test quality analysis
    quality_metrics = processor.analyze_music_quality(sample_audio)
    print(f"Quality metrics: {quality_metrics}")
    
    print("Music processing module initialized successfully!")
