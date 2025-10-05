"""
Audio Processing and Enhancement Utilities for Hydax AI TTS Engine
"""

import numpy as np
import librosa
import soundfile as sf
import scipy.signal
from typing import Optional, Dict, List, Tuple, Union
import warnings
warnings.filterwarnings("ignore")

try:
    import noisereduce as nr
    NOISE_REDUCE_AVAILABLE = True
except ImportError:
    NOISE_REDUCE_AVAILABLE = False
    print("Warning: noisereduce not available. Install with: pip install noisereduce")

try:
    from pydub import AudioSegment
    from pydub.effects import normalize, compress_dynamic_range
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    print("Warning: pydub not available. Install with: pip install pydub")

class AudioProcessor:
    """
    Advanced audio processing and enhancement utilities
    """
    
    def __init__(self, sample_rate: int = 22050):
        """
        Initialize the audio processor
        
        Args:
            sample_rate: Target sample rate for processing
        """
        self.sample_rate = sample_rate
        self.noise_profile = None
    
    def enhance_audio(self, 
                     audio: np.ndarray, 
                     enhancement_type: str = "full",
                     noise_reduction: bool = True,
                     normalize_audio: bool = True,
                     remove_silence: bool = True,
                     enhance_clarity: bool = True) -> np.ndarray:
        """
        Apply comprehensive audio enhancement
        
        Args:
            audio: Input audio waveform
            enhancement_type: Type of enhancement ('light', 'medium', 'full', 'custom')
            noise_reduction: Whether to apply noise reduction
            normalize_audio: Whether to normalize audio levels
            remove_silence: Whether to remove silence
            enhance_clarity: Whether to enhance clarity
        
        Returns:
            np.ndarray: Enhanced audio waveform
        """
        if len(audio) == 0:
            return audio
        
        enhanced_audio = audio.copy()
        
        # Apply enhancement based on type
        if enhancement_type == "light":
            enhanced_audio = self._light_enhancement(enhanced_audio)
        elif enhancement_type == "medium":
            enhanced_audio = self._medium_enhancement(enhanced_audio)
        elif enhancement_type == "full":
            enhanced_audio = self._full_enhancement(enhanced_audio, noise_reduction, normalize_audio, remove_silence, enhance_clarity)
        elif enhancement_type == "custom":
            enhanced_audio = self._custom_enhancement(enhanced_audio, noise_reduction, normalize_audio, remove_silence, enhance_clarity)
        
        return enhanced_audio
    
    def _light_enhancement(self, audio: np.ndarray) -> np.ndarray:
        """Apply light enhancement"""
        # Basic normalization
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio)) * 0.8
        
        # Light noise reduction
        if NOISE_REDUCE_AVAILABLE:
            audio = nr.reduce_noise(y=audio, sr=self.sample_rate, stationary=True, prop_decrease=0.1)
        
        return audio
    
    def _medium_enhancement(self, audio: np.ndarray) -> np.ndarray:
        """Apply medium enhancement"""
        # Remove silence
        audio = self.remove_silence(audio, threshold=0.01)
        
        # Noise reduction
        if NOISE_REDUCE_AVAILABLE:
            audio = nr.reduce_noise(y=audio, sr=self.sample_rate, stationary=True, prop_decrease=0.3)
        
        # Normalize
        audio = self.normalize_audio(audio)
        
        # Light clarity enhancement
        audio = self.enhance_clarity(audio, strength=0.3)
        
        return audio
    
    def _full_enhancement(self, audio: np.ndarray, noise_reduction: bool, normalize_audio: bool, 
                         remove_silence: bool, enhance_clarity: bool) -> np.ndarray:
        """Apply full enhancement"""
        if remove_silence:
            audio = self.remove_silence(audio, threshold=0.005)
        
        if noise_reduction:
            audio = self.reduce_noise(audio)
        
        if enhance_clarity:
            audio = self.enhance_clarity(audio, strength=0.5)
        
        if normalize_audio:
            audio = self.normalize_audio(audio)
        
        # Apply dynamic range compression
        audio = self.compress_dynamic_range(audio)
        
        # Apply gentle EQ
        audio = self.apply_eq(audio, eq_type="speech")
        
        return audio
    
    def _custom_enhancement(self, audio: np.ndarray, noise_reduction: bool, normalize_audio: bool,
                           remove_silence: bool, enhance_clarity: bool) -> np.ndarray:
        """Apply custom enhancement based on parameters"""
        if remove_silence:
            audio = self.remove_silence(audio)
        
        if noise_reduction:
            audio = self.reduce_noise(audio)
        
        if enhance_clarity:
            audio = self.enhance_clarity(audio)
        
        if normalize_audio:
            audio = self.normalize_audio(audio)
        
        return audio
    
    def reduce_noise(self, audio: np.ndarray, noise_reduction_strength: float = 0.5) -> np.ndarray:
        """
        Reduce noise in audio
        
        Args:
            audio: Input audio waveform
            noise_reduction_strength: Strength of noise reduction (0.0 to 1.0)
        
        Returns:
            np.ndarray: Noise-reduced audio
        """
        if not NOISE_REDUCE_AVAILABLE:
            print("Noise reduction not available. Install noisereduce package.")
            return audio
        
        try:
            # Apply noise reduction
            reduced_audio = nr.reduce_noise(
                y=audio, 
                sr=self.sample_rate, 
                stationary=True, 
                prop_decrease=noise_reduction_strength
            )
            
            return reduced_audio
            
        except Exception as e:
            print(f"Noise reduction failed: {e}")
            return audio
    
    def remove_silence(self, audio: np.ndarray, threshold: float = 0.01, min_silence_duration: float = 0.1) -> np.ndarray:
        """
        Remove silence from audio
        
        Args:
            audio: Input audio waveform
            threshold: Silence threshold
            min_silence_duration: Minimum duration of silence to remove (seconds)
        
        Returns:
            np.ndarray: Audio with silence removed
        """
        try:
            # Find non-silent regions
            frame_length = int(0.025 * self.sample_rate)  # 25ms frames
            hop_length = int(0.01 * self.sample_rate)     # 10ms hop
            
            # Calculate RMS energy
            rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
            
            # Find frames above threshold
            non_silent_frames = rms > threshold
            
            # Convert frame indices to sample indices
            non_silent_samples = []
            for i, is_non_silent in enumerate(non_silent_frames):
                if is_non_silent:
                    start_sample = i * hop_length
                    end_sample = min(start_sample + frame_length, len(audio))
                    non_silent_samples.extend(range(start_sample, end_sample))
            
            if not non_silent_samples:
                return np.array([])
            
            # Extract non-silent audio
            non_silent_audio = audio[non_silent_samples]
            
            return non_silent_audio
            
        except Exception as e:
            print(f"Silence removal failed: {e}")
            return audio
    
    def normalize_audio(self, audio: np.ndarray, target_level: float = 0.8) -> np.ndarray:
        """
        Normalize audio levels
        
        Args:
            audio: Input audio waveform
            target_level: Target peak level (0.0 to 1.0)
        
        Returns:
            np.ndarray: Normalized audio
        """
        try:
            if len(audio) == 0:
                return audio
            
            # Find peak level
            peak_level = np.max(np.abs(audio))
            
            if peak_level > 0:
                # Normalize to target level
                normalized_audio = audio / peak_level * target_level
                return normalized_audio
            else:
                return audio
                
        except Exception as e:
            print(f"Audio normalization failed: {e}")
            return audio
    
    def enhance_clarity(self, audio: np.ndarray, strength: float = 0.5) -> np.ndarray:
        """
        Enhance speech clarity
        
        Args:
            audio: Input audio waveform
            strength: Enhancement strength (0.0 to 1.0)
        
        Returns:
            np.ndarray: Clarity-enhanced audio
        """
        try:
            # Apply high-pass filter to remove low-frequency noise
            nyquist = self.sample_rate / 2
            cutoff = 80  # Hz
            normalized_cutoff = cutoff / nyquist
            
            # Design high-pass filter
            b, a = scipy.signal.butter(4, normalized_cutoff, btype='high')
            filtered_audio = scipy.signal.filtfilt(b, a, audio)
            
            # Apply gentle compression to enhance speech
            compressed_audio = self._apply_gentle_compression(filtered_audio, strength)
            
            return compressed_audio
            
        except Exception as e:
            print(f"Clarity enhancement failed: {e}")
            return audio
    
    def _apply_gentle_compression(self, audio: np.ndarray, strength: float) -> np.ndarray:
        """Apply gentle dynamic range compression"""
        try:
            # Simple compression algorithm
            threshold = 0.3
            ratio = 1.0 + strength * 2.0  # Compression ratio
            
            compressed_audio = np.copy(audio)
            
            # Apply compression to samples above threshold
            above_threshold = np.abs(audio) > threshold
            compressed_audio[above_threshold] = np.sign(audio[above_threshold]) * (
                threshold + (np.abs(audio[above_threshold]) - threshold) / ratio
            )
            
            return compressed_audio
            
        except Exception as e:
            print(f"Compression failed: {e}")
            return audio
    
    def compress_dynamic_range(self, audio: np.ndarray, ratio: float = 3.0, threshold: float = 0.3) -> np.ndarray:
        """
        Apply dynamic range compression
        
        Args:
            audio: Input audio waveform
            ratio: Compression ratio
            threshold: Compression threshold
        
        Returns:
            np.ndarray: Compressed audio
        """
        try:
            if PYDUB_AVAILABLE:
                # Convert to pydub format for compression
                audio_int16 = (audio * 32767).astype(np.int16)
                audio_segment = AudioSegment(
                    audio_int16.tobytes(),
                    frame_rate=self.sample_rate,
                    sample_width=2,
                    channels=1
                )
                
                # Apply compression
                compressed_segment = compress_dynamic_range(audio_segment)
                
                # Convert back to numpy
                compressed_audio = np.array(compressed_segment.get_array_of_samples(), dtype=np.float32) / 32767.0
                
                return compressed_audio
            else:
                # Fallback: simple compression
                return self._apply_gentle_compression(audio, 0.5)
                
        except Exception as e:
            print(f"Dynamic range compression failed: {e}")
            return audio
    
    def apply_eq(self, audio: np.ndarray, eq_type: str = "speech") -> np.ndarray:
        """
        Apply equalization
        
        Args:
            audio: Input audio waveform
            eq_type: Type of EQ ('speech', 'music', 'bright', 'warm')
        
        Returns:
            np.ndarray: EQ'd audio
        """
        try:
            if eq_type == "speech":
                # Speech-optimized EQ: boost mid frequencies, reduce low frequencies
                return self._apply_speech_eq(audio)
            elif eq_type == "music":
                # Music-optimized EQ: balanced frequency response
                return self._apply_music_eq(audio)
            elif eq_type == "bright":
                # Bright EQ: boost high frequencies
                return self._apply_bright_eq(audio)
            elif eq_type == "warm":
                # Warm EQ: boost low-mid frequencies
                return self._apply_warm_eq(audio)
            else:
                return audio
                
        except Exception as e:
            print(f"EQ application failed: {e}")
            return audio
    
    def _apply_speech_eq(self, audio: np.ndarray) -> np.ndarray:
        """Apply speech-optimized EQ"""
        try:
            # Design speech EQ filter
            # Boost around 1-4 kHz (speech intelligibility range)
            # Reduce below 200 Hz (rumble and noise)
            
            # High-pass filter for low frequencies
            nyquist = self.sample_rate / 2
            high_cutoff = 200 / nyquist
            b_high, a_high = scipy.signal.butter(2, high_cutoff, btype='high')
            
            # Band-pass filter for speech range
            low_cutoff = 1000 / nyquist
            high_cutoff = 4000 / nyquist
            b_band, a_band = scipy.signal.butter(2, [low_cutoff, high_cutoff], btype='band')
            
            # Apply filters
            filtered_audio = scipy.signal.filtfilt(b_high, a_high, audio)
            filtered_audio = scipy.signal.filtfilt(b_band, a_band, filtered_audio)
            
            # Mix with original (50% filtered, 50% original)
            eq_audio = 0.5 * filtered_audio + 0.5 * audio
            
            return eq_audio
            
        except Exception as e:
            print(f"Speech EQ failed: {e}")
            return audio
    
    def _apply_music_eq(self, audio: np.ndarray) -> np.ndarray:
        """Apply music-optimized EQ"""
        # Simple music EQ: slight boost in mid frequencies
        return self._apply_gentle_boost(audio, center_freq=2000, gain=2.0)
    
    def _apply_bright_eq(self, audio: np.ndarray) -> np.ndarray:
        """Apply bright EQ"""
        return self._apply_gentle_boost(audio, center_freq=8000, gain=3.0)
    
    def _apply_warm_eq(self, audio: np.ndarray) -> np.ndarray:
        """Apply warm EQ"""
        return self._apply_gentle_boost(audio, center_freq=500, gain=2.5)
    
    def _apply_gentle_boost(self, audio: np.ndarray, center_freq: float, gain: float) -> np.ndarray:
        """Apply gentle frequency boost"""
        try:
            nyquist = self.sample_rate / 2
            normalized_freq = center_freq / nyquist
            
            # Design peak filter
            Q = 1.0  # Quality factor
            b, a = scipy.signal.iirpeak(normalized_freq, Q, gain)
            
            # Apply filter
            boosted_audio = scipy.signal.filtfilt(b, a, audio)
            
            return boosted_audio
            
        except Exception as e:
            print(f"Frequency boost failed: {e}")
            return audio
    
    def add_reverb(self, audio: np.ndarray, reverb_type: str = "room", intensity: float = 0.3) -> np.ndarray:
        """
        Add reverb to audio
        
        Args:
            audio: Input audio waveform
            reverb_type: Type of reverb ('room', 'hall', 'plate', 'spring')
            intensity: Reverb intensity (0.0 to 1.0)
        
        Returns:
            np.ndarray: Audio with reverb
        """
        try:
            # Generate impulse response based on reverb type
            if reverb_type == "room":
                impulse_response = self._generate_room_impulse(intensity)
            elif reverb_type == "hall":
                impulse_response = self._generate_hall_impulse(intensity)
            elif reverb_type == "plate":
                impulse_response = self._generate_plate_impulse(intensity)
            elif reverb_type == "spring":
                impulse_response = self._generate_spring_impulse(intensity)
            else:
                return audio
            
            # Apply convolution
            reverb_audio = scipy.signal.convolve(audio, impulse_response, mode='same')
            
            # Mix with original
            mixed_audio = (1 - intensity) * audio + intensity * reverb_audio
            
            return mixed_audio
            
        except Exception as e:
            print(f"Reverb application failed: {e}")
            return audio
    
    def _generate_room_impulse(self, intensity: float) -> np.ndarray:
        """Generate room reverb impulse response"""
        length = int(0.5 * self.sample_rate * intensity)  # 0.5 seconds max
        impulse = np.random.normal(0, 0.1, length)
        
        # Apply exponential decay
        decay = np.exp(-np.arange(length) / (length * 0.3))
        impulse = impulse * decay
        
        return impulse
    
    def _generate_hall_impulse(self, intensity: float) -> np.ndarray:
        """Generate hall reverb impulse response"""
        length = int(2.0 * self.sample_rate * intensity)  # 2 seconds max
        impulse = np.random.normal(0, 0.05, length)
        
        # Apply slower decay
        decay = np.exp(-np.arange(length) / (length * 0.5))
        impulse = impulse * decay
        
        return impulse
    
    def _generate_plate_impulse(self, intensity: float) -> np.ndarray:
        """Generate plate reverb impulse response"""
        length = int(1.0 * self.sample_rate * intensity)  # 1 second max
        impulse = np.random.normal(0, 0.08, length)
        
        # Apply metallic decay
        decay = np.exp(-np.arange(length) / (length * 0.4))
        impulse = impulse * decay
        
        return impulse
    
    def _generate_spring_impulse(self, intensity: float) -> np.ndarray:
        """Generate spring reverb impulse response"""
        length = int(0.3 * self.sample_rate * intensity)  # 0.3 seconds max
        impulse = np.random.normal(0, 0.12, length)
        
        # Apply spring-like decay with oscillations
        t = np.arange(length) / self.sample_rate
        decay = np.exp(-t * 10) * np.sin(2 * np.pi * 50 * t)
        impulse = impulse * decay
        
        return impulse
    
    def analyze_audio_quality(self, audio: np.ndarray) -> Dict:
        """
        Analyze audio quality metrics
        
        Args:
            audio: Input audio waveform
        
        Returns:
            Dict: Audio quality metrics
        """
        try:
            if len(audio) == 0:
                return {"error": "Empty audio"}
            
            # Calculate various quality metrics
            rms = np.sqrt(np.mean(audio**2))
            peak = np.max(np.abs(audio))
            snr = 20 * np.log10(peak / (rms + 1e-10))
            
            # Spectral analysis
            stft = librosa.stft(audio)
            magnitude = np.abs(stft)
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(S=magnitude))
            spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(S=magnitude))
            
            # Zero crossing rate
            zcr = np.mean(librosa.feature.zero_crossing_rate(audio))
            
            # Dynamic range
            dynamic_range = 20 * np.log10(peak / (rms + 1e-10))
            
            quality_metrics = {
                "rms_level": float(rms),
                "peak_level": float(peak),
                "snr_db": float(snr),
                "spectral_centroid": float(spectral_centroid),
                "spectral_rolloff": float(spectral_rolloff),
                "zero_crossing_rate": float(zcr),
                "dynamic_range_db": float(dynamic_range),
                "duration_seconds": len(audio) / self.sample_rate,
                "sample_rate": self.sample_rate
            }
            
            return quality_metrics
            
        except Exception as e:
            print(f"Audio quality analysis failed: {e}")
            return {"error": str(e)}
    
    def convert_format(self, audio: np.ndarray, target_format: str = "wav") -> bytes:
        """
        Convert audio to different formats
        
        Args:
            audio: Input audio waveform
            target_format: Target format ('wav', 'mp3', 'flac')
        
        Returns:
            bytes: Audio data in target format
        """
        try:
            if target_format.lower() == "wav":
                # Convert to WAV bytes
                audio_int16 = (audio * 32767).astype(np.int16)
                return audio_int16.tobytes()
            
            elif target_format.lower() == "mp3" and PYDUB_AVAILABLE:
                # Convert to MP3 using pydub
                audio_int16 = (audio * 32767).astype(np.int16)
                audio_segment = AudioSegment(
                    audio_int16.tobytes(),
                    frame_rate=self.sample_rate,
                    sample_width=2,
                    channels=1
                )
                return audio_segment.export(format="mp3").read()
            
            else:
                print(f"Format {target_format} not supported or pydub not available")
                return b""
                
        except Exception as e:
            print(f"Format conversion failed: {e}")
            return b""


# Example usage
if __name__ == "__main__":
    # Initialize audio processor
    processor = AudioProcessor()
    
    # Test with sample audio
    sample_audio = np.random.normal(0, 0.1, 22050)  # 1 second of noise
    
    # Test different enhancement types
    enhancement_types = ["light", "medium", "full"]
    
    for enhancement_type in enhancement_types:
        enhanced_audio = processor.enhance_audio(sample_audio, enhancement_type=enhancement_type)
        print(f"Applied {enhancement_type} enhancement: {len(enhanced_audio)} samples")
    
    # Test audio quality analysis
    quality_metrics = processor.analyze_audio_quality(sample_audio)
    print(f"Audio quality metrics: {quality_metrics}")
    
    print("Audio processing module initialized successfully!")
