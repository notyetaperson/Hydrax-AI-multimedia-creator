"""
Hydax AI Music Generation Engine - A powerful AI-powered music generator
"""

import os
import torch
import numpy as np
import librosa
import soundfile as sf
from typing import Optional, Dict, List, Tuple, Union
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

try:
    import torchaudio
    TORCHAUDIO_AVAILABLE = True
except ImportError:
    TORCHAUDIO_AVAILABLE = False
    print("Warning: torchaudio not available. Install with: pip install torchaudio")

try:
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not available. Install with: pip install transformers")

try:
    import midi2audio
    import pretty_midi
    MIDI_AVAILABLE = True
except ImportError:
    MIDI_AVAILABLE = False
    print("Warning: MIDI libraries not available. Install with: pip install midi2audio pretty_midi")

class MusicGenerator:
    """
    Advanced AI-powered music generation engine
    """
    
    def __init__(self, 
                 device: str = "auto",
                 sample_rate: int = 44100,
                 duration: float = 120.0):  # 2 minutes default
        """
        Initialize the music generator
        
        Args:
            device: Device to run on ('cpu', 'cuda', or 'auto')
            sample_rate: Audio sample rate
            duration: Default duration in seconds
        """
        self.device = self._setup_device(device)
        self.sample_rate = sample_rate
        self.default_duration = duration
        self.models = {}
        self.style_presets = self._create_style_presets()
        self.genre_presets = self._create_genre_presets()
        
        # Initialize models
        self._initialize_models()
        
        # Music generation parameters
        self.tempo_range = (60, 180)
        self.key_signatures = ['C', 'G', 'D', 'A', 'E', 'B', 'F#', 'C#', 'F', 'Bb', 'Eb', 'Ab']
        self.time_signatures = ['4/4', '3/4', '2/4', '6/8', '12/8']
        
    def _setup_device(self, device: str) -> str:
        """Setup the device for inference"""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        return device
    
    def _create_style_presets(self) -> Dict:
        """Create predefined music style presets"""
        return {
            'ambient': {
                'tempo': (60, 80),
                'instruments': ['pad', 'strings', 'ambient'],
                'dynamics': 'soft',
                'harmony': 'diatonic',
                'rhythm': 'sparse',
                'effects': ['reverb', 'delay', 'chorus']
            },
            'electronic': {
                'tempo': (120, 140),
                'instruments': ['synth', 'bass', 'drums'],
                'dynamics': 'dynamic',
                'harmony': 'modal',
                'rhythm': 'complex',
                'effects': ['distortion', 'filter', 'sidechain']
            },
            'classical': {
                'tempo': (80, 120),
                'instruments': ['strings', 'piano', 'woodwinds'],
                'dynamics': 'expressive',
                'harmony': 'functional',
                'rhythm': 'structured',
                'effects': ['reverb', 'eq']
            },
            'jazz': {
                'tempo': (100, 160),
                'instruments': ['piano', 'bass', 'saxophone', 'drums'],
                'dynamics': 'swing',
                'harmony': 'extended',
                'rhythm': 'syncopated',
                'effects': ['reverb', 'compression']
            },
            'rock': {
                'tempo': (120, 160),
                'instruments': ['guitar', 'bass', 'drums', 'vocals'],
                'dynamics': 'aggressive',
                'harmony': 'power_chords',
                'rhythm': 'driving',
                'effects': ['distortion', 'reverb', 'compression']
            },
            'pop': {
                'tempo': (100, 130),
                'instruments': ['synth', 'bass', 'drums', 'vocals'],
                'dynamics': 'polished',
                'harmony': 'catchy',
                'rhythm': 'steady',
                'effects': ['compression', 'eq', 'reverb']
            },
            'cinematic': {
                'tempo': (60, 100),
                'instruments': ['orchestra', 'choir', 'percussion'],
                'dynamics': 'epic',
                'harmony': 'dramatic',
                'rhythm': 'varied',
                'effects': ['reverb', 'eq', 'compression']
            },
            'lo-fi': {
                'tempo': (70, 90),
                'instruments': ['piano', 'bass', 'drums', 'vinyl'],
                'dynamics': 'chill',
                'harmony': 'jazzy',
                'rhythm': 'relaxed',
                'effects': ['vinyl_crackle', 'lowpass', 'compression']
            }
        }
    
    def _create_genre_presets(self) -> Dict:
        """Create predefined genre presets"""
        return {
            'house': {
                'bpm': 128,
                'key': 'C',
                'time_signature': '4/4',
                'structure': 'intro-verse-chorus-verse-chorus-bridge-chorus-outro',
                'instruments': ['kick', 'hihat', 'bass', 'lead', 'pad']
            },
            'trance': {
                'bpm': 132,
                'key': 'A',
                'time_signature': '4/4',
                'structure': 'intro-build-drop-breakdown-build-drop-outro',
                'instruments': ['kick', 'bass', 'lead', 'pad', 'arpeggio']
            },
            'dubstep': {
                'bpm': 140,
                'key': 'F#',
                'time_signature': '4/4',
                'structure': 'intro-drop-breakdown-drop-outro',
                'instruments': ['kick', 'snare', 'bass', 'wobble', 'fx']
            },
            'ambient': {
                'bpm': 60,
                'key': 'D',
                'time_signature': '4/4',
                'structure': 'intro-development-climax-resolution-outro',
                'instruments': ['pad', 'strings', 'ambient', 'field_recordings']
            },
            'orchestral': {
                'bpm': 90,
                'key': 'C',
                'time_signature': '4/4',
                'structure': 'exposition-development-recapitulation',
                'instruments': ['strings', 'brass', 'woodwinds', 'percussion']
            }
        }
    
    def _initialize_models(self):
        """Initialize music generation models"""
        print("Initializing music generation models...")
        
        # Note: In a real implementation, you would load actual music generation models
        # For this example, we'll create placeholder models
        try:
            # Placeholder for music generation model
            self.models['musicgen'] = self._create_placeholder_model()
            print("✓ Music generation model initialized")
            
            # Placeholder for style transfer model
            self.models['style_transfer'] = self._create_placeholder_model()
            print("✓ Style transfer model initialized")
            
        except Exception as e:
            print(f"Model initialization failed: {e}")
            self.models = {}
    
    def _create_placeholder_model(self):
        """Create a placeholder model for demonstration"""
        # In a real implementation, this would load actual models like:
        # - MusicGen from Meta
        # - Jukebox from OpenAI
        # - MuseNet from OpenAI
        # - Custom trained models
        return {"type": "placeholder", "device": self.device}
    
    def generate_music(self, 
                      prompt: str,
                      duration: float = None,
                      style: str = "ambient",
                      genre: str = None,
                      tempo: int = None,
                      key: str = None,
                      time_signature: str = "4/4",
                      instruments: List[str] = None,
                      mood: str = "neutral",
                      complexity: str = "medium") -> np.ndarray:
        """
        Generate music from a text prompt
        
        Args:
            prompt: Text description of the music to generate
            duration: Duration in seconds (default: 2-5 minutes)
            style: Music style ('ambient', 'electronic', 'classical', etc.)
            genre: Specific genre ('house', 'trance', 'dubstep', etc.)
            tempo: BPM (beats per minute)
            key: Musical key
            time_signature: Time signature
            instruments: List of instruments to include
            mood: Mood of the music ('happy', 'sad', 'energetic', etc.)
            complexity: Complexity level ('simple', 'medium', 'complex')
        
        Returns:
            np.ndarray: Generated music waveform
        """
        try:
            if duration is None:
                duration = self.default_duration
            
            # Validate and set parameters
            params = self._prepare_generation_params(
                prompt, duration, style, genre, tempo, key, 
                time_signature, instruments, mood, complexity
            )
            
            print(f"Generating {duration:.1f}s of {style} music...")
            print(f"Prompt: {prompt}")
            
            # Generate music using the appropriate method
            if self.models.get('musicgen'):
                music = self._generate_with_ai_model(params)
            else:
                music = self._generate_with_algorithmic_method(params)
            
            # Post-process the generated music
            music = self._post_process_music(music, params)
            
            return music
            
        except Exception as e:
            print(f"Music generation failed: {e}")
            return np.array([])
    
    def _prepare_generation_params(self, prompt: str, duration: float, style: str,
                                 genre: str, tempo: int, key: str, time_signature: str,
                                 instruments: List[str], mood: str, complexity: str) -> Dict:
        """Prepare parameters for music generation"""
        params = {
            'prompt': prompt,
            'duration': duration,
            'style': style,
            'genre': genre,
            'tempo': tempo,
            'key': key,
            'time_signature': time_signature,
            'instruments': instruments,
            'mood': mood,
            'complexity': complexity
        }
        
        # Set defaults based on style
        if style in self.style_presets:
            style_preset = self.style_presets[style]
            if tempo is None:
                tempo = np.random.randint(style_preset['tempo'][0], style_preset['tempo'][1])
            if instruments is None:
                instruments = style_preset['instruments']
        
        # Set defaults based on genre
        if genre and genre in self.genre_presets:
            genre_preset = self.genre_presets[genre]
            if tempo is None:
                tempo = genre_preset['bpm']
            if key is None:
                key = genre_preset['key']
            if time_signature is None:
                time_signature = genre_preset['time_signature']
        
        # Set random defaults if not specified
        if tempo is None:
            tempo = np.random.randint(self.tempo_range[0], self.tempo_range[1])
        if key is None:
            key = np.random.choice(self.key_signatures)
        if instruments is None:
            instruments = ['piano', 'strings', 'bass', 'drums']
        
        params.update({
            'tempo': tempo,
            'key': key,
            'instruments': instruments
        })
        
        return params
    
    def _generate_with_ai_model(self, params: Dict) -> np.ndarray:
        """Generate music using AI model (placeholder implementation)"""
        try:
            # In a real implementation, this would use models like:
            # - MusicGen: model.generate(prompt, duration, style)
            # - Jukebox: model.sample(prompt, length, style)
            # - Custom models trained on music datasets
            
            duration = params['duration']
            tempo = params['tempo']
            instruments = params['instruments']
            style = params['style']
            
            # Generate placeholder music with basic structure
            music = self._generate_algorithmic_music(params)
            
            return music
            
        except Exception as e:
            print(f"AI model generation failed: {e}")
            return self._generate_with_algorithmic_method(params)
    
    def _generate_with_algorithmic_method(self, params: Dict) -> np.ndarray:
        """Generate music using algorithmic methods"""
        try:
            duration = params['duration']
            tempo = params['tempo']
            key = params['key']
            instruments = params['instruments']
            style = params['style']
            mood = params['mood']
            
            # Generate different layers of music
            music_layers = []
            
            # Generate rhythm section
            if 'drums' in instruments or 'percussion' in instruments:
                rhythm = self._generate_rhythm_layer(duration, tempo, style)
                music_layers.append(rhythm)
            
            # Generate bass line
            if 'bass' in instruments:
                bass = self._generate_bass_layer(duration, tempo, key, style)
                music_layers.append(bass)
            
            # Generate harmonic layer
            if any(inst in instruments for inst in ['piano', 'strings', 'pad', 'synth']):
                harmony = self._generate_harmony_layer(duration, tempo, key, style, mood)
                music_layers.append(harmony)
            
            # Generate melody layer
            if any(inst in instruments for inst in ['lead', 'melody', 'saxophone', 'guitar']):
                melody = self._generate_melody_layer(duration, tempo, key, style, mood)
                music_layers.append(melody)
            
            # Mix all layers
            if music_layers:
                music = self._mix_layers(music_layers)
            else:
                # Fallback: generate simple ambient music
                music = self._generate_ambient_music(duration, tempo, key)
            
            return music
            
        except Exception as e:
            print(f"Algorithmic generation failed: {e}")
            return self._generate_fallback_music(params['duration'])
    
    def _generate_rhythm_layer(self, duration: float, tempo: int, style: str) -> np.ndarray:
        """Generate rhythm/drum layer"""
        try:
            samples = int(duration * self.sample_rate)
            rhythm = np.zeros(samples)
            
            # Calculate beat timing
            beat_duration = 60.0 / tempo  # seconds per beat
            beat_samples = int(beat_duration * self.sample_rate)
            
            if style in ['electronic', 'house', 'trance', 'dubstep']:
                # Electronic drum pattern
                for i in range(0, samples, beat_samples):
                    if i < samples:
                        # Kick on every beat
                        kick = self._generate_kick_sound()
                        end_idx = min(i + len(kick), samples)
                        rhythm[i:end_idx] += kick[:end_idx-i] * 0.8
                        
                        # Hi-hat on off-beats
                        if i + beat_samples//2 < samples:
                            hihat = self._generate_hihat_sound()
                            start_idx = i + beat_samples//2
                            end_idx = min(start_idx + len(hihat), samples)
                            rhythm[start_idx:end_idx] += hihat[:end_idx-start_idx] * 0.3
            
            elif style in ['jazz', 'rock', 'pop']:
                # Acoustic drum pattern
                for i in range(0, samples, beat_samples):
                    if i < samples:
                        # Kick on beats 1 and 3
                        if (i // beat_samples) % 4 in [0, 2]:
                            kick = self._generate_acoustic_kick()
                            end_idx = min(i + len(kick), samples)
                            rhythm[i:end_idx] += kick[:end_idx-i] * 0.7
                        
                        # Snare on beats 2 and 4
                        if (i // beat_samples) % 4 in [1, 3]:
                            snare = self._generate_snare_sound()
                            end_idx = min(i + len(snare), samples)
                            rhythm[i:end_idx] += snare[:end_idx-i] * 0.6
            
            return rhythm
            
        except Exception as e:
            print(f"Rhythm generation failed: {e}")
            return np.zeros(int(duration * self.sample_rate))
    
    def _generate_bass_layer(self, duration: float, tempo: int, key: str, style: str) -> np.ndarray:
        """Generate bass line"""
        try:
            samples = int(duration * self.sample_rate)
            bass = np.zeros(samples)
            
            # Generate bass notes based on key and style
            if style in ['electronic', 'house', 'trance']:
                # Simple electronic bass pattern
                beat_duration = 60.0 / tempo
                beat_samples = int(beat_duration * self.sample_rate)
                
                for i in range(0, samples, beat_samples):
                    if i < samples:
                        note = self._generate_bass_note(key, style)
                        end_idx = min(i + len(note), samples)
                        bass[i:end_idx] += note[:end_idx-i] * 0.6
            
            elif style in ['jazz', 'classical']:
                # More complex bass line
                note_duration = beat_samples // 2
                for i in range(0, samples, note_duration):
                    if i < samples:
                        note = self._generate_bass_note(key, style)
                        end_idx = min(i + len(note), samples)
                        bass[i:end_idx] += note[:end_idx-i] * 0.5
            
            return bass
            
        except Exception as e:
            print(f"Bass generation failed: {e}")
            return np.zeros(int(duration * self.sample_rate))
    
    def _generate_harmony_layer(self, duration: float, tempo: int, key: str, style: str, mood: str) -> np.ndarray:
        """Generate harmonic layer (chords, pads, etc.)"""
        try:
            samples = int(duration * self.sample_rate)
            harmony = np.zeros(samples)
            
            # Generate chord progression
            chord_duration = 60.0 / tempo * 4  # 4 beats per chord
            chord_samples = int(chord_duration * self.sample_rate)
            
            chord_progression = self._get_chord_progression(key, mood)
            
            for i, chord in enumerate(chord_progression):
                start_idx = (i * chord_samples) % samples
                end_idx = min(start_idx + chord_samples, samples)
                
                if start_idx < samples:
                    chord_audio = self._generate_chord(chord, chord_duration, style)
                    actual_length = min(len(chord_audio), end_idx - start_idx)
                    harmony[start_idx:start_idx+actual_length] += chord_audio[:actual_length] * 0.4
            
            return harmony
            
        except Exception as e:
            print(f"Harmony generation failed: {e}")
            return np.zeros(int(duration * self.sample_rate))
    
    def _generate_melody_layer(self, duration: float, tempo: int, key: str, style: str, mood: str) -> np.ndarray:
        """Generate melody layer"""
        try:
            samples = int(duration * self.sample_rate)
            melody = np.zeros(samples)
            
            # Generate melody based on key and mood
            note_duration = 60.0 / tempo / 2  # Half notes
            note_samples = int(note_duration * self.sample_rate)
            
            melody_notes = self._generate_melody_sequence(key, mood, duration)
            
            for i, note in enumerate(melody_notes):
                start_idx = (i * note_samples) % samples
                end_idx = min(start_idx + note_samples, samples)
                
                if start_idx < samples:
                    note_audio = self._generate_melody_note(note, note_duration, style)
                    actual_length = min(len(note_audio), end_idx - start_idx)
                    melody[start_idx:start_idx+actual_length] += note_audio[:actual_length] * 0.3
            
            return melody
            
        except Exception as e:
            print(f"Melody generation failed: {e}")
            return np.zeros(int(duration * self.sample_rate))
    
    def _generate_ambient_music(self, duration: float, tempo: int, key: str) -> np.ndarray:
        """Generate simple ambient music as fallback"""
        try:
            samples = int(duration * self.sample_rate)
            
            # Generate ambient pad
            t = np.linspace(0, duration, samples)
            ambient = np.zeros(samples)
            
            # Add multiple sine waves for rich ambient sound
            frequencies = [220, 330, 440, 550]  # A3, E4, A4, C#5
            
            for freq in frequencies:
                wave = np.sin(2 * np.pi * freq * t) * 0.1
                ambient += wave
            
            # Add slow modulation
            modulation = np.sin(2 * np.pi * 0.1 * t) * 0.05
            ambient += modulation
            
            # Apply envelope
            envelope = np.exp(-t / (duration * 0.8))
            ambient *= envelope
            
            return ambient
            
        except Exception as e:
            print(f"Ambient generation failed: {e}")
            return self._generate_fallback_music(duration)
    
    def _generate_fallback_music(self, duration: float) -> np.ndarray:
        """Generate very simple fallback music"""
        try:
            samples = int(duration * self.sample_rate)
            t = np.linspace(0, duration, samples)
            
            # Simple sine wave
            frequency = 440  # A4
            music = np.sin(2 * np.pi * frequency * t) * 0.1
            
            # Add some variation
            music += np.sin(2 * np.pi * frequency * 1.5 * t) * 0.05
            
            return music
            
        except Exception as e:
            print(f"Fallback generation failed: {e}")
            return np.zeros(int(duration * self.sample_rate))
    
    # Sound generation methods
    def _generate_kick_sound(self) -> np.ndarray:
        """Generate electronic kick drum sound"""
        duration = 0.1  # 100ms
        samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, samples)
        
        # Kick drum: low frequency with quick decay
        frequency = 60  # Hz
        kick = np.sin(2 * np.pi * frequency * t)
        
        # Apply envelope
        envelope = np.exp(-t * 20)  # Quick decay
        kick *= envelope
        
        return kick
    
    def _generate_hihat_sound(self) -> np.ndarray:
        """Generate hi-hat sound"""
        duration = 0.05  # 50ms
        samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, samples)
        
        # Hi-hat: high frequency noise
        hihat = np.random.normal(0, 0.1, samples)
        
        # Apply high-pass filter effect
        hihat = np.diff(hihat, prepend=0)
        
        # Apply envelope
        envelope = np.exp(-t * 50)
        hihat *= envelope
        
        return hihat
    
    def _generate_snare_sound(self) -> np.ndarray:
        """Generate snare drum sound"""
        duration = 0.1  # 100ms
        samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, samples)
        
        # Snare: combination of tone and noise
        tone = np.sin(2 * np.pi * 200 * t) * 0.3
        noise = np.random.normal(0, 0.1, samples)
        
        snare = tone + noise
        
        # Apply envelope
        envelope = np.exp(-t * 15)
        snare *= envelope
        
        return snare
    
    def _generate_bass_note(self, key: str, style: str) -> np.ndarray:
        """Generate bass note"""
        duration = 0.5  # 500ms
        samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, samples)
        
        # Get frequency for key
        frequency = self._get_key_frequency(key, octave=2)  # Low octave
        
        # Generate bass note
        if style in ['electronic', 'house', 'trance']:
            # Sawtooth wave for electronic bass
            bass = np.sign(np.sin(2 * np.pi * frequency * t)) * 0.3
        else:
            # Sine wave for acoustic bass
            bass = np.sin(2 * np.pi * frequency * t) * 0.4
        
        # Apply envelope
        envelope = np.exp(-t * 2)
        bass *= envelope
        
        return bass
    
    def _generate_chord(self, chord: str, duration: float, style: str) -> np.ndarray:
        """Generate chord audio"""
        samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, samples)
        
        # Get chord frequencies
        frequencies = self._get_chord_frequencies(chord)
        
        chord_audio = np.zeros(samples)
        
        for freq in frequencies:
            if style in ['electronic', 'house', 'trance']:
                # Square wave for electronic style
                wave = np.sign(np.sin(2 * np.pi * freq * t)) * 0.1
            else:
                # Sine wave for acoustic style
                wave = np.sin(2 * np.pi * freq * t) * 0.1
            
            chord_audio += wave
        
        # Apply envelope
        envelope = np.exp(-t * 0.5)
        chord_audio *= envelope
        
        return chord_audio
    
    def _generate_melody_note(self, note: str, duration: float, style: str) -> np.ndarray:
        """Generate melody note"""
        samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, samples)
        
        # Get note frequency
        frequency = self._get_note_frequency(note)
        
        # Generate note
        if style in ['electronic', 'house', 'trance']:
            # Sawtooth wave
            note_audio = np.sign(np.sin(2 * np.pi * frequency * t)) * 0.2
        else:
            # Sine wave
            note_audio = np.sin(2 * np.pi * frequency * t) * 0.2
        
        # Apply envelope
        envelope = np.exp(-t * 3)
        note_audio *= envelope
        
        return note_audio
    
    # Music theory helper methods
    def _get_key_frequency(self, key: str, octave: int = 4) -> float:
        """Get frequency for a musical key"""
        # A4 = 440 Hz
        note_frequencies = {
            'C': 261.63, 'C#': 277.18, 'D': 293.66, 'D#': 311.13,
            'E': 329.63, 'F': 349.23, 'F#': 369.99, 'G': 392.00,
            'G#': 415.30, 'A': 440.00, 'A#': 466.16, 'B': 493.88
        }
        
        base_freq = note_frequencies.get(key, 440.0)
        return base_freq * (2 ** (octave - 4))
    
    def _get_chord_frequencies(self, chord: str) -> List[float]:
        """Get frequencies for a chord"""
        # Simple chord mapping
        chord_notes = {
            'C': ['C', 'E', 'G'],
            'D': ['D', 'F#', 'A'],
            'E': ['E', 'G#', 'B'],
            'F': ['F', 'A', 'C'],
            'G': ['G', 'B', 'D'],
            'A': ['A', 'C#', 'E'],
            'B': ['B', 'D#', 'F#']
        }
        
        notes = chord_notes.get(chord, ['C', 'E', 'G'])
        return [self._get_key_frequency(note, 4) for note in notes]
    
    def _get_note_frequency(self, note: str) -> float:
        """Get frequency for a note"""
        return self._get_key_frequency(note, 5)  # Higher octave for melody
    
    def _get_chord_progression(self, key: str, mood: str) -> List[str]:
        """Get chord progression based on key and mood"""
        # Simple chord progressions
        if mood in ['happy', 'energetic', 'uplifting']:
            progressions = {
                'C': ['C', 'G', 'Am', 'F'],
                'G': ['G', 'D', 'Em', 'C'],
                'D': ['D', 'A', 'Bm', 'G'],
                'A': ['A', 'E', 'F#m', 'D']
            }
        else:  # sad, melancholic, calm
            progressions = {
                'C': ['Am', 'F', 'C', 'G'],
                'G': ['Em', 'C', 'G', 'D'],
                'D': ['Bm', 'G', 'D', 'A'],
                'A': ['F#m', 'D', 'A', 'E']
            }
        
        return progressions.get(key, ['C', 'G', 'Am', 'F'])
    
    def _generate_melody_sequence(self, key: str, mood: str, duration: float) -> List[str]:
        """Generate melody note sequence"""
        # Simple melody generation
        scale_notes = {
            'C': ['C', 'D', 'E', 'F', 'G', 'A', 'B'],
            'G': ['G', 'A', 'B', 'C', 'D', 'E', 'F#'],
            'D': ['D', 'E', 'F#', 'G', 'A', 'B', 'C#'],
            'A': ['A', 'B', 'C#', 'D', 'E', 'F#', 'G#']
        }
        
        notes = scale_notes.get(key, ['C', 'D', 'E', 'F', 'G', 'A', 'B'])
        
        # Generate sequence based on duration
        num_notes = int(duration * 2)  # 2 notes per second
        melody = []
        
        for i in range(num_notes):
            if mood in ['happy', 'energetic']:
                # More ascending movement
                note = np.random.choice(notes[2:])  # Higher notes
            else:
                # More descending movement
                note = np.random.choice(notes[:5])  # Lower notes
            
            melody.append(note)
        
        return melody
    
    def _mix_layers(self, layers: List[np.ndarray]) -> np.ndarray:
        """Mix multiple audio layers"""
        if not layers:
            return np.array([])
        
        # Find the maximum length
        max_length = max(len(layer) for layer in layers)
        
        # Mix all layers
        mixed = np.zeros(max_length)
        for layer in layers:
            # Pad shorter layers with zeros
            padded_layer = np.pad(layer, (0, max_length - len(layer)))
            mixed += padded_layer
        
        # Normalize to prevent clipping
        if np.max(np.abs(mixed)) > 0:
            mixed = mixed / np.max(np.abs(mixed)) * 0.8
        
        return mixed
    
    def _post_process_music(self, music: np.ndarray, params: Dict) -> np.ndarray:
        """Post-process generated music"""
        try:
            if len(music) == 0:
                return music
            
            # Apply style-specific effects
            style = params.get('style', 'ambient')
            
            if style in ['electronic', 'house', 'trance', 'dubstep']:
                # Add compression and EQ for electronic music
                music = self._apply_compression(music)
                music = self._apply_eq(music, 'electronic')
            
            elif style in ['ambient', 'cinematic']:
                # Add reverb for ambient music
                music = self._apply_reverb(music, 'hall', 0.3)
            
            elif style in ['lo-fi']:
                # Add lo-fi effects
                music = self._apply_lofi_effects(music)
            
            # Normalize final output
            if np.max(np.abs(music)) > 0:
                music = music / np.max(np.abs(music)) * 0.8
            
            return music
            
        except Exception as e:
            print(f"Post-processing failed: {e}")
            return music
    
    def _apply_compression(self, audio: np.ndarray) -> np.ndarray:
        """Apply compression to audio"""
        # Simple compression
        threshold = 0.3
        ratio = 3.0
        
        compressed = np.copy(audio)
        above_threshold = np.abs(audio) > threshold
        
        compressed[above_threshold] = np.sign(audio[above_threshold]) * (
            threshold + (np.abs(audio[above_threshold]) - threshold) / ratio
        )
        
        return compressed
    
    def _apply_eq(self, audio: np.ndarray, eq_type: str) -> np.ndarray:
        """Apply equalization"""
        # Simple EQ using filtering
        if eq_type == 'electronic':
            # Boost high frequencies
            from scipy import signal
            b, a = signal.butter(2, 0.3, btype='high')
            high_boost = signal.filtfilt(b, a, audio) * 0.3
            return audio + high_boost
        else:
            return audio
    
    def _apply_reverb(self, audio: np.ndarray, reverb_type: str, intensity: float) -> np.ndarray:
        """Apply reverb effect"""
        # Simple reverb using convolution
        try:
            from scipy import signal
            
            # Generate impulse response
            if reverb_type == 'hall':
                length = int(0.5 * self.sample_rate * intensity)
                impulse = np.random.normal(0, 0.1, length)
                decay = np.exp(-np.arange(length) / (length * 0.3))
                impulse *= decay
            
            # Apply convolution
            reverb_audio = signal.convolve(audio, impulse, mode='same')
            
            # Mix with original
            return (1 - intensity) * audio + intensity * reverb_audio
            
        except Exception as e:
            print(f"Reverb application failed: {e}")
            return audio
    
    def _apply_lofi_effects(self, audio: np.ndarray) -> np.ndarray:
        """Apply lo-fi effects"""
        # Add vinyl crackle
        crackle = np.random.normal(0, 0.01, len(audio))
        
        # Apply low-pass filter
        from scipy import signal
        b, a = signal.butter(2, 0.2, btype='low')
        filtered = signal.filtfilt(b, a, audio)
        
        return filtered + crackle
    
    def save_music(self, music: np.ndarray, filename: str, format: str = "wav") -> bool:
        """Save generated music to file"""
        try:
            if len(music) == 0:
                print("No music to save")
                return False
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
            
            # Save audio
            if format.lower() == "wav":
                sf.write(filename, music, self.sample_rate)
            elif format.lower() == "mp3":
                from pydub import AudioSegment
                # Convert to pydub format and export as mp3
                audio_int16 = (music * 32767).astype(np.int16)
                audio_segment = AudioSegment(
                    audio_int16.tobytes(),
                    frame_rate=self.sample_rate,
                    sample_width=2,
                    channels=1
                )
                audio_segment.export(filename, format="mp3")
            else:
                sf.write(filename, music, self.sample_rate)
            
            print(f"✓ Music saved to: {filename}")
            return True
            
        except Exception as e:
            print(f"Failed to save music: {e}")
            return False
    
    def get_available_styles(self) -> List[str]:
        """Get list of available music styles"""
        return list(self.style_presets.keys())
    
    def get_available_genres(self) -> List[str]:
        """Get list of available genres"""
        return list(self.genre_presets.keys())
    
    def get_music_info(self, music: np.ndarray) -> Dict:
        """Get information about the generated music"""
        if len(music) == 0:
            return {"duration": 0, "sample_rate": 0, "channels": 0}
        
        duration = len(music) / self.sample_rate
        return {
            "duration": duration,
            "sample_rate": self.sample_rate,
            "channels": 1 if music.ndim == 1 else music.shape[1],
            "samples": len(music),
            "max_amplitude": float(np.max(np.abs(music))),
            "rms_level": float(np.sqrt(np.mean(music**2)))
        }


# Example usage
if __name__ == "__main__":
    # Initialize music generator
    generator = MusicGenerator()
    
    # Test basic generation
    print("Testing music generation...")
    music = generator.generate_music(
        prompt="A peaceful ambient track with soft piano and strings",
        duration=30,  # 30 seconds for testing
        style="ambient",
        mood="calm"
    )
    
    if len(music) > 0:
        # Save the result
        generator.save_music(music, "test_music.wav")
        
        # Get music info
        info = generator.get_music_info(music)
        print(f"Generated music: {info['duration']:.2f}s, {info['samples']} samples")
    else:
        print("Failed to generate music")
    
    print("Music generation test completed!")
