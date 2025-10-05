"""
Advanced Music Style and Genre Control for Hydax AI Music Generator
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

class MusicStyleController:
    """
    Advanced music style and genre controller
    """
    
    def __init__(self):
        """Initialize the music style controller"""
        self.styles = self._create_advanced_styles()
        self.genres = self._create_advanced_genres()
        self.moods = self._create_mood_presets()
        self.instruments = self._create_instrument_presets()
        self.effects = self._create_effect_presets()
        
    def _create_advanced_styles(self) -> Dict:
        """Create advanced music style presets"""
        return {
            'ambient': {
                'tempo_range': (40, 80),
                'key_signatures': ['D', 'A', 'E', 'F#'],
                'time_signatures': ['4/4', '3/4'],
                'instruments': ['pad', 'strings', 'ambient', 'field_recordings'],
                'dynamics': 'soft',
                'harmony': 'diatonic',
                'rhythm': 'sparse',
                'effects': ['reverb', 'delay', 'chorus', 'lowpass'],
                'structure': 'free_form',
                'characteristics': ['atmospheric', 'textural', 'evolving', 'meditative']
            },
            'electronic': {
                'tempo_range': (120, 140),
                'key_signatures': ['C', 'G', 'D', 'A'],
                'time_signatures': ['4/4'],
                'instruments': ['synth', 'bass', 'drums', 'lead', 'fx'],
                'dynamics': 'dynamic',
                'harmony': 'modal',
                'rhythm': 'complex',
                'effects': ['distortion', 'filter', 'sidechain', 'compression'],
                'structure': 'build_drop',
                'characteristics': ['synthetic', 'rhythmic', 'energetic', 'futuristic']
            },
            'classical': {
                'tempo_range': (60, 120),
                'key_signatures': ['C', 'G', 'D', 'A', 'F', 'Bb'],
                'time_signatures': ['4/4', '3/4', '2/4'],
                'instruments': ['strings', 'piano', 'woodwinds', 'brass', 'percussion'],
                'dynamics': 'expressive',
                'harmony': 'functional',
                'rhythm': 'structured',
                'effects': ['reverb', 'eq', 'compression'],
                'structure': 'sonata_form',
                'characteristics': ['orchestral', 'melodic', 'harmonic', 'sophisticated']
            },
            'jazz': {
                'tempo_range': (80, 200),
                'key_signatures': ['C', 'F', 'Bb', 'Eb', 'G', 'D'],
                'time_signatures': ['4/4', '3/4', '12/8'],
                'instruments': ['piano', 'bass', 'saxophone', 'trumpet', 'drums'],
                'dynamics': 'swing',
                'harmony': 'extended',
                'rhythm': 'syncopated',
                'effects': ['reverb', 'compression', 'eq'],
                'structure': 'head_solos_head',
                'characteristics': ['improvisational', 'sophisticated', 'rhythmic', 'harmonic']
            },
            'rock': {
                'tempo_range': (100, 160),
                'key_signatures': ['E', 'A', 'D', 'G', 'C'],
                'time_signatures': ['4/4'],
                'instruments': ['guitar', 'bass', 'drums', 'vocals', 'keyboard'],
                'dynamics': 'aggressive',
                'harmony': 'power_chords',
                'rhythm': 'driving',
                'effects': ['distortion', 'reverb', 'compression', 'eq'],
                'structure': 'verse_chorus',
                'characteristics': ['energetic', 'rhythmic', 'powerful', 'rebellious']
            },
            'pop': {
                'tempo_range': (100, 130),
                'key_signatures': ['C', 'G', 'D', 'A', 'F'],
                'time_signatures': ['4/4'],
                'instruments': ['synth', 'bass', 'drums', 'vocals', 'guitar'],
                'dynamics': 'polished',
                'harmony': 'catchy',
                'rhythm': 'steady',
                'effects': ['compression', 'eq', 'reverb', 'autotune'],
                'structure': 'verse_chorus_bridge',
                'characteristics': ['catchy', 'accessible', 'polished', 'commercial']
            },
            'cinematic': {
                'tempo_range': (60, 100),
                'key_signatures': ['C', 'G', 'D', 'A', 'F', 'Bb'],
                'time_signatures': ['4/4', '3/4', '2/4'],
                'instruments': ['orchestra', 'choir', 'percussion', 'solo_instruments'],
                'dynamics': 'epic',
                'harmony': 'dramatic',
                'rhythm': 'varied',
                'effects': ['reverb', 'eq', 'compression', 'stereo_imaging'],
                'structure': 'narrative',
                'characteristics': ['dramatic', 'emotional', 'orchestral', 'cinematic']
            },
            'lo-fi': {
                'tempo_range': (70, 90),
                'key_signatures': ['C', 'F', 'G', 'Am', 'Dm'],
                'time_signatures': ['4/4'],
                'instruments': ['piano', 'bass', 'drums', 'vinyl', 'guitar'],
                'dynamics': 'chill',
                'harmony': 'jazzy',
                'rhythm': 'relaxed',
                'effects': ['vinyl_crackle', 'lowpass', 'compression', 'tape_saturation'],
                'structure': 'loop_based',
                'characteristics': ['relaxed', 'nostalgic', 'warm', 'chill']
            },
            'experimental': {
                'tempo_range': (40, 180),
                'key_signatures': ['C', 'F#', 'Bb', 'E'],
                'time_signatures': ['4/4', '5/4', '7/8', 'free'],
                'instruments': ['synthesizers', 'field_recordings', 'found_sounds', 'vocals'],
                'dynamics': 'unpredictable',
                'harmony': 'atonal',
                'rhythm': 'irregular',
                'effects': ['granular', 'spectral', 'modulation', 'distortion'],
                'structure': 'free_form',
                'characteristics': ['avant_garde', 'unconventional', 'textural', 'innovative']
            },
            'world': {
                'tempo_range': (60, 140),
                'key_signatures': ['C', 'G', 'D', 'A', 'F'],
                'time_signatures': ['4/4', '3/4', '6/8', '12/8'],
                'instruments': ['ethnic_instruments', 'percussion', 'vocals', 'strings'],
                'dynamics': 'expressive',
                'harmony': 'modal',
                'rhythm': 'polyrhythmic',
                'effects': ['reverb', 'eq', 'compression'],
                'structure': 'call_response',
                'characteristics': ['cultural', 'rhythmic', 'melodic', 'traditional']
            }
        }
    
    def _create_advanced_genres(self) -> Dict:
        """Create advanced genre presets"""
        return {
            'house': {
                'bpm': 128,
                'key': 'C',
                'time_signature': '4/4',
                'structure': 'intro-verse-chorus-verse-chorus-bridge-chorus-outro',
                'instruments': ['kick', 'hihat', 'bass', 'lead', 'pad', 'vocal_samples'],
                'characteristics': ['four_on_floor', 'repetitive', 'groove_based', 'club_ready'],
                'effects': ['sidechain_compression', 'filter_sweeps', 'reverb', 'delay']
            },
            'trance': {
                'bpm': 132,
                'key': 'A',
                'time_signature': '4/4',
                'structure': 'intro-build-drop-breakdown-build-drop-outro',
                'instruments': ['kick', 'bass', 'lead', 'pad', 'arpeggio', 'vocal_pads'],
                'characteristics': ['uplifting', 'melodic', 'euphoric', 'progressive'],
                'effects': ['reverb', 'delay', 'chorus', 'compression']
            },
            'dubstep': {
                'bpm': 140,
                'key': 'F#',
                'time_signature': '4/4',
                'structure': 'intro-drop-breakdown-drop-outro',
                'instruments': ['kick', 'snare', 'bass', 'wobble', 'fx', 'vocals'],
                'characteristics': ['heavy', 'aggressive', 'bass_heavy', 'syncopated'],
                'effects': ['distortion', 'filter_modulation', 'reverb', 'compression']
            },
            'techno': {
                'bpm': 130,
                'key': 'C',
                'time_signature': '4/4',
                'structure': 'intro-loop-breakdown-loop-outro',
                'instruments': ['kick', 'hihat', 'percussion', 'bass', 'lead', 'fx'],
                'characteristics': ['mechanical', 'repetitive', 'driving', 'minimal'],
                'effects': ['distortion', 'filter', 'delay', 'compression']
            },
            'drum_and_bass': {
                'bpm': 174,
                'key': 'A',
                'time_signature': '4/4',
                'structure': 'intro-drop-breakdown-drop-outro',
                'instruments': ['kick', 'snare', 'bass', 'lead', 'fx', 'vocals'],
                'characteristics': ['fast', 'complex', 'bass_heavy', 'breakbeat'],
                'effects': ['distortion', 'filter', 'reverb', 'compression']
            },
            'ambient_techno': {
                'bpm': 120,
                'key': 'D',
                'time_signature': '4/4',
                'structure': 'intro-development-climax-resolution-outro',
                'instruments': ['kick', 'pad', 'bass', 'ambient', 'field_recordings'],
                'characteristics': ['atmospheric', 'textural', 'evolving', 'meditative'],
                'effects': ['reverb', 'delay', 'lowpass', 'compression']
            },
            'orchestral': {
                'bpm': 90,
                'key': 'C',
                'time_signature': '4/4',
                'structure': 'exposition-development-recapitulation',
                'instruments': ['strings', 'brass', 'woodwinds', 'percussion', 'harp'],
                'characteristics': ['symphonic', 'melodic', 'harmonic', 'dramatic'],
                'effects': ['reverb', 'eq', 'compression', 'stereo_imaging']
            },
            'neo_soul': {
                'bpm': 95,
                'key': 'F',
                'time_signature': '4/4',
                'structure': 'intro-verse-chorus-verse-chorus-bridge-chorus-outro',
                'instruments': ['piano', 'bass', 'drums', 'guitar', 'vocals', 'horns'],
                'characteristics': ['smooth', 'rhythmic', 'harmonic', 'soulful'],
                'effects': ['reverb', 'compression', 'eq', 'chorus']
            },
            'progressive_rock': {
                'bpm': 110,
                'key': 'G',
                'time_signature': '4/4',
                'structure': 'intro-verse-chorus-verse-chorus-instrumental-bridge-chorus-outro',
                'instruments': ['guitar', 'bass', 'drums', 'keyboard', 'vocals'],
                'characteristics': ['complex', 'melodic', 'harmonic', 'virtuosic'],
                'effects': ['distortion', 'reverb', 'delay', 'compression']
            },
            'trip_hop': {
                'bpm': 90,
                'key': 'Am',
                'time_signature': '4/4',
                'structure': 'intro-verse-chorus-verse-chorus-bridge-chorus-outro',
                'instruments': ['drums', 'bass', 'synth', 'vocals', 'samples'],
                'characteristics': ['laid_back', 'atmospheric', 'rhythmic', 'moody'],
                'effects': ['reverb', 'delay', 'lowpass', 'compression']
            }
        }
    
    def _create_mood_presets(self) -> Dict:
        """Create mood presets for music generation"""
        return {
            'happy': {
                'tempo_modifier': 1.1,
                'key_preference': ['C', 'G', 'D', 'A'],
                'harmony_style': 'major',
                'dynamics': 'bright',
                'rhythm': 'upbeat',
                'characteristics': ['uplifting', 'energetic', 'positive', 'cheerful']
            },
            'sad': {
                'tempo_modifier': 0.8,
                'key_preference': ['Am', 'Em', 'Dm', 'Gm'],
                'harmony_style': 'minor',
                'dynamics': 'soft',
                'rhythm': 'slow',
                'characteristics': ['melancholic', 'emotional', 'introspective', 'touching']
            },
            'energetic': {
                'tempo_modifier': 1.3,
                'key_preference': ['C', 'G', 'D', 'A'],
                'harmony_style': 'major',
                'dynamics': 'loud',
                'rhythm': 'driving',
                'characteristics': ['powerful', 'intense', 'motivating', 'dynamic']
            },
            'calm': {
                'tempo_modifier': 0.7,
                'key_preference': ['C', 'F', 'G', 'Am'],
                'harmony_style': 'diatonic',
                'dynamics': 'soft',
                'rhythm': 'gentle',
                'characteristics': ['peaceful', 'relaxing', 'serene', 'meditative']
            },
            'mysterious': {
                'tempo_modifier': 0.9,
                'key_preference': ['F#', 'C#', 'G#', 'D#'],
                'harmony_style': 'modal',
                'dynamics': 'moderate',
                'rhythm': 'unpredictable',
                'characteristics': ['enigmatic', 'atmospheric', 'intriguing', 'suspenseful']
            },
            'romantic': {
                'tempo_modifier': 0.8,
                'key_preference': ['C', 'F', 'G', 'Am'],
                'harmony_style': 'major',
                'dynamics': 'expressive',
                'rhythm': 'flowing',
                'characteristics': ['passionate', 'emotional', 'intimate', 'beautiful']
            },
            'aggressive': {
                'tempo_modifier': 1.2,
                'key_preference': ['E', 'A', 'D', 'G'],
                'harmony_style': 'power_chords',
                'dynamics': 'loud',
                'rhythm': 'pounding',
                'characteristics': ['intense', 'powerful', 'confrontational', 'energetic']
            },
            'nostalgic': {
                'tempo_modifier': 0.9,
                'key_preference': ['C', 'F', 'G', 'Am'],
                'harmony_style': 'major',
                'dynamics': 'warm',
                'rhythm': 'steady',
                'characteristics': ['wistful', 'sentimental', 'retro', 'comforting']
            }
        }
    
    def _create_instrument_presets(self) -> Dict:
        """Create instrument presets"""
        return {
            'electronic': {
                'synth': {'type': 'synthesizer', 'characteristics': ['synthetic', 'modular', 'digital']},
                'bass': {'type': 'bass_synth', 'characteristics': ['deep', 'punchy', 'synthetic']},
                'lead': {'type': 'lead_synth', 'characteristics': ['melodic', 'bright', 'synthetic']},
                'pad': {'type': 'pad_synth', 'characteristics': ['atmospheric', 'sustained', 'textural']},
                'drums': {'type': 'electronic_drums', 'characteristics': ['punchy', 'precise', 'synthetic']},
                'fx': {'type': 'effects', 'characteristics': ['atmospheric', 'textural', 'synthetic']}
            },
            'acoustic': {
                'piano': {'type': 'acoustic_piano', 'characteristics': ['warm', 'natural', 'harmonic']},
                'guitar': {'type': 'acoustic_guitar', 'characteristics': ['warm', 'natural', 'melodic']},
                'bass': {'type': 'acoustic_bass', 'characteristics': ['warm', 'natural', 'rhythmic']},
                'drums': {'type': 'acoustic_drums', 'characteristics': ['natural', 'dynamic', 'rhythmic']},
                'strings': {'type': 'string_section', 'characteristics': ['warm', 'expressive', 'harmonic']},
                'brass': {'type': 'brass_section', 'characteristics': ['bright', 'powerful', 'harmonic']}
            },
            'world': {
                'sitar': {'type': 'indian_sitar', 'characteristics': ['exotic', 'melodic', 'resonant']},
                'djembe': {'type': 'african_drum', 'characteristics': ['rhythmic', 'percussive', 'natural']},
                'shakuhachi': {'type': 'japanese_flute', 'characteristics': ['meditative', 'melodic', 'breathy']},
                'oud': {'type': 'middle_eastern_lute', 'characteristics': ['exotic', 'melodic', 'resonant']},
                'hang_drum': {'type': 'steel_drum', 'characteristics': ['metallic', 'resonant', 'meditative']},
                'didgeridoo': {'type': 'australian_drone', 'characteristics': ['drone', 'resonant', 'primal']}
            }
        }
    
    def _create_effect_presets(self) -> Dict:
        """Create effect presets"""
        return {
            'reverb': {
                'room': {'size': 0.3, 'damping': 0.5, 'wet': 0.2},
                'hall': {'size': 0.7, 'damping': 0.3, 'wet': 0.4},
                'cathedral': {'size': 1.0, 'damping': 0.1, 'wet': 0.6},
                'plate': {'size': 0.5, 'damping': 0.4, 'wet': 0.3}
            },
            'delay': {
                'short': {'time': 0.125, 'feedback': 0.3, 'wet': 0.2},
                'medium': {'time': 0.25, 'feedback': 0.4, 'wet': 0.3},
                'long': {'time': 0.5, 'feedback': 0.5, 'wet': 0.4},
                'ping_pong': {'time': 0.25, 'feedback': 0.6, 'wet': 0.3}
            },
            'distortion': {
                'light': {'drive': 0.3, 'tone': 0.5, 'level': 0.7},
                'medium': {'drive': 0.6, 'tone': 0.4, 'level': 0.8},
                'heavy': {'drive': 0.9, 'tone': 0.3, 'level': 0.9},
                'fuzz': {'drive': 1.0, 'tone': 0.2, 'level': 0.8}
            },
            'filter': {
                'lowpass': {'cutoff': 0.3, 'resonance': 0.5, 'type': 'lowpass'},
                'highpass': {'cutoff': 0.7, 'resonance': 0.3, 'type': 'highpass'},
                'bandpass': {'cutoff': 0.5, 'resonance': 0.7, 'type': 'bandpass'},
                'notch': {'cutoff': 0.5, 'resonance': 0.8, 'type': 'notch'}
            }
        }
    
    def get_style_parameters(self, style: str) -> Dict:
        """Get parameters for a specific style"""
        return self.styles.get(style, self.styles['ambient'])
    
    def get_genre_parameters(self, genre: str) -> Dict:
        """Get parameters for a specific genre"""
        return self.genres.get(genre, self.genres['ambient'])
    
    def get_mood_parameters(self, mood: str) -> Dict:
        """Get parameters for a specific mood"""
        return self.moods.get(mood, self.moods['neutral'])
    
    def get_instrument_preset(self, category: str, instrument: str) -> Dict:
        """Get instrument preset"""
        category_presets = self.instruments.get(category, {})
        return category_presets.get(instrument, {})
    
    def get_effect_preset(self, effect_type: str, preset: str) -> Dict:
        """Get effect preset"""
        effect_presets = self.effects.get(effect_type, {})
        return effect_presets.get(preset, {})
    
    def combine_style_and_mood(self, style: str, mood: str) -> Dict:
        """Combine style and mood parameters"""
        style_params = self.get_style_parameters(style)
        mood_params = self.get_mood_parameters(mood)
        
        # Combine parameters
        combined = style_params.copy()
        
        # Apply mood modifications
        if 'tempo_range' in style_params:
            tempo_range = style_params['tempo_range']
            modifier = mood_params.get('tempo_modifier', 1.0)
            combined['tempo_range'] = (
                int(tempo_range[0] * modifier),
                int(tempo_range[1] * modifier)
            )
        
        # Override key preferences with mood preferences
        if 'key_preference' in mood_params:
            combined['key_signatures'] = mood_params['key_preference']
        
        # Add mood characteristics
        if 'characteristics' in combined:
            combined['characteristics'].extend(mood_params.get('characteristics', []))
        else:
            combined['characteristics'] = mood_params.get('characteristics', [])
        
        return combined
    
    def get_available_styles(self) -> List[str]:
        """Get list of available styles"""
        return list(self.styles.keys())
    
    def get_available_genres(self) -> List[str]:
        """Get list of available genres"""
        return list(self.genres.keys())
    
    def get_available_moods(self) -> List[str]:
        """Get list of available moods"""
        return list(self.moods.keys())
    
    def get_available_instruments(self, category: str = None) -> List[str]:
        """Get list of available instruments"""
        if category:
            return list(self.instruments.get(category, {}).keys())
        else:
            all_instruments = []
            for category_instruments in self.instruments.values():
                all_instruments.extend(category_instruments.keys())
            return all_instruments
    
    def get_available_effects(self, effect_type: str = None) -> List[str]:
        """Get list of available effects"""
        if effect_type:
            return list(self.effects.get(effect_type, {}).keys())
        else:
            return list(self.effects.keys())
    
    def create_custom_style(self, name: str, parameters: Dict) -> bool:
        """Create a custom style preset"""
        try:
            self.styles[name] = parameters
            return True
        except Exception as e:
            print(f"Failed to create custom style: {e}")
            return False
    
    def create_custom_genre(self, name: str, parameters: Dict) -> bool:
        """Create a custom genre preset"""
        try:
            self.genres[name] = parameters
            return True
        except Exception as e:
            print(f"Failed to create custom genre: {e}")
            return False
    
    def create_custom_mood(self, name: str, parameters: Dict) -> bool:
        """Create a custom mood preset"""
        try:
            self.moods[name] = parameters
            return True
        except Exception as e:
            print(f"Failed to create custom mood: {e}")
            return False
    
    def analyze_music_style(self, audio: np.ndarray, sample_rate: int) -> Dict:
        """Analyze the style characteristics of audio"""
        try:
            # Extract features for style analysis
            tempo = self._estimate_tempo(audio, sample_rate)
            key = self._estimate_key(audio, sample_rate)
            dynamics = self._analyze_dynamics(audio)
            rhythm = self._analyze_rhythm(audio, sample_rate)
            
            return {
                'tempo': tempo,
                'key': key,
                'dynamics': dynamics,
                'rhythm': rhythm,
                'style_suggestions': self._suggest_styles(tempo, key, dynamics, rhythm)
            }
            
        except Exception as e:
            print(f"Style analysis failed: {e}")
            return {}
    
    def _estimate_tempo(self, audio: np.ndarray, sample_rate: int) -> float:
        """Estimate tempo from audio"""
        try:
            # Use librosa for tempo estimation
            tempo, _ = librosa.beat.beat_track(y=audio, sr=sample_rate)
            return float(tempo)
        except:
            return 120.0  # Default tempo
    
    def _estimate_key(self, audio: np.ndarray, sample_rate: int) -> str:
        """Estimate musical key from audio"""
        try:
            # Simple key estimation (in practice, you'd use more sophisticated methods)
            chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
            chroma_mean = np.mean(chroma, axis=1)
            key_index = np.argmax(chroma_mean)
            keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            return keys[key_index]
        except:
            return 'C'  # Default key
    
    def _analyze_dynamics(self, audio: np.ndarray) -> str:
        """Analyze dynamic characteristics"""
        try:
            rms = np.sqrt(np.mean(audio**2))
            if rms < 0.1:
                return 'soft'
            elif rms < 0.3:
                return 'moderate'
            else:
                return 'loud'
        except:
            return 'moderate'
    
    def _analyze_rhythm(self, audio: np.ndarray, sample_rate: int) -> str:
        """Analyze rhythmic characteristics"""
        try:
            # Simple rhythm analysis
            onset_frames = librosa.onset.onset_detect(y=audio, sr=sample_rate)
            onset_times = librosa.frames_to_time(onset_frames, sr=sample_rate)
            
            if len(onset_times) < 10:
                return 'sparse'
            elif len(onset_times) < 50:
                return 'moderate'
            else:
                return 'complex'
        except:
            return 'moderate'
    
    def _suggest_styles(self, tempo: float, key: str, dynamics: str, rhythm: str) -> List[str]:
        """Suggest styles based on analysis"""
        suggestions = []
        
        # Tempo-based suggestions
        if tempo < 80:
            suggestions.extend(['ambient', 'lo-fi', 'cinematic'])
        elif tempo < 120:
            suggestions.extend(['classical', 'jazz', 'pop'])
        else:
            suggestions.extend(['electronic', 'rock', 'house', 'trance'])
        
        # Dynamics-based suggestions
        if dynamics == 'soft':
            suggestions.extend(['ambient', 'lo-fi', 'cinematic'])
        elif dynamics == 'loud':
            suggestions.extend(['rock', 'electronic', 'dubstep'])
        
        # Rhythm-based suggestions
        if rhythm == 'sparse':
            suggestions.extend(['ambient', 'cinematic', 'lo-fi'])
        elif rhythm == 'complex':
            suggestions.extend(['jazz', 'progressive_rock', 'drum_and_bass'])
        
        # Remove duplicates and return top suggestions
        return list(set(suggestions))[:5]


# Example usage
if __name__ == "__main__":
    # Initialize style controller
    controller = MusicStyleController()
    
    # Test style parameters
    style_params = controller.get_style_parameters('ambient')
    print(f"Ambient style parameters: {style_params}")
    
    # Test mood parameters
    mood_params = controller.get_mood_parameters('happy')
    print(f"Happy mood parameters: {mood_params}")
    
    # Test combining style and mood
    combined = controller.combine_style_and_mood('electronic', 'energetic')
    print(f"Combined electronic + energetic: {combined}")
    
    # Test available options
    print(f"Available styles: {controller.get_available_styles()}")
    print(f"Available genres: {controller.get_available_genres()}")
    print(f"Available moods: {controller.get_available_moods()}")
    
    print("Music style controller initialized successfully!")
