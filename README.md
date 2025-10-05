# 🎤🎵🎬 Hydax AI - Complete Audio & Video Suite

A powerful, realistic AI-powered Text-to-Speech engine, Music Generator, and Video Generator built with Python. This complete multimedia suite supports multiple AI models, voice cloning, emotion control, music generation, video creation, and real-time processing with beautiful web interfaces and a modern desktop GUI.

## 🖥️ **Desktop GUI Application!**
- **Beautiful modern interface** with dark theme
- **Quick access** to TTS, Music, and Video generation
- **One-click launch** - just run `python hydax_ai.py`
- **No browser needed** - native desktop application
- **All-in-one file** - everything in a single application

## ✨ Features

### 🎯 Core TTS Capabilities
- **Multiple AI Models**: Support for Coqui TTS, pyttsx3, and custom models
- **High-Quality Synthesis**: Realistic, natural-sounding speech generation
- **Multi-Language Support**: English, Spanish, French, German, Italian, Portuguese, Russian, Chinese, Japanese, Korean
- **Real-time Processing**: Fast inference for real-time applications

### 🎭 Voice Cloning
- **Advanced Voice Cloning**: Clone voices from audio samples using XTTS
- **Voice Embeddings**: Save and load voice embeddings for reuse
- **Reference Text Support**: Automatic transcription or manual reference text
- **Voice Management**: Create, delete, and manage multiple cloned voices

### 😊 Emotion & Style Control
- **Predefined Emotions**: Happy, sad, angry, excited, calm, surprised, whisper, shouting
- **Speaking Styles**: Conversational, formal, dramatic, news, storytelling
- **Custom Emotions**: Create your own emotion presets
- **Intensity Control**: Adjust emotion and style intensity (0.0 to 2.0)

### 🔧 Audio Processing & Enhancement
- **Audio Enhancement**: Light, medium, full, and custom enhancement modes
- **Noise Reduction**: Advanced noise reduction algorithms
- **Audio Effects**: Reverb (room, hall, plate, spring), EQ (speech, music, bright, warm)
- **Quality Analysis**: Comprehensive audio quality metrics
- **Format Support**: WAV, MP3, FLAC output formats

### 🌐 Web Interface
- **Modern UI**: Beautiful, responsive web interface built with Gradio
- **Real-time Testing**: Test TTS features instantly in the browser
- **Batch Processing**: Process multiple texts simultaneously
- **File Management**: Upload, download, and manage audio files
- **Settings Panel**: Configure engine settings and preferences

### 📦 Batch Processing
- **Bulk Synthesis**: Process multiple texts in one operation
- **Progress Tracking**: Real-time progress updates
- **Export Options**: Download individual files or batch archives
- **Custom Parameters**: Apply different settings to each batch

### 🎵 Music Generation
- **AI-Powered Music Creation**: Generate music from text prompts
- **Multiple Styles**: Ambient, electronic, classical, jazz, rock, pop, cinematic, lo-fi
- **Genre Support**: House, trance, dubstep, orchestral, world music, and more
- **Mood Control**: Happy, sad, energetic, calm, mysterious, romantic, and more
- **Professional Mastering**: Club, radio, streaming, cinematic, ambient presets
- **Advanced Effects**: Reverb, delay, distortion, compression, EQ, stereo imaging
- **Custom Styles**: Create your own music style presets
- **Batch Generation**: Generate multiple tracks simultaneously

### 🎬 Video Generation
- **AI-Powered Video Creation**: Generate videos from text prompts
- **Multiple Styles**: Cinematic, documentary, animation, corporate, artistic, social media
- **Scene Types**: Nature, urban, abstract, technology, fantasy, space, underwater
- **Mood Control**: Peaceful, energetic, mysterious, romantic, dramatic, nostalgic
- **Professional Effects**: Color grading, stabilization, film grain, vignette, sharpening
- **Advanced Processing**: Transitions, text overlays, audio integration
- **Custom Styles**: Create your own video style presets
- **Batch Generation**: Generate multiple videos simultaneously
- **Duration Control**: 30 seconds to 5 minutes with customizable resolution

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (optional, for faster processing)

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Optional Dependencies
For enhanced voice cloning capabilities:
```bash
pip install TTS[all] openai-whisper
```

For advanced audio processing:
```bash
pip install noisereduce pydub
```

For music generation features:
```bash
pip install pretty_midi midi2audio music21 essentia
```

For video generation features:
```bash
pip install torchvision opencv-python moviepy Pillow
```

## 🎯 Quick Start

### Basic Usage
```python
from tts_engine import HydaxTTS

# Initialize the TTS engine
tts = HydaxTTS()

# Generate speech
audio = tts.synthesize("Hello, this is a test of the Hydax AI TTS engine.")

# Save audio
tts.save_audio(audio, "output.wav")
```

### Voice Cloning
```python
from voice_cloning import VoiceCloner

# Initialize voice cloner
cloner = VoiceCloner()

# Clone a voice from audio file
cloner.clone_voice_from_audio(
    voice_name="my_voice",
    audio_file="reference_audio.wav",
    reference_text="This is the reference text."
)

# Synthesize with cloned voice
audio = cloner.synthesize_with_cloned_voice(
    text="Hello, this is my cloned voice!",
    voice_name="my_voice"
)
```

### Emotion Control
```python
from emotion_control import EmotionController

# Initialize emotion controller
emotion_controller = EmotionController()

# Apply emotion to audio
emotional_audio = emotion_controller.apply_emotion(
    audio=base_audio,
    emotion="happy",
    intensity=1.0
)

# Apply speaking style
styled_audio = emotion_controller.apply_style(
    audio=emotional_audio,
    style="dramatic",
    intensity=0.8
)
```

### Audio Enhancement
```python
from audio_processing import AudioProcessor

# Initialize audio processor
processor = AudioProcessor()

# Enhance audio
enhanced_audio = processor.enhance_audio(
    audio=raw_audio,
    enhancement_type="full",
    noise_reduction=True,
    normalize_audio=True,
    remove_silence=True,
    enhance_clarity=True
)

# Add effects
final_audio = processor.add_reverb(enhanced_audio, "hall", 0.3)
final_audio = processor.apply_eq(final_audio, "speech")
```

### Music Generation
```python
from music_generator import MusicGenerator

# Initialize music generator
generator = MusicGenerator()

# Generate music from prompt
music = generator.generate_music(
    prompt="A peaceful ambient track with soft piano and strings",
    duration=120,  # 2 minutes
    style="ambient",
    mood="calm",
    tempo=80
)

# Save the generated music
generator.save_music(music, "my_music.wav")
```

### Video Generation
```python
from video_generator import VideoGenerator

# Initialize video generator
generator = VideoGenerator()

# Generate video from prompt
video_path = generator.generate_video(
    prompt="A peaceful nature scene with trees and sunset",
    duration=120,  # 2 minutes
    style="cinematic",
    scene_type="nature",
    mood="peaceful"
)

# The video is automatically saved to video_outputs/
```

### Music Style Control
```python
from music_styles import MusicStyleController

# Initialize style controller
controller = MusicStyleController()

# Get style parameters
style_params = controller.get_style_parameters("electronic")
mood_params = controller.get_mood_parameters("energetic")

# Combine style and mood
combined_params = controller.combine_style_and_mood("electronic", "energetic")
```

### Music Processing & Mastering
```python
from music_processing import MusicProcessor

# Initialize music processor
processor = MusicProcessor()

# Master music for different platforms
mastered_audio = processor.master_music(
    audio=raw_music,
    preset="streaming"  # or "club", "radio", "cinematic"
)

# Apply effect chains
processed_audio = processor.apply_effect_chain(
    audio=mastered_audio,
    style="electronic"
)
```

### Video Style Control
```python
from video_styles import VideoStyleController

# Initialize style controller
controller = VideoStyleController()

# Get available styles
styles = controller.get_available_styles()
print(f"Available styles: {styles}")

# Get style parameters
params = controller.get_style_parameters("cinematic")
print(f"Cinematic style: {params}")

# Combine style and mood
combined = controller.combine_style_and_mood("cinematic", "dramatic")
```

### Video Processing
```python
from video_processing import VideoProcessor

# Initialize processor
processor = VideoProcessor()

# Process video with effects
processed_video = processor.process_video(
    video_path="input_video.mp4",
    style="cinematic",
    custom_effects={
        'color_grading': {'warm': True, 'contrast': 1.2},
        'film_grain': {'intensity': 0.3},
        'vignette': {'intensity': 0.4}
    }
)
```

## 🖥️ Desktop GUI Application

### Launch the Desktop GUI
```bash
# Method 1: Python command
python hydax_ai.py

# Method 2: Batch file (Windows)
"Hydax AI.bat"
```

The desktop GUI provides:
- **🚀 Quick Start**: Fast access to all features
- **🎤 TTS Generation**: Create speech from text
- **🎵 Music Generation**: Generate music from prompts
- **🎬 Video Generation**: Create videos from descriptions
- **🎭 Multimedia Creation**: Combine TTS, Music, and Video
- **⚙️ Settings**: Configure and manage the application
- **🌐 Web Interface Launcher**: Launch the full web interface

### Features
- 🎨 **Modern Dark Theme**: Beautiful, easy-on-the-eyes interface
- 🚀 **Quick Access**: Launch any feature with one click
- 🎯 **Intuitive Controls**: Simple dropdowns and text areas
- 📊 **Status Updates**: Real-time progress tracking
- 💾 **Auto-Save**: All generated content saved automatically
- 🔄 **Background Processing**: Generate while using other tabs
- 📁 **All-in-One**: Single file with all functionality

## 🌐 Web Interfaces

### Launch the Integrated Interface
```bash
python integrated_launcher.py
```

### Launch Individual Interfaces
```bash
# TTS Interface
python web_interface.py

# Music Generation Interface  
python music_web_interface.py

# Video Generation Interface
python video_web_interface.py

# Integrated Interface (TTS + Music + Video)
python integrated_web_interface.py
```

The interfaces will be available at:
- **Integrated Interface**: `http://localhost:7860`
- **TTS Interface**: `http://localhost:7860`
- **Music Interface**: `http://localhost:7861`
- **Video Interface**: `http://localhost:7862`

### Web Interface Features

#### 🎤 TTS Interface
- **🎯 Basic TTS**: Simple text-to-speech synthesis
- **🎭 Voice Cloning**: Clone voices from audio samples
- **😊 Emotion Control**: Apply emotions and speaking styles
- **🔧 Audio Processing**: Enhance and process audio files
- **📦 Batch Processing**: Process multiple texts at once
- **⚙️ Settings**: Configure engine settings and preferences

#### 🎵 Music Interface
- **🎼 Music Generation**: Generate music from text prompts
- **🎨 Style Control**: Control music styles, genres, and moods
- **🔧 Music Processing**: Professional mastering and effects
- **📦 Batch Generation**: Generate multiple tracks simultaneously
- **⚙️ Advanced Settings**: Configure music generation parameters

#### 🎬 Video Interface
- **🎬 Video Generation**: Create videos from text prompts
- **🎨 Style Control**: Choose from multiple video styles and scenes
- **🔧 Processing**: Apply professional effects and color grading
- **📦 Batch Generation**: Generate multiple videos at once
- **🎛️ Advanced Settings**: Customize resolution, duration, and effects

#### 🎭 Integrated Interface
- **🚀 Quick Start**: Fast TTS, music, and video generation
- **🎬 Multimedia Projects**: Create complete multimedia content
- **🔧 Unified Processing**: Process TTS, music, and video together
- **📦 Mixed Batch**: Process TTS, music, and video in batches

## 📚 Examples

### Run All Examples
```bash
# Run all examples (TTS + Music)
python integrated_launcher.py --examples

# Run TTS examples only
python examples.py

# Run Music examples only
python music_examples.py

# Run Video examples only
python video_examples.py
```

### TTS Examples
This will demonstrate:
1. Basic TTS synthesis
2. Emotion and style control
3. Voice cloning (requires reference audio)
4. Audio processing and enhancement
5. Batch processing
6. Custom emotion creation
7. Advanced workflow combining multiple features

### Music Examples
This will demonstrate:
1. Basic music generation
2. Style and genre control
3. Mood control in music
4. Music processing and enhancement
5. Custom style creation
6. Advanced workflow with mastering
7. Batch music generation

### Video Examples
This will demonstrate:
1. Basic video generation
2. Style and scene control
3. Video processing and effects
4. Text overlays and transitions
5. Batch generation
6. Custom style creation
7. Advanced workflow with post-processing

## ⚙️ Configuration

### Environment Variables
```bash
# Model settings
export TTS_DEFAULT_MODEL="tts_models/en/ljspeech/tacotron2-DDC"
export TTS_DEVICE="auto"  # auto, cpu, cuda
export TTS_USE_GPU="true"

# Audio settings
export TTS_SAMPLE_RATE="22050"
export TTS_AUDIO_FORMAT="wav"

# Web interface settings
export TTS_WEB_HOST="127.0.0.1"
export TTS_WEB_PORT="7860"
export TTS_WEB_SHARE="false"

# Output settings
export TTS_OUTPUT_DIR="outputs"
export TTS_EXAMPLES_DIR="examples"
```

### Configuration File
```python
from config import config

# Access configuration
model_config = config.get_model_config()
audio_config = config.get_audio_config()

# Save/load configuration
config.save_config("my_config.json")
config.load_config("my_config.json")
```

## 🎭 Supported Models

### TTS Models
- **Coqui TTS**: Various high-quality models
  - `tts_models/en/ljspeech/tacotron2-DDC`
  - `tts_models/en/vctk/vits`
  - `tts_models/multilingual/multi-dataset/xtts_v2`
- **pyttsx3**: Cross-platform TTS engine
- **Custom Models**: Support for fine-tuned models

### Voice Cloning Models
- **XTTS v2**: Advanced voice cloning with multilingual support
- **Custom Embeddings**: Save and reuse voice embeddings

## 🎨 Emotion & Style Presets

### Predefined Emotions
- **neutral**: Balanced, natural speech
- **happy**: Upward pitch, faster speed, higher energy
- **sad**: Downward pitch, slower speed, lower energy
- **angry**: Sharp pitch changes, faster speed, high energy
- **excited**: High pitch, fast speed, very high energy
- **calm**: Lower pitch, slower speed, relaxed energy
- **surprised**: Sudden pitch changes, moderate speed
- **whisper**: Low energy, quiet breathing
- **shouting**: High energy, strong breathing

### Speaking Styles
- **conversational**: Natural pauses, irregular rhythm
- **formal**: Structured pauses, regular rhythm
- **dramatic**: Expressive pauses, varied rhythm
- **news**: Clear pauses, steady rhythm
- **storytelling**: Engaging pauses, flowing rhythm

## 🔧 Audio Enhancement Options

### Enhancement Types
- **light**: Basic normalization and light noise reduction
- **medium**: Remove silence, noise reduction, clarity enhancement
- **full**: Comprehensive enhancement with all features
- **custom**: User-defined enhancement parameters

### Audio Effects
- **Reverb**: Room, hall, plate, spring reverb types
- **EQ**: Speech, music, bright, warm equalizer presets
- **Compression**: Dynamic range compression
- **Noise Reduction**: Advanced noise reduction algorithms

## 📊 Performance & Optimization

### Hardware Requirements
- **CPU**: Multi-core processor recommended
- **RAM**: 4GB minimum, 8GB recommended
- **GPU**: CUDA-compatible GPU for faster processing (optional)
- **Storage**: 2GB for models and dependencies

### Performance Tips
- Use GPU acceleration when available
- Enable audio caching for repeated synthesis
- Use batch processing for multiple texts
- Optimize audio enhancement settings based on your needs

## 🐛 Troubleshooting

### Common Issues

#### Model Loading Errors
```bash
# Reinstall TTS with all dependencies
pip uninstall TTS
pip install TTS[all]
```

#### CUDA/GPU Issues
```bash
# Check CUDA installation
python -c "import torch; print(torch.cuda.is_available())"

# Use CPU if GPU issues persist
export TTS_DEVICE="cpu"
```

#### Audio Quality Issues
- Ensure input text is clean and properly formatted
- Try different enhancement settings
- Check audio sample rate compatibility
- Verify model compatibility with your hardware

#### Web Interface Issues
```bash
# Check if port is available
netstat -an | grep 7860

# Use different port
export TTS_WEB_PORT="7861"
```

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Coqui TTS](https://github.com/coqui-ai/TTS) for the excellent TTS models
- [Gradio](https://gradio.app/) for the beautiful web interface framework
- [Librosa](https://librosa.org/) for audio processing capabilities
- [PyTorch](https://pytorch.org/) for the deep learning framework

## 📞 Support

- **Issues**: Report bugs and request features on GitHub Issues
- **Discussions**: Join community discussions on GitHub Discussions
- **Documentation**: Check the examples and configuration files

---

**Made with ❤️ by the Hydax AI Team**
