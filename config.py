"""
Configuration file for Hydax AI TTS Engine
"""

import os
from pathlib import Path
from typing import Dict, List, Optional

class TTSConfig:
    """Configuration class for the TTS engine"""
    
    def __init__(self):
        """Initialize configuration with default values"""
        self._load_defaults()
        self._load_from_env()
    
    def _load_defaults(self):
        """Load default configuration values"""
        # Model settings
        self.DEFAULT_MODEL = "tts_models/en/ljspeech/tacotron2-DDC"
        self.DEFAULT_DEVICE = "auto"
        self.USE_GPU = True
        
        # Audio settings
        self.SAMPLE_RATE = 22050
        self.TARGET_LENGTH = None
        self.AUDIO_FORMAT = "wav"
        
        # Voice cloning settings
        self.VOICE_CLONING_ENABLED = True
        self.MIN_VOICE_DURATION = 3.0  # seconds
        self.MAX_VOICE_DURATION = 30.0  # seconds
        self.VOICE_EMBEDDINGS_DIR = "voice_embeddings"
        
        # Emotion control settings
        self.EMOTION_INTENSITY_RANGE = (0.0, 2.0)
        self.STYLE_INTENSITY_RANGE = (0.0, 2.0)
        self.DEFAULT_EMOTION = "neutral"
        self.DEFAULT_STYLE = "conversational"
        
        # Audio processing settings
        self.ENHANCEMENT_TYPES = ["light", "medium", "full", "custom"]
        self.DEFAULT_ENHANCEMENT = "medium"
        self.NOISE_REDUCTION_STRENGTH = 0.5
        self.REVERB_TYPES = ["room", "hall", "plate", "spring"]
        self.EQ_TYPES = ["speech", "music", "bright", "warm"]
        
        # Web interface settings
        self.WEB_HOST = "127.0.0.1"
        self.WEB_PORT = 7860
        self.WEB_SHARE = False
        self.MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
        
        # Batch processing settings
        self.MAX_BATCH_SIZE = 100
        self.BATCH_TIMEOUT = 300  # seconds
        
        # Cache settings
        self.CACHE_ENABLED = True
        self.MAX_CACHE_SIZE = 1000
        self.CACHE_TTL = 3600  # seconds
        
        # Output settings
        self.OUTPUT_DIR = "outputs"
        self.EXAMPLES_DIR = "examples"
        self.TEMP_DIR = "temp"
        
        # Logging settings
        self.LOG_LEVEL = "INFO"
        self.LOG_FILE = "tts_engine.log"
        self.LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        
        # Performance settings
        self.MAX_CONCURRENT_REQUESTS = 4
        self.REQUEST_TIMEOUT = 30  # seconds
        self.MEMORY_LIMIT = 4 * 1024 * 1024 * 1024  # 4GB
    
    def _load_from_env(self):
        """Load configuration from environment variables"""
        # Model settings
        self.DEFAULT_MODEL = os.getenv("TTS_DEFAULT_MODEL", self.DEFAULT_MODEL)
        self.DEFAULT_DEVICE = os.getenv("TTS_DEVICE", self.DEFAULT_DEVICE)
        self.USE_GPU = os.getenv("TTS_USE_GPU", "true").lower() == "true"
        
        # Audio settings
        self.SAMPLE_RATE = int(os.getenv("TTS_SAMPLE_RATE", self.SAMPLE_RATE))
        self.AUDIO_FORMAT = os.getenv("TTS_AUDIO_FORMAT", self.AUDIO_FORMAT)
        
        # Web interface settings
        self.WEB_HOST = os.getenv("TTS_WEB_HOST", self.WEB_HOST)
        self.WEB_PORT = int(os.getenv("TTS_WEB_PORT", self.WEB_PORT))
        self.WEB_SHARE = os.getenv("TTS_WEB_SHARE", "false").lower() == "true"
        
        # Output settings
        self.OUTPUT_DIR = os.getenv("TTS_OUTPUT_DIR", self.OUTPUT_DIR)
        self.EXAMPLES_DIR = os.getenv("TTS_EXAMPLES_DIR", self.EXAMPLES_DIR)
        
        # Logging settings
        self.LOG_LEVEL = os.getenv("TTS_LOG_LEVEL", self.LOG_LEVEL)
        self.LOG_FILE = os.getenv("TTS_LOG_FILE", self.LOG_FILE)
    
    def get_model_config(self) -> Dict:
        """Get model configuration"""
        return {
            "default_model": self.DEFAULT_MODEL,
            "device": self.DEFAULT_DEVICE,
            "use_gpu": self.USE_GPU,
            "sample_rate": self.SAMPLE_RATE
        }
    
    def get_audio_config(self) -> Dict:
        """Get audio configuration"""
        return {
            "sample_rate": self.SAMPLE_RATE,
            "target_length": self.TARGET_LENGTH,
            "audio_format": self.AUDIO_FORMAT
        }
    
    def get_voice_cloning_config(self) -> Dict:
        """Get voice cloning configuration"""
        return {
            "enabled": self.VOICE_CLONING_ENABLED,
            "min_duration": self.MIN_VOICE_DURATION,
            "max_duration": self.MAX_VOICE_DURATION,
            "embeddings_dir": self.VOICE_EMBEDDINGS_DIR
        }
    
    def get_emotion_config(self) -> Dict:
        """Get emotion control configuration"""
        return {
            "emotion_intensity_range": self.EMOTION_INTENSITY_RANGE,
            "style_intensity_range": self.STYLE_INTENSITY_RANGE,
            "default_emotion": self.DEFAULT_EMOTION,
            "default_style": self.DEFAULT_STYLE
        }
    
    def get_audio_processing_config(self) -> Dict:
        """Get audio processing configuration"""
        return {
            "enhancement_types": self.ENHANCEMENT_TYPES,
            "default_enhancement": self.DEFAULT_ENHANCEMENT,
            "noise_reduction_strength": self.NOISE_REDUCTION_STRENGTH,
            "reverb_types": self.REVERB_TYPES,
            "eq_types": self.EQ_TYPES
        }
    
    def get_web_config(self) -> Dict:
        """Get web interface configuration"""
        return {
            "host": self.WEB_HOST,
            "port": self.WEB_PORT,
            "share": self.WEB_SHARE,
            "max_file_size": self.MAX_FILE_SIZE
        }
    
    def get_batch_config(self) -> Dict:
        """Get batch processing configuration"""
        return {
            "max_batch_size": self.MAX_BATCH_SIZE,
            "timeout": self.BATCH_TIMEOUT
        }
    
    def get_cache_config(self) -> Dict:
        """Get cache configuration"""
        return {
            "enabled": self.CACHE_ENABLED,
            "max_size": self.MAX_CACHE_SIZE,
            "ttl": self.CACHE_TTL
        }
    
    def get_output_config(self) -> Dict:
        """Get output configuration"""
        return {
            "output_dir": self.OUTPUT_DIR,
            "examples_dir": self.EXAMPLES_DIR,
            "temp_dir": self.TEMP_DIR
        }
    
    def get_logging_config(self) -> Dict:
        """Get logging configuration"""
        return {
            "level": self.LOG_LEVEL,
            "file": self.LOG_FILE,
            "format": self.LOG_FORMAT
        }
    
    def get_performance_config(self) -> Dict:
        """Get performance configuration"""
        return {
            "max_concurrent_requests": self.MAX_CONCURRENT_REQUESTS,
            "request_timeout": self.REQUEST_TIMEOUT,
            "memory_limit": self.MEMORY_LIMIT
        }
    
    def create_directories(self):
        """Create necessary directories"""
        directories = [
            self.OUTPUT_DIR,
            self.EXAMPLES_DIR,
            self.TEMP_DIR,
            self.VOICE_EMBEDDINGS_DIR
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def validate_config(self) -> List[str]:
        """Validate configuration and return any errors"""
        errors = []
        
        # Validate sample rate
        if self.SAMPLE_RATE < 8000 or self.SAMPLE_RATE > 48000:
            errors.append(f"Invalid sample rate: {self.SAMPLE_RATE}. Must be between 8000 and 48000.")
        
        # Validate web port
        if self.WEB_PORT < 1024 or self.WEB_PORT > 65535:
            errors.append(f"Invalid web port: {self.WEB_PORT}. Must be between 1024 and 65535.")
        
        # Validate batch size
        if self.MAX_BATCH_SIZE < 1 or self.MAX_BATCH_SIZE > 1000:
            errors.append(f"Invalid max batch size: {self.MAX_BATCH_SIZE}. Must be between 1 and 1000.")
        
        # Validate voice duration limits
        if self.MIN_VOICE_DURATION >= self.MAX_VOICE_DURATION:
            errors.append("Min voice duration must be less than max voice duration.")
        
        # Validate intensity ranges
        if self.EMOTION_INTENSITY_RANGE[0] >= self.EMOTION_INTENSITY_RANGE[1]:
            errors.append("Invalid emotion intensity range.")
        
        if self.STYLE_INTENSITY_RANGE[0] >= self.STYLE_INTENSITY_RANGE[1]:
            errors.append("Invalid style intensity range.")
        
        return errors
    
    def save_config(self, filepath: str):
        """Save configuration to file"""
        import json
        
        config_dict = {
            "model": self.get_model_config(),
            "audio": self.get_audio_config(),
            "voice_cloning": self.get_voice_cloning_config(),
            "emotion": self.get_emotion_config(),
            "audio_processing": self.get_audio_processing_config(),
            "web": self.get_web_config(),
            "batch": self.get_batch_config(),
            "cache": self.get_cache_config(),
            "output": self.get_output_config(),
            "logging": self.get_logging_config(),
            "performance": self.get_performance_config()
        }
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def load_config(self, filepath: str):
        """Load configuration from file"""
        import json
        
        if not os.path.exists(filepath):
            return False
        
        try:
            with open(filepath, 'r') as f:
                config_dict = json.load(f)
            
            # Update configuration values
            if "model" in config_dict:
                model_config = config_dict["model"]
                self.DEFAULT_MODEL = model_config.get("default_model", self.DEFAULT_MODEL)
                self.DEFAULT_DEVICE = model_config.get("device", self.DEFAULT_DEVICE)
                self.USE_GPU = model_config.get("use_gpu", self.USE_GPU)
                self.SAMPLE_RATE = model_config.get("sample_rate", self.SAMPLE_RATE)
            
            if "audio" in config_dict:
                audio_config = config_dict["audio"]
                self.SAMPLE_RATE = audio_config.get("sample_rate", self.SAMPLE_RATE)
                self.AUDIO_FORMAT = audio_config.get("audio_format", self.AUDIO_FORMAT)
            
            if "web" in config_dict:
                web_config = config_dict["web"]
                self.WEB_HOST = web_config.get("host", self.WEB_HOST)
                self.WEB_PORT = web_config.get("port", self.WEB_PORT)
                self.WEB_SHARE = web_config.get("share", self.WEB_SHARE)
            
            return True
            
        except Exception as e:
            print(f"Error loading configuration: {e}")
            return False


# Global configuration instance
config = TTSConfig()

# Validate configuration on import
config_errors = config.validate_config()
if config_errors:
    print("Configuration validation errors:")
    for error in config_errors:
        print(f"  - {error}")
    print("Using default configuration values.")

# Create necessary directories
config.create_directories()
