"""
Advanced Video Post-Processing and Effects for Hydax AI Video Generator
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings("ignore")

try:
    import moviepy.editor as mp
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False
    print("Warning: moviepy not available. Install with: pip install moviepy")

try:
    from PIL import Image, ImageFilter, ImageEnhance
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Warning: PIL not available. Install with: pip install Pillow")

class VideoProcessor:
    """
    Advanced video post-processing and effects engine
    """
    
    def __init__(self, fps: int = 30):
        """
        Initialize the video processor
        
        Args:
            fps: Default frames per second
        """
        self.fps = fps
        self.effect_presets = self._create_effect_presets()
        self.color_presets = self._create_color_presets()
        self.transition_presets = self._create_transition_presets()
        
    def _create_effect_presets(self) -> Dict:
        """Create effect presets for different video styles"""
        return {
            'cinematic': {
                'color_grading': {'warm': True, 'contrast': 1.2, 'saturation': 1.1},
                'stabilization': {'enabled': True, 'strength': 0.8},
                'film_grain': {'intensity': 0.3, 'size': 1.0},
                'vignette': {'intensity': 0.4, 'size': 0.8},
                'sharpening': {'intensity': 0.2}
            },
            'documentary': {
                'color_grading': {'natural': True, 'contrast': 1.1, 'saturation': 1.0},
                'stabilization': {'enabled': True, 'strength': 1.0},
                'noise_reduction': {'enabled': True, 'strength': 0.5},
                'sharpening': {'intensity': 0.3}
            },
            'animation': {
                'color_grading': {'vibrant': True, 'contrast': 1.3, 'saturation': 1.4},
                'cartoon_filter': {'enabled': True, 'intensity': 0.7},
                'motion_blur': {'enabled': True, 'strength': 0.3},
                'sharpening': {'intensity': 0.4}
            },
            'corporate': {
                'color_grading': {'clean': True, 'contrast': 1.1, 'saturation': 1.0},
                'stabilization': {'enabled': True, 'strength': 0.9},
                'sharpening': {'intensity': 0.3},
                'clean_graphics': {'enabled': True}
            },
            'artistic': {
                'color_grading': {'creative': True, 'contrast': 1.4, 'saturation': 1.3},
                'artistic_filters': {'enabled': True, 'type': 'painterly'},
                'texture_overlay': {'enabled': True, 'intensity': 0.5},
                'color_shift': {'enabled': True, 'hue_shift': 30}
            },
            'social_media': {
                'color_grading': {'trendy': True, 'contrast': 1.2, 'saturation': 1.3},
                'vibrant_colors': {'enabled': True, 'intensity': 0.8},
                'fast_pacing': {'enabled': True, 'speed_multiplier': 1.2},
                'trendy_filters': {'enabled': True, 'type': 'viral'}
            }
        }
    
    def _create_color_presets(self) -> Dict:
        """Create color grading presets"""
        return {
            'warm': {
                'red': 1.1, 'green': 1.0, 'blue': 0.9,
                'temperature': 5500, 'tint': 0.1
            },
            'cool': {
                'red': 0.9, 'green': 1.0, 'blue': 1.1,
                'temperature': 6500, 'tint': -0.1
            },
            'vintage': {
                'red': 1.2, 'green': 1.1, 'blue': 0.8,
                'temperature': 4000, 'tint': 0.2
            },
            'dramatic': {
                'red': 1.3, 'green': 0.8, 'blue': 0.7,
                'temperature': 3500, 'tint': 0.3
            },
            'natural': {
                'red': 1.0, 'green': 1.0, 'blue': 1.0,
                'temperature': 5500, 'tint': 0.0
            },
            'vibrant': {
                'red': 1.2, 'green': 1.2, 'blue': 1.1,
                'temperature': 6000, 'tint': 0.1
            }
        }
    
    def _create_transition_presets(self) -> Dict:
        """Create transition presets"""
        return {
            'fade': {
                'type': 'fade',
                'duration': 1.0,
                'easing': 'ease_in_out'
            },
            'slide': {
                'type': 'slide',
                'duration': 0.8,
                'direction': 'left',
                'easing': 'ease_out'
            },
            'zoom': {
                'type': 'zoom',
                'duration': 0.6,
                'scale': 1.5,
                'easing': 'ease_in'
            },
            'dissolve': {
                'type': 'dissolve',
                'duration': 1.5,
                'easing': 'linear'
            }
        }
    
    def process_video(self, 
                     video_path: str,
                     style: str = "cinematic",
                     custom_effects: Optional[Dict] = None,
                     output_path: Optional[str] = None) -> str:
        """
        Process video with effects and enhancements
        
        Args:
            video_path: Path to input video
            style: Processing style preset
            custom_effects: Custom effect parameters
            output_path: Output video path
        
        Returns:
            str: Path to processed video
        """
        try:
            if not os.path.exists(video_path):
                print(f"Video file not found: {video_path}")
                return None
            
            # Get effect parameters
            if custom_effects:
                effects = custom_effects
            else:
                effects = self.effect_presets.get(style, self.effect_presets['cinematic'])
            
            # Set output path
            if output_path is None:
                base_name = os.path.splitext(video_path)[0]
                output_path = f"{base_name}_processed_{style}.mp4"
            
            print(f"Processing video with {style} style...")
            
            # Process video
            if MOVIEPY_AVAILABLE:
                processed_path = self._process_with_moviepy(video_path, effects, output_path)
            else:
                processed_path = self._process_with_opencv(video_path, effects, output_path)
            
            return processed_path
            
        except Exception as e:
            print(f"Video processing failed: {e}")
            return None
    
    def _process_with_moviepy(self, video_path: str, effects: Dict, output_path: str) -> str:
        """Process video using MoviePy"""
        try:
            # Load video
            video = mp.VideoFileClip(video_path)
            
            # Apply effects
            processed_video = video
            
            # Color grading
            if 'color_grading' in effects:
                processed_video = self._apply_color_grading_moviepy(processed_video, effects['color_grading'])
            
            # Stabilization
            if 'stabilization' in effects and effects['stabilization']['enabled']:
                processed_video = self._apply_stabilization_moviepy(processed_video, effects['stabilization'])
            
            # Film grain
            if 'film_grain' in effects:
                processed_video = self._apply_film_grain_moviepy(processed_video, effects['film_grain'])
            
            # Vignette
            if 'vignette' in effects:
                processed_video = self._apply_vignette_moviepy(processed_video, effects['vignette'])
            
            # Sharpening
            if 'sharpening' in effects:
                processed_video = self._apply_sharpening_moviepy(processed_video, effects['sharpening'])
            
            # Save processed video
            processed_video.write_videofile(output_path, fps=self.fps)
            
            # Clean up
            video.close()
            processed_video.close()
            
            return output_path
            
        except Exception as e:
            print(f"MoviePy processing failed: {e}")
            return None
    
    def _process_with_opencv(self, video_path: str, effects: Dict, output_path: str) -> str:
        """Process video using OpenCV"""
        try:
            # Open video
            cap = cv2.VideoCapture(video_path)
            
            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            # Process frames
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Apply effects to frame
                processed_frame = self._apply_effects_to_frame(frame, effects)
                
                # Write processed frame
                out.write(processed_frame)
            
            # Clean up
            cap.release()
            out.release()
            
            return output_path
            
        except Exception as e:
            print(f"OpenCV processing failed: {e}")
            return None
    
    def _apply_effects_to_frame(self, frame: np.ndarray, effects: Dict) -> np.ndarray:
        """Apply effects to a single frame"""
        try:
            processed_frame = frame.copy()
            
            # Color grading
            if 'color_grading' in effects:
                processed_frame = self._apply_color_grading_frame(processed_frame, effects['color_grading'])
            
            # Film grain
            if 'film_grain' in effects:
                processed_frame = self._apply_film_grain_frame(processed_frame, effects['film_grain'])
            
            # Vignette
            if 'vignette' in effects:
                processed_frame = self._apply_vignette_frame(processed_frame, effects['vignette'])
            
            # Sharpening
            if 'sharpening' in effects:
                processed_frame = self._apply_sharpening_frame(processed_frame, effects['sharpening'])
            
            # Cartoon filter
            if 'cartoon_filter' in effects and effects['cartoon_filter']['enabled']:
                processed_frame = self._apply_cartoon_filter_frame(processed_frame, effects['cartoon_filter'])
            
            return processed_frame
            
        except Exception as e:
            print(f"Frame effects application failed: {e}")
            return frame
    
    def _apply_color_grading_frame(self, frame: np.ndarray, color_params: Dict) -> np.ndarray:
        """Apply color grading to frame"""
        try:
            # Convert to float for processing
            frame_float = frame.astype(np.float32) / 255.0
            
            # Apply color adjustments
            if 'warm' in color_params and color_params['warm']:
                frame_float[:, :, 0] *= 1.1  # Increase red
                frame_float[:, :, 2] *= 0.9  # Decrease blue
            elif 'cool' in color_params and color_params['cool']:
                frame_float[:, :, 0] *= 0.9  # Decrease red
                frame_float[:, :, 2] *= 1.1  # Increase blue
            elif 'vintage' in color_params and color_params['vintage']:
                frame_float[:, :, 0] *= 1.2  # Increase red
                frame_float[:, :, 1] *= 1.1  # Increase green
                frame_float[:, :, 2] *= 0.8  # Decrease blue
            
            # Apply contrast
            if 'contrast' in color_params:
                contrast = color_params['contrast']
                frame_float = (frame_float - 0.5) * contrast + 0.5
            
            # Apply saturation
            if 'saturation' in color_params:
                saturation = color_params['saturation']
                gray = np.dot(frame_float, [0.299, 0.587, 0.114])
                gray = np.stack([gray, gray, gray], axis=2)
                frame_float = gray + saturation * (frame_float - gray)
            
            # Clamp values and convert back
            frame_float = np.clip(frame_float, 0, 1)
            return (frame_float * 255).astype(np.uint8)
            
        except Exception as e:
            print(f"Color grading failed: {e}")
            return frame
    
    def _apply_film_grain_frame(self, frame: np.ndarray, grain_params: Dict) -> np.ndarray:
        """Apply film grain effect to frame"""
        try:
            intensity = grain_params.get('intensity', 0.3)
            size = grain_params.get('size', 1.0)
            
            # Generate noise
            noise = np.random.normal(0, intensity * 25, frame.shape).astype(np.int16)
            
            # Apply noise
            frame_with_noise = frame.astype(np.int16) + noise
            
            # Clamp values
            return np.clip(frame_with_noise, 0, 255).astype(np.uint8)
            
        except Exception as e:
            print(f"Film grain application failed: {e}")
            return frame
    
    def _apply_vignette_frame(self, frame: np.ndarray, vignette_params: Dict) -> np.ndarray:
        """Apply vignette effect to frame"""
        try:
            intensity = vignette_params.get('intensity', 0.4)
            size = vignette_params.get('size', 0.8)
            
            height, width = frame.shape[:2]
            
            # Create vignette mask
            center_x, center_y = width // 2, height // 2
            max_distance = np.sqrt(center_x**2 + center_y**2)
            
            y, x = np.ogrid[:height, :width]
            distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            
            # Normalize distance
            distance = distance / (max_distance * size)
            
            # Create vignette effect
            vignette = 1 - (distance * intensity)
            vignette = np.clip(vignette, 0, 1)
            
            # Apply vignette
            frame_float = frame.astype(np.float32)
            vignette_3d = np.stack([vignette, vignette, vignette], axis=2)
            frame_vignetted = frame_float * vignette_3d
            
            return np.clip(frame_vignetted, 0, 255).astype(np.uint8)
            
        except Exception as e:
            print(f"Vignette application failed: {e}")
            return frame
    
    def _apply_sharpening_frame(self, frame: np.ndarray, sharpening_params: Dict) -> np.ndarray:
        """Apply sharpening to frame"""
        try:
            intensity = sharpening_params.get('intensity', 0.2)
            
            # Create sharpening kernel
            kernel = np.array([[-1, -1, -1],
                              [-1,  9, -1],
                              [-1, -1, -1]]) * intensity
            
            # Apply convolution
            sharpened = cv2.filter2D(frame, -1, kernel)
            
            # Blend with original
            result = cv2.addWeighted(frame, 1 - intensity, sharpened, intensity, 0)
            
            return result
            
        except Exception as e:
            print(f"Sharpening failed: {e}")
            return frame
    
    def _apply_cartoon_filter_frame(self, frame: np.ndarray, cartoon_params: Dict) -> np.ndarray:
        """Apply cartoon filter to frame"""
        try:
            intensity = cartoon_params.get('intensity', 0.7)
            
            # Reduce colors
            frame_float = frame.astype(np.float32)
            frame_quantized = (frame_float // (32 * intensity)) * (32 * intensity)
            
            # Apply edge detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            
            # Combine quantized image with edges
            cartoon = cv2.bitwise_and(frame_quantized.astype(np.uint8), 255 - edges)
            
            return cartoon
            
        except Exception as e:
            print(f"Cartoon filter failed: {e}")
            return frame
    
    def add_transitions(self, 
                       video_clips: List[str], 
                       transition_type: str = "fade",
                       transition_duration: float = 1.0) -> str:
        """
        Add transitions between video clips
        
        Args:
            video_clips: List of video file paths
            transition_type: Type of transition
            transition_duration: Duration of transition
        
        Returns:
            str: Path to final video with transitions
        """
        try:
            if not MOVIEPY_AVAILABLE:
                print("MoviePy not available for transitions")
                return None
            
            if len(video_clips) < 2:
                print("Need at least 2 video clips for transitions")
                return None
            
            # Load video clips
            clips = [mp.VideoFileClip(clip) for clip in video_clips]
            
            # Create transitions
            if transition_type == "fade":
                final_clip = mp.concatenate_videoclips(clips, method="compose")
            elif transition_type == "dissolve":
                final_clip = mp.concatenate_videoclips(clips, method="compose")
            else:
                final_clip = mp.concatenate_videoclips(clips, method="compose")
            
            # Save final video
            output_path = "transitions_output.mp4"
            final_clip.write_videofile(output_path, fps=self.fps)
            
            # Clean up
            for clip in clips:
                clip.close()
            final_clip.close()
            
            return output_path
            
        except Exception as e:
            print(f"Transition addition failed: {e}")
            return None
    
    def add_text_overlay(self, 
                        video_path: str, 
                        text: str,
                        position: str = "bottom_center",
                        font_size: int = 40,
                        color: Tuple[int, int, int] = (255, 255, 255),
                        duration: Optional[float] = None) -> str:
        """
        Add text overlay to video
        
        Args:
            video_path: Path to input video
            text: Text to overlay
            position: Position of text
            font_size: Font size
            color: Text color (BGR)
            duration: Duration to show text (None for entire video)
        
        Returns:
            str: Path to video with text overlay
        """
        try:
            if not MOVIEPY_AVAILABLE:
                print("MoviePy not available for text overlay")
                return None
            
            # Load video
            video = mp.VideoFileClip(video_path)
            
            # Create text clip
            text_clip = mp.TextClip(text, 
                                  fontsize=font_size, 
                                  color=color, 
                                  font='Arial-Bold')
            
            # Set duration
            if duration is None:
                text_clip = text_clip.set_duration(video.duration)
            else:
                text_clip = text_clip.set_duration(duration)
            
            # Set position
            if position == "bottom_center":
                text_clip = text_clip.set_position(('center', 'bottom'))
            elif position == "top_center":
                text_clip = text_clip.set_position(('center', 'top'))
            elif position == "center":
                text_clip = text_clip.set_position('center')
            
            # Composite video with text
            final_video = mp.CompositeVideoClip([video, text_clip])
            
            # Save video
            output_path = video_path.replace('.mp4', '_with_text.mp4')
            final_video.write_videofile(output_path, fps=self.fps)
            
            # Clean up
            video.close()
            text_clip.close()
            final_video.close()
            
            return output_path
            
        except Exception as e:
            print(f"Text overlay failed: {e}")
            return None
    
    def add_audio_track(self, 
                       video_path: str, 
                       audio_path: str,
                       volume: float = 1.0,
                       start_time: float = 0.0) -> str:
        """
        Add audio track to video
        
        Args:
            video_path: Path to input video
            audio_path: Path to audio file
            volume: Audio volume (0.0 to 2.0)
            start_time: Start time for audio
        
        Returns:
            str: Path to video with audio
        """
        try:
            if not MOVIEPY_AVAILABLE:
                print("MoviePy not available for audio addition")
                return None
            
            # Load video and audio
            video = mp.VideoFileClip(video_path)
            audio = mp.AudioFileClip(audio_path)
            
            # Adjust audio volume and start time
            audio = audio.set_start(start_time).volumex(volume)
            
            # Set audio duration to match video
            if audio.duration > video.duration:
                audio = audio.subclip(0, video.duration)
            
            # Add audio to video
            final_video = video.set_audio(audio)
            
            # Save video
            output_path = video_path.replace('.mp4', '_with_audio.mp4')
            final_video.write_videofile(output_path, fps=self.fps)
            
            # Clean up
            video.close()
            audio.close()
            final_video.close()
            
            return output_path
            
        except Exception as e:
            print(f"Audio addition failed: {e}")
            return None
    
    def create_slideshow(self, 
                        image_paths: List[str], 
                        duration_per_image: float = 3.0,
                        transition_type: str = "fade",
                        background_music: Optional[str] = None) -> str:
        """
        Create slideshow from images
        
        Args:
            image_paths: List of image file paths
            duration_per_image: Duration to show each image
            transition_type: Type of transition between images
            background_music: Path to background music file
        
        Returns:
            str: Path to slideshow video
        """
        try:
            if not MOVIEPY_AVAILABLE:
                print("MoviePy not available for slideshow creation")
                return None
            
            # Create image clips
            image_clips = []
            for image_path in image_paths:
                clip = mp.ImageClip(image_path, duration=duration_per_image)
                image_clips.append(clip)
            
            # Concatenate clips
            slideshow = mp.concatenate_videoclips(image_clips, method="compose")
            
            # Add background music if provided
            if background_music and os.path.exists(background_music):
                audio = mp.AudioFileClip(background_music)
                if audio.duration > slideshow.duration:
                    audio = audio.subclip(0, slideshow.duration)
                slideshow = slideshow.set_audio(audio)
            
            # Save slideshow
            output_path = "slideshow_output.mp4"
            slideshow.write_videofile(output_path, fps=self.fps)
            
            # Clean up
            for clip in image_clips:
                clip.close()
            slideshow.close()
            if background_music:
                audio.close()
            
            return output_path
            
        except Exception as e:
            print(f"Slideshow creation failed: {e}")
            return None
    
    def get_video_info(self, video_path: str) -> Dict:
        """Get information about a video file"""
        try:
            if not os.path.exists(video_path):
                return {"error": "Video file not found"}
            
            if MOVIEPY_AVAILABLE:
                video = mp.VideoFileClip(video_path)
                info = {
                    "duration": video.duration,
                    "fps": video.fps,
                    "size": video.size,
                    "width": video.w,
                    "height": video.h,
                    "has_audio": video.audio is not None
                }
                video.close()
                return info
            else:
                # Fallback using OpenCV
                cap = cv2.VideoCapture(video_path)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
                
                return {
                    "duration": frame_count / fps if fps > 0 else 0,
                    "fps": fps,
                    "size": (width, height),
                    "width": width,
                    "height": height,
                    "frame_count": frame_count
                }
                
        except Exception as e:
            return {"error": str(e)}
    
    def get_available_effects(self) -> List[str]:
        """Get list of available effect presets"""
        return list(self.effect_presets.keys())
    
    def get_available_color_presets(self) -> List[str]:
        """Get list of available color presets"""
        return list(self.color_presets.keys())
    
    def get_available_transitions(self) -> List[str]:
        """Get list of available transitions"""
        return list(self.transition_presets.keys())


# Example usage
if __name__ == "__main__":
    # Initialize video processor
    processor = VideoProcessor()
    
    # Test with sample video (would need actual video file)
    # video_path = "sample_video.mp4"
    # processed_path = processor.process_video(video_path, style="cinematic")
    # print(f"Processed video: {processed_path}")
    
    print("Video processing module initialized successfully!")
