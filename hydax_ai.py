#!/usr/bin/env python3
"""
Hydax AI - Complete Audio & Video Suite
Single file application with GUI and all functionality
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import webbrowser
import subprocess
import os
import time
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# Import all our modules
try:
    from tts_engine import HydaxTTS
    from voice_cloning import VoiceCloner
    from emotion_control import EmotionController
    from audio_processing import AudioProcessor
    from music_generator import MusicGenerator
    from music_styles import MusicStyleController
    from music_processing import MusicProcessor
    from video_generator import VideoGenerator
    from video_styles import VideoStyleController
    from video_processing import VideoProcessor
    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some modules not available: {e}")
    MODULES_AVAILABLE = False

class HydaxAI:
    """Complete Hydax AI Application with GUI"""
    
    def __init__(self):
        """Initialize the application"""
        self.root = tk.Tk()
        self.root.title("Hydax AI - Complete Audio & Video Suite")
        self.root.geometry("1000x800")
        self.root.configure(bg='#1e1e2e')
        
        # Initialize engines if available
        if MODULES_AVAILABLE:
            self.tts_engine = HydaxTTS()
            self.voice_cloner = VoiceCloner()
            self.emotion_controller = EmotionController()
            self.audio_processor = AudioProcessor()
            self.music_generator = MusicGenerator()
            self.music_style_controller = MusicStyleController()
            self.music_processor = MusicProcessor()
            self.video_generator = VideoGenerator()
            self.video_style_controller = VideoStyleController()
            self.video_processor = VideoProcessor()
        else:
            self.tts_engine = None
            self.voice_cloner = None
            self.emotion_controller = None
            self.audio_processor = None
            self.music_generator = None
            self.music_style_controller = None
            self.music_processor = None
            self.video_generator = None
            self.video_style_controller = None
            self.video_processor = None
        
        # Create output directories
        self.output_dir = Path("outputs")
        self.output_dir.mkdir(exist_ok=True)
        
        # Create interface
        self.create_header()
        self.create_tabs()
        self.create_status_bar()
        
        # Center window
        self.center_window()
    
    def center_window(self):
        """Center the window on screen"""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')
    
    def create_header(self):
        """Create the header section"""
        header_frame = tk.Frame(self.root, bg='#2d2d44', height=100)
        header_frame.pack(fill='x', padx=0, pady=0)
        header_frame.pack_propagate(False)
        
        # Title
        title = tk.Label(
            header_frame,
            text="🎤🎵🎬 Hydax AI",
            font=('Arial', 24, 'bold'),
            bg='#2d2d44',
            fg='#ffffff'
        )
        title.pack(pady=10)
        
        # Subtitle
        subtitle = tk.Label(
            header_frame,
            text="Complete Audio & Video Suite",
            font=('Arial', 12),
            bg='#2d2d44',
            fg='#a0a0b0'
        )
        subtitle.pack()
    
    def create_tabs(self):
        """Create the main tabbed interface"""
        # Create notebook for tabs
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TNotebook', background='#1e1e2e', borderwidth=0)
        style.configure('TNotebook.Tab', background='#2d2d44', foreground='#ffffff', padding=[20, 10])
        style.map('TNotebook.Tab', background=[('selected', '#667eea')], foreground=[('selected', '#ffffff')])
        
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create tabs
        self.create_quick_start_tab()
        self.create_tts_tab()
        self.create_music_tab()
        self.create_video_tab()
        self.create_multimedia_tab()
        self.create_settings_tab()
    
    def create_quick_start_tab(self):
        """Create the Quick Start tab"""
        frame = tk.Frame(self.notebook, bg='#1e1e2e')
        self.notebook.add(frame, text='🚀 Quick Start')
        
        # Welcome message
        welcome = tk.Label(
            frame,
            text="Welcome to Hydax AI!",
            font=('Arial', 18, 'bold'),
            bg='#1e1e2e',
            fg='#ffffff'
        )
        welcome.pack(pady=20)
        
        # Status check
        if MODULES_AVAILABLE:
            status_text = "✅ All modules loaded successfully!"
            status_color = '#4CAF50'
        else:
            status_text = "⚠️ Some modules not available. Check requirements.txt"
            status_color = '#FF9800'
        
        status = tk.Label(
            frame,
            text=status_text,
            font=('Arial', 12),
            bg='#1e1e2e',
            fg=status_color
        )
        status.pack(pady=10)
        
        # Quick actions
        actions_frame = tk.Frame(frame, bg='#1e1e2e')
        actions_frame.pack(pady=20)
        
        # Launch Web Interface button
        web_btn = tk.Button(
            actions_frame,
            text="🌐 Launch Web Interface",
            command=self.launch_web_interface,
            font=('Arial', 14, 'bold'),
            bg='#667eea',
            fg='#ffffff',
            activebackground='#5568d3',
            activeforeground='#ffffff',
            bd=0,
            padx=30,
            pady=15,
            cursor='hand2'
        )
        web_btn.pack(pady=10)
        
        # Quick TTS button
        tts_btn = tk.Button(
            actions_frame,
            text="🎤 Quick Text-to-Speech",
            command=lambda: self.notebook.select(1),
            font=('Arial', 12),
            bg='#4CAF50',
            fg='#ffffff',
            activebackground='#45a049',
            activeforeground='#ffffff',
            bd=0,
            padx=25,
            pady=12,
            cursor='hand2'
        )
        tts_btn.pack(pady=5)
        
        # Quick Music button
        music_btn = tk.Button(
            actions_frame,
            text="🎵 Quick Music Generation",
            command=lambda: self.notebook.select(2),
            font=('Arial', 12),
            bg='#2196F3',
            fg='#ffffff',
            activebackground='#1976D2',
            activeforeground='#ffffff',
            bd=0,
            padx=25,
            pady=12,
            cursor='hand2'
        )
        music_btn.pack(pady=5)
        
        # Quick Video button
        video_btn = tk.Button(
            actions_frame,
            text="🎬 Quick Video Generation",
            command=lambda: self.notebook.select(3),
            font=('Arial', 12),
            bg='#FF9800',
            fg='#ffffff',
            activebackground='#F57C00',
            activeforeground='#ffffff',
            bd=0,
            padx=25,
            pady=12,
            cursor='hand2'
        )
        video_btn.pack(pady=5)
        
        # Info text
        info = tk.Label(
            frame,
            text="💡 Tip: The web interface provides the most features and best experience.\nUse this desktop app for quick access and simple tasks.",
            font=('Arial', 10),
            bg='#1e1e2e',
            fg='#a0a0b0',
            justify='center'
        )
        info.pack(side='bottom', pady=20)
    
    def create_tts_tab(self):
        """Create the TTS tab"""
        frame = tk.Frame(self.notebook, bg='#1e1e2e')
        self.notebook.add(frame, text='🎤 TTS')
        
        # Title
        title = tk.Label(
            frame,
            text="Text-to-Speech",
            font=('Arial', 16, 'bold'),
            bg='#1e1e2e',
            fg='#ffffff'
        )
        title.pack(pady=10)
        
        # Text input
        text_label = tk.Label(frame, text="Enter Text:", font=('Arial', 11), bg='#1e1e2e', fg='#ffffff')
        text_label.pack(anchor='w', padx=20)
        
        self.tts_text = scrolledtext.ScrolledText(
            frame,
            height=8,
            font=('Arial', 10),
            bg='#2d2d44',
            fg='#ffffff',
            insertbackground='#ffffff',
            bd=0,
            padx=10,
            pady=10
        )
        self.tts_text.pack(padx=20, pady=5, fill='x')
        
        # Voice selection
        voice_frame = tk.Frame(frame, bg='#1e1e2e')
        voice_frame.pack(pady=10, fill='x', padx=20)
        
        tk.Label(voice_frame, text="Voice:", font=('Arial', 10), bg='#1e1e2e', fg='#ffffff').pack(side='left', padx=5)
        self.tts_voice = ttk.Combobox(
            voice_frame,
            values=["en-US-AriaNeural", "en-US-JennyNeural", "en-US-GuyNeural"],
            state='readonly',
            width=20
        )
        self.tts_voice.set("en-US-AriaNeural")
        self.tts_voice.pack(side='left', padx=5)
        
        tk.Label(voice_frame, text="Emotion:", font=('Arial', 10), bg='#1e1e2e', fg='#ffffff').pack(side='left', padx=5)
        self.tts_emotion = ttk.Combobox(
            voice_frame,
            values=["neutral", "happy", "sad", "angry", "fearful", "excited"],
            state='readonly',
            width=15
        )
        self.tts_emotion.set("neutral")
        self.tts_emotion.pack(side='left', padx=5)
        
        # Generate button
        gen_btn = tk.Button(
            frame,
            text="🎤 Generate Speech",
            command=self.generate_tts,
            font=('Arial', 12, 'bold'),
            bg='#4CAF50',
            fg='#ffffff',
            activebackground='#45a049',
            bd=0,
            padx=30,
            pady=10,
            cursor='hand2'
        )
        gen_btn.pack(pady=15)
        
        # Status
        self.tts_status = tk.Label(
            frame,
            text="Ready to generate speech",
            font=('Arial', 10),
            bg='#1e1e2e',
            fg='#a0a0b0'
        )
        self.tts_status.pack(pady=5)
    
    def create_music_tab(self):
        """Create the Music tab"""
        frame = tk.Frame(self.notebook, bg='#1e1e2e')
        self.notebook.add(frame, text='🎵 Music')
        
        # Title
        title = tk.Label(
            frame,
            text="Music Generation",
            font=('Arial', 16, 'bold'),
            bg='#1e1e2e',
            fg='#ffffff'
        )
        title.pack(pady=10)
        
        # Prompt input
        prompt_label = tk.Label(frame, text="Music Description:", font=('Arial', 11), bg='#1e1e2e', fg='#ffffff')
        prompt_label.pack(anchor='w', padx=20)
        
        self.music_prompt = scrolledtext.ScrolledText(
            frame,
            height=6,
            font=('Arial', 10),
            bg='#2d2d44',
            fg='#ffffff',
            insertbackground='#ffffff',
            bd=0,
            padx=10,
            pady=10
        )
        self.music_prompt.pack(padx=20, pady=5, fill='x')
        
        # Settings
        settings_frame = tk.Frame(frame, bg='#1e1e2e')
        settings_frame.pack(pady=10, fill='x', padx=20)
        
        tk.Label(settings_frame, text="Style:", font=('Arial', 10), bg='#1e1e2e', fg='#ffffff').pack(side='left', padx=5)
        self.music_style = ttk.Combobox(
            settings_frame,
            values=["pop", "rock", "jazz", "classical", "electronic", "ambient"],
            state='readonly',
            width=15
        )
        self.music_style.set("pop")
        self.music_style.pack(side='left', padx=5)
        
        tk.Label(settings_frame, text="Duration:", font=('Arial', 10), bg='#1e1e2e', fg='#ffffff').pack(side='left', padx=5)
        self.music_duration = ttk.Combobox(
            settings_frame,
            values=["60s", "120s", "180s", "240s", "300s"],
            state='readonly',
            width=10
        )
        self.music_duration.set("180s")
        self.music_duration.pack(side='left', padx=5)
        
        # Generate button
        gen_btn = tk.Button(
            frame,
            text="🎵 Generate Music",
            command=self.generate_music,
            font=('Arial', 12, 'bold'),
            bg='#2196F3',
            fg='#ffffff',
            activebackground='#1976D2',
            bd=0,
            padx=30,
            pady=10,
            cursor='hand2'
        )
        gen_btn.pack(pady=15)
        
        # Status
        self.music_status = tk.Label(
            frame,
            text="Ready to generate music",
            font=('Arial', 10),
            bg='#1e1e2e',
            fg='#a0a0b0'
        )
        self.music_status.pack(pady=5)
    
    def create_video_tab(self):
        """Create the Video tab"""
        frame = tk.Frame(self.notebook, bg='#1e1e2e')
        self.notebook.add(frame, text='🎬 Video')
        
        # Title
        title = tk.Label(
            frame,
            text="Video Generation",
            font=('Arial', 16, 'bold'),
            bg='#1e1e2e',
            fg='#ffffff'
        )
        title.pack(pady=10)
        
        # Prompt input
        prompt_label = tk.Label(frame, text="Video Description:", font=('Arial', 11), bg='#1e1e2e', fg='#ffffff')
        prompt_label.pack(anchor='w', padx=20)
        
        self.video_prompt = scrolledtext.ScrolledText(
            frame,
            height=6,
            font=('Arial', 10),
            bg='#2d2d44',
            fg='#ffffff',
            insertbackground='#ffffff',
            bd=0,
            padx=10,
            pady=10
        )
        self.video_prompt.pack(padx=20, pady=5, fill='x')
        
        # Settings
        settings_frame = tk.Frame(frame, bg='#1e1e2e')
        settings_frame.pack(pady=10, fill='x', padx=20)
        
        tk.Label(settings_frame, text="Style:", font=('Arial', 10), bg='#1e1e2e', fg='#ffffff').pack(side='left', padx=5)
        self.video_style = ttk.Combobox(
            settings_frame,
            values=["cinematic", "documentary", "animation", "corporate"],
            state='readonly',
            width=15
        )
        self.video_style.set("cinematic")
        self.video_style.pack(side='left', padx=5)
        
        tk.Label(settings_frame, text="Duration:", font=('Arial', 10), bg='#1e1e2e', fg='#ffffff').pack(side='left', padx=5)
        self.video_duration = ttk.Combobox(
            settings_frame,
            values=["60s", "120s", "180s", "240s", "300s"],
            state='readonly',
            width=10
        )
        self.video_duration.set("120s")
        self.video_duration.pack(side='left', padx=5)
        
        # Generate button
        gen_btn = tk.Button(
            frame,
            text="🎬 Generate Video",
            command=self.generate_video,
            font=('Arial', 12, 'bold'),
            bg='#FF9800',
            fg='#ffffff',
            activebackground='#F57C00',
            bd=0,
            padx=30,
            pady=10,
            cursor='hand2'
        )
        gen_btn.pack(pady=15)
        
        # Status
        self.video_status = tk.Label(
            frame,
            text="Ready to generate video",
            font=('Arial', 10),
            bg='#1e1e2e',
            fg='#a0a0b0'
        )
        self.video_status.pack(pady=5)
    
    def create_multimedia_tab(self):
        """Create the Multimedia tab"""
        frame = tk.Frame(self.notebook, bg='#1e1e2e')
        self.notebook.add(frame, text='🎭 Multimedia')
        
        # Title
        title = tk.Label(
            frame,
            text="Complete Multimedia Creation",
            font=('Arial', 16, 'bold'),
            bg='#1e1e2e',
            fg='#ffffff'
        )
        title.pack(pady=10)
        
        # Content description
        content_label = tk.Label(frame, text="Content Description:", font=('Arial', 11), bg='#1e1e2e', fg='#ffffff')
        content_label.pack(anchor='w', padx=20)
        
        self.content_description = scrolledtext.ScrolledText(
            frame,
            height=4,
            font=('Arial', 10),
            bg='#2d2d44',
            fg='#ffffff',
            insertbackground='#ffffff',
            bd=0,
            padx=10,
            pady=10
        )
        self.content_description.pack(padx=20, pady=5, fill='x')
        
        # Component selection
        components_frame = tk.Frame(frame, bg='#1e1e2e')
        components_frame.pack(pady=10, fill='x', padx=20)
        
        self.include_tts = tk.BooleanVar(value=True)
        self.include_music = tk.BooleanVar(value=True)
        self.include_video = tk.BooleanVar(value=True)
        
        tk.Checkbutton(components_frame, text="Include TTS", variable=self.include_tts, 
                      bg='#1e1e2e', fg='#ffffff', selectcolor='#2d2d44').pack(side='left', padx=10)
        tk.Checkbutton(components_frame, text="Include Music", variable=self.include_music,
                      bg='#1e1e2e', fg='#ffffff', selectcolor='#2d2d44').pack(side='left', padx=10)
        tk.Checkbutton(components_frame, text="Include Video", variable=self.include_video,
                      bg='#1e1e2e', fg='#ffffff', selectcolor='#2d2d44').pack(side='left', padx=10)
        
        # Duration
        duration_frame = tk.Frame(frame, bg='#1e1e2e')
        duration_frame.pack(pady=10, fill='x', padx=20)
        
        tk.Label(duration_frame, text="Duration:", font=('Arial', 10), bg='#1e1e2e', fg='#ffffff').pack(side='left', padx=5)
        self.multimedia_duration = ttk.Combobox(
            duration_frame,
            values=["60s", "120s", "180s", "240s", "300s"],
            state='readonly',
            width=10
        )
        self.multimedia_duration.set("180s")
        self.multimedia_duration.pack(side='left', padx=5)
        
        # Generate button
        gen_btn = tk.Button(
            frame,
            text="🎭 Create Multimedia Content",
            command=self.create_multimedia,
            font=('Arial', 12, 'bold'),
            bg='#9C27B0',
            fg='#ffffff',
            activebackground='#7B1FA2',
            bd=0,
            padx=30,
            pady=10,
            cursor='hand2'
        )
        gen_btn.pack(pady=15)
        
        # Status
        self.multimedia_status = tk.Label(
            frame,
            text="Ready to create multimedia content",
            font=('Arial', 10),
            bg='#1e1e2e',
            fg='#a0a0b0'
        )
        self.multimedia_status.pack(pady=5)
    
    def create_settings_tab(self):
        """Create the Settings tab"""
        frame = tk.Frame(self.notebook, bg='#1e1e2e')
        self.notebook.add(frame, text='⚙️ Settings')
        
        # Title
        title = tk.Label(
            frame,
            text="Settings & Configuration",
            font=('Arial', 16, 'bold'),
            bg='#1e1e2e',
            fg='#ffffff'
        )
        title.pack(pady=20)
        
        # Buttons frame
        buttons_frame = tk.Frame(frame, bg='#1e1e2e')
        buttons_frame.pack(pady=20)
        
        # Open output folder
        output_btn = tk.Button(
            buttons_frame,
            text="📁 Open Output Folder",
            command=self.open_output_folder,
            font=('Arial', 11),
            bg='#607D8B',
            fg='#ffffff',
            bd=0,
            padx=20,
            pady=10,
            cursor='hand2'
        )
        output_btn.pack(pady=5)
        
        # Run examples
        examples_btn = tk.Button(
            buttons_frame,
            text="📚 Run Examples",
            command=self.run_examples,
            font=('Arial', 11),
            bg='#9C27B0',
            fg='#ffffff',
            bd=0,
            padx=20,
            pady=10,
            cursor='hand2'
        )
        examples_btn.pack(pady=5)
        
        # View documentation
        docs_btn = tk.Button(
            buttons_frame,
            text="📖 View Documentation",
            command=self.view_docs,
            font=('Arial', 11),
            bg='#00BCD4',
            fg='#ffffff',
            bd=0,
            padx=20,
            pady=10,
            cursor='hand2'
        )
        docs_btn.pack(pady=5)
        
        # About
        about_text = tk.Label(
            frame,
            text="Hydax AI - Complete Audio & Video Suite\nVersion 1.0.0\n\n" +
                 "Features:\n• Text-to-Speech with voice cloning\n• AI Music Generation\n• AI Video Generation\n" +
                 "• Professional audio/video processing\n• Batch operations\n• Web interface integration",
            font=('Arial', 10),
            bg='#1e1e2e',
            fg='#a0a0b0',
            justify='center'
        )
        about_text.pack(side='bottom', pady=30)
    
    def create_status_bar(self):
        """Create the status bar"""
        self.status_bar = tk.Label(
            self.root,
            text="Ready",
            font=('Arial', 9),
            bg='#2d2d44',
            fg='#a0a0b0',
            anchor='w',
            padx=10
        )
        self.status_bar.pack(side='bottom', fill='x')
    
    # Action methods
    def launch_web_interface(self):
        """Launch the web interface"""
        try:
            self.status_bar.config(text="Launching web interface...")
            threading.Thread(target=lambda: subprocess.Popen(['python', 'integrated_launcher.py']), daemon=True).start()
            self.root.after(2000, lambda: webbrowser.open('http://localhost:7860'))
            messagebox.showinfo(
                "Hydax AI",
                "Web interface is launching!\n\nIt will open in your browser at:\nhttp://localhost:7860\n\nPlease wait a moment for the server to start."
            )
            self.status_bar.config(text="Web interface launched at http://localhost:7860")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to launch web interface:\n{e}")
            self.status_bar.config(text="Error launching web interface")
    
    def generate_tts(self):
        """Generate TTS"""
        if not self.tts_engine:
            messagebox.showerror("Error", "TTS engine not available. Check requirements.txt")
            return
            
        text = self.tts_text.get("1.0", "end-1c").strip()
        if not text:
            messagebox.showwarning("Warning", "Please enter some text to convert to speech.")
            return
        
        self.tts_status.config(text="Generating speech...")
        self.status_bar.config(text="Processing TTS request...")
        
        def run_tts():
            try:
                audio = self.tts_engine.synthesize(text=text, voice=self.tts_voice.get())
                if len(audio) > 0:
                    output_file = f"tts_output_{int(time.time())}.wav"
                    self.tts_engine.save_audio(audio, output_file)
                    self.root.after(0, lambda: self.tts_status.config(text=f"✓ Generated: {output_file}"))
                    self.root.after(0, lambda: self.status_bar.config(text=f"TTS completed: {output_file}"))
                    self.root.after(0, lambda: messagebox.showinfo("Success", f"Speech generated successfully!\n\nSaved to: {output_file}"))
                else:
                    self.root.after(0, lambda: self.tts_status.config(text="✗ Generation failed"))
                    self.root.after(0, lambda: self.status_bar.config(text="TTS generation failed"))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"TTS generation failed:\n{e}"))
                self.root.after(0, lambda: self.tts_status.config(text="✗ Error occurred"))
                self.root.after(0, lambda: self.status_bar.config(text="TTS error"))
        
        threading.Thread(target=run_tts, daemon=True).start()
    
    def generate_music(self):
        """Generate music"""
        if not self.music_generator:
            messagebox.showerror("Error", "Music generator not available. Check requirements.txt")
            return
            
        prompt = self.music_prompt.get("1.0", "end-1c").strip()
        if not prompt:
            messagebox.showwarning("Warning", "Please enter a music description.")
            return
        
        self.music_status.config(text="Generating music...")
        self.status_bar.config(text="Processing music generation request...")
        
        def run_music():
            try:
                duration = int(self.music_duration.get().replace('s', ''))
                music = self.music_generator.generate_music(
                    prompt=prompt,
                    duration=duration,
                    style=self.music_style.get()
                )
                if len(music) > 0:
                    output_file = f"music_output_{int(time.time())}.wav"
                    self.music_generator.save_music(music, output_file)
                    self.root.after(0, lambda: self.music_status.config(text=f"✓ Generated: {output_file}"))
                    self.root.after(0, lambda: self.status_bar.config(text=f"Music completed: {output_file}"))
                    self.root.after(0, lambda: messagebox.showinfo("Success", f"Music generated successfully!\n\nSaved to: {output_file}"))
                else:
                    self.root.after(0, lambda: self.music_status.config(text="✗ Generation failed"))
                    self.root.after(0, lambda: self.status_bar.config(text="Music generation failed"))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Music generation failed:\n{e}"))
                self.root.after(0, lambda: self.music_status.config(text="✗ Error occurred"))
                self.root.after(0, lambda: self.status_bar.config(text="Music error"))
        
        threading.Thread(target=run_music, daemon=True).start()
    
    def generate_video(self):
        """Generate video"""
        if not self.video_generator:
            messagebox.showerror("Error", "Video generator not available. Check requirements.txt")
            return
            
        prompt = self.video_prompt.get("1.0", "end-1c").strip()
        if not prompt:
            messagebox.showwarning("Warning", "Please enter a video description.")
            return
        
        self.video_status.config(text="Generating video...")
        self.status_bar.config(text="Processing video generation request...")
        
        def run_video():
            try:
                duration = int(self.video_duration.get().replace('s', ''))
                video_path = self.video_generator.generate_video(
                    prompt=prompt,
                    duration=duration,
                    style=self.video_style.get(),
                    scene_type="abstract"
                )
                if video_path and os.path.exists(video_path):
                    self.root.after(0, lambda: self.video_status.config(text=f"✓ Generated: {video_path}"))
                    self.root.after(0, lambda: self.status_bar.config(text=f"Video completed: {video_path}"))
                    self.root.after(0, lambda: messagebox.showinfo("Success", f"Video generated successfully!\n\nSaved to: {video_path}"))
                else:
                    self.root.after(0, lambda: self.video_status.config(text="✗ Generation failed"))
                    self.root.after(0, lambda: self.status_bar.config(text="Video generation failed"))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Video generation failed:\n{e}"))
                self.root.after(0, lambda: self.video_status.config(text="✗ Error occurred"))
                self.root.after(0, lambda: self.status_bar.config(text="Video error"))
        
        threading.Thread(target=run_video, daemon=True).start()
    
    def create_multimedia(self):
        """Create complete multimedia content"""
        description = self.content_description.get("1.0", "end-1c").strip()
        if not description:
            messagebox.showwarning("Warning", "Please enter a content description.")
            return
        
        self.multimedia_status.config(text="Creating multimedia content...")
        self.status_bar.config(text="Processing multimedia creation...")
        
        def run_multimedia():
            try:
                results = []
                duration = int(self.multimedia_duration.get().replace('s', ''))
                
                # Generate TTS if requested
                if self.include_tts.get() and self.tts_engine:
                    try:
                        audio = self.tts_engine.synthesize(text=description, voice="en-US-AriaNeural")
                        if len(audio) > 0:
                            output_file = f"multimedia_tts_{int(time.time())}.wav"
                            self.tts_engine.save_audio(audio, output_file)
                            results.append(f"TTS: {output_file}")
                    except Exception as e:
                        results.append(f"TTS Error: {e}")
                
                # Generate Music if requested
                if self.include_music.get() and self.music_generator:
                    try:
                        music = self.music_generator.generate_music(
                            prompt=f"Background music for: {description}",
                            duration=duration,
                            style="ambient"
                        )
                        if len(music) > 0:
                            output_file = f"multimedia_music_{int(time.time())}.wav"
                            self.music_generator.save_music(music, output_file)
                            results.append(f"Music: {output_file}")
                    except Exception as e:
                        results.append(f"Music Error: {e}")
                
                # Generate Video if requested
                if self.include_video.get() and self.video_generator:
                    try:
                        video_path = self.video_generator.generate_video(
                            prompt=description,
                            duration=duration,
                            style="cinematic",
                            scene_type="abstract"
                        )
                        if video_path and os.path.exists(video_path):
                            results.append(f"Video: {video_path}")
                    except Exception as e:
                        results.append(f"Video Error: {e}")
                
                # Show results
                if results:
                    result_text = "\n".join(results)
                    self.root.after(0, lambda: self.multimedia_status.config(text="✓ Multimedia creation completed"))
                    self.root.after(0, lambda: self.status_bar.config(text="Multimedia creation completed"))
                    self.root.after(0, lambda: messagebox.showinfo("Success", f"Multimedia content created!\n\n{result_text}"))
                else:
                    self.root.after(0, lambda: self.multimedia_status.config(text="✗ No content generated"))
                    self.root.after(0, lambda: self.status_bar.config(text="Multimedia creation failed"))
                    
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Multimedia creation failed:\n{e}"))
                self.root.after(0, lambda: self.multimedia_status.config(text="✗ Error occurred"))
                self.root.after(0, lambda: self.status_bar.config(text="Multimedia error"))
        
        threading.Thread(target=run_multimedia, daemon=True).start()
    
    def open_output_folder(self):
        """Open the output folder"""
        try:
            if os.path.exists("outputs"):
                os.startfile("outputs")
            else:
                messagebox.showinfo("Info", "No output folder found yet.\nGenerate some content first!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open output folder:\n{e}")
    
    def run_examples(self):
        """Run examples"""
        try:
            subprocess.Popen(['python', 'integrated_launcher.py', '--examples'])
            messagebox.showinfo(
                "Examples",
                "Examples are running!\n\nCheck the console window for progress."
            )
        except Exception as e:
            messagebox.showerror("Error", f"Failed to run examples:\n{e}")
    
    def view_docs(self):
        """View documentation"""
        try:
            if os.path.exists('README.md'):
                os.startfile('README.md')
            else:
                messagebox.showinfo("Info", "README.md not found.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open documentation:\n{e}")
    
    def run(self):
        """Start the application"""
        self.root.mainloop()


# Main execution
if __name__ == "__main__":
    print("🎤🎵🎬 Launching Hydax AI - Complete Audio & Video Suite...")
    app = HydaxAI()
    app.run()
