from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.widget import Widget
from kivy.graphics import Line, Color
from kivy.clock import Clock
from threading import Thread, Event
import numpy as np
import time
import logging
from pydub import AudioSegment
import os
import sys
import win32file
import win32pipe
import win32con  # For GENERIC_READ

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class WaveformWidget(Widget):
    def update_waveform(self, audio_chunk):
        """Update the waveform with new audio data."""
        self.canvas.clear()
        with self.canvas:
            Color(0, 1, 0)  # Green color for the waveform line

            # Normalize audio data for drawing
            normalized_data = (
                (audio_chunk - np.mean(audio_chunk)) / np.max(np.abs(audio_chunk))
            ) * (self.height / 2) + (self.height / 2)

            # Scale the x-axis to fit the widget's width
            step = max(1, len(audio_chunk) // self.width)

            # Generate points to fit the widget's size dynamically
            points = []
            for i in range(0, len(normalized_data), step):
                x = i * (self.width / len(normalized_data))
                y = normalized_data[i]
                points.extend([x, y])

            # Draw the waveform
            Line(points=points, width=1.5)

class AudioClientApp(App):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pipe_connected = False
        self.pipe = None
        self.stop_event = Event()
        self.buffered_audio = AudioSegment.silent(duration=0)  # Start with empty audio buffer
        self.output_file = "./tmp/audio_files/output.wav"
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)

    def build(self):
        layout = BoxLayout(orientation='vertical')
        self.start_button = Button(text='Start Listening', on_press=self.start_audio_thread)
        layout.add_widget(self.start_button)
        self.stop_button = Button(text='Stop Listening', on_press=self.stop_audio_thread)
        layout.add_widget(self.stop_button)
        self.pipe_name_input = TextInput(text='Enter Pipe Name Here', multiline=False)
        layout.add_widget(self.pipe_name_input)
        self.waveform_widget = WaveformWidget(size_hint=(1, 0.8))
        layout.add_widget(self.waveform_widget)
        return layout

    def start_audio_thread(self, instance):
        self.stop_event.clear()
        self.audio_thread = Thread(target=self.audio_buffering, daemon=True)
        self.audio_thread.start()

        # Schedule periodic export to MP3 every 5 seconds
        Clock.schedule_interval(self.export_to_mp3, 5)

    def stop_audio_thread(self, instance=None):
        self.stop_event.set()
        logging.info("Stopped audio listening.")

    def connect_to_pipe(self):
        pipe_name = self.pipe_name_input.text.strip()
        logging.info(f"Attempting to connect to pipe: {pipe_name}")
        retries = 3  # Number of retries
        for attempt in range(retries):
            try:
                self.pipe = win32file.CreateFile(
                    r'\\.\pipe\\' + pipe_name,
                    win32con.GENERIC_READ | win32con.GENERIC_WRITE,
                    0,  # No sharing
                    None,  # Default security attributes
                    win32con.OPEN_EXISTING,
                    0,  # Default flags
                    None  # No template
                )
                self.pipe_connected = True
                logging.info("Connected to the pipe.")
                return True
            except Exception as e:
                logging.error(f"Attempt {attempt + 1}: Failed to connect to pipe: {e}")
                time.sleep(1)  # Wait before retrying
        return False

    def audio_buffering(self):
        sample_width = 2  # For 32-bit float audio
        channels = 2  # Stereo audio
        expected_length = sample_width * channels  # Calculate expected length

        while not self.stop_event.is_set():
            if not self.connect_to_pipe():
                time.sleep(2)
                continue

            try:
                while not self.stop_event.is_set():
                    audio_chunk = self.read_audio_chunk()
                    if audio_chunk is not None and len(audio_chunk) > 0:
                        logging.info(f"Processing audio chunk with {len(audio_chunk)} samples.")

                        if len(audio_chunk) % expected_length != 0:
                            logging.warning(
                                f"Audio chunk length {len(audio_chunk)} is not a multiple of {expected_length}.")
                            # Padding logic or error handling can go here
                            padding_length = expected_length - (len(audio_chunk) % expected_length)
                            audio_chunk = np.concatenate(
                                (audio_chunk, np.zeros(padding_length, dtype=audio_chunk.dtype)))

                        # Only proceed if the chunk has valid audio samples
                        segment = AudioSegment(
                            audio_chunk.tobytes(),
                            frame_rate=128000,
                            sample_width=sample_width,
                            channels=channels
                        )
                        self.buffered_audio += segment
                    else:
                        logging.warning("Received empty or invalid audio chunk.")
            except Exception as e:
                logging.error(f"Buffering error: {e}")
            finally:
                self.close_pipe()

    def read_audio_chunk(self):
        try:
            result, data = win32file.ReadFile(self.pipe, 512*1024)  # 64KB buffer size
            logging.info(f"Read {len(data)} bytes from pipe.")
            if len(data) % 4 != 0:  # Ensure it's a multiple of 4 (32-bit float)
                logging.warning("Received data length is not a multiple of 4.")
                return None
            return np.frombuffer(data, dtype=np.float32)
        except Exception as e:
            logging.error(f"Error reading from pipe: {e}")
            return None

    def export_to_mp3(self, dt):
        """Export buffered audio to the MP3 file periodically."""
        if len(self.buffered_audio) > 0:
            try:
                if os.path.exists(self.output_file):
                    existing_audio = AudioSegment.from_file(self.output_file, format="wav")
                    combined_audio = existing_audio + self.buffered_audio
                else:
                    combined_audio = self.buffered_audio

                # Export the combined audio
                combined_audio.export(self.output_file, format="wav", bitrate="16k")
                logging.info(f"Appended audio to {self.output_file}")

                # Clear the buffer after exporting
                self.buffered_audio = AudioSegment.silent(duration=0)
            except Exception as e:
                logging.error(f"Failed to append to MP3: {e}")

    def close_pipe(self):
        try:
            if self.pipe:
                win32file.CloseHandle(self.pipe)
                self.pipe_connected = False
                logging.info("Pipe closed.")
        except Exception as e:
            logging.error(f"Failed to close pipe: {e}")

    def on_stop(self):
        self.stop_audio_thread()
        self.close_pipe()

if __name__ == '__main__':
    try:
        AudioClientApp().run()
    except KeyboardInterrupt:
        logging.info("Application interrupted. Exiting...")
        sys.exit(0)
