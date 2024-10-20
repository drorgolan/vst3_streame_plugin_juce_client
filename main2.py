from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.widget import Widget
from kivy.graphics import Line, Color
from kivy.clock import Clock
from threading import Thread, Event
import numpy as np
import win32file
import time
import logging
from pydub import AudioSegment
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class WaveformWidget(Widget):
    """Custom widget to display the audio waveform."""

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

        # For saving audio data
        self.audio_segment = AudioSegment.empty()
        self.output_file = "./tmp/audio_files/output.mp3"
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)  # Ensure directory exists

    def build(self):
        layout = BoxLayout(orientation='vertical')

        # Start and Stop Buttons
        self.start_button = Button(text='Start Listening', on_press=self.start_audio_thread)
        layout.add_widget(self.start_button)

        self.stop_button = Button(text='Stop Listening', on_press=self.stop_audio_thread)
        layout.add_widget(self.stop_button)

        # Pipe Name Input Field
        self.pipe_name_input = TextInput(text='Enter Pipe Name Here', multiline=False)
        layout.add_widget(self.pipe_name_input)

        # Waveform display widget
        self.waveform_widget = WaveformWidget(size_hint=(1, 0.8))
        layout.add_widget(self.waveform_widget)

        return layout

    def start_audio_thread(self, instance):
        """Start audio listening in a separate thread."""
        self.stop_event.clear()
        self.audio_segment = AudioSegment.empty()  # Reset audio segment
        self.audio_thread = Thread(target=self.audio_buffering, daemon=True)
        self.audio_thread.start()

    def stop_audio_thread(self, instance):
        """Stop the audio listening thread and save the MP3 file."""
        self.stop_event.set()
        self.save_to_mp3()

    def connect_to_pipe(self):
        """Connect to the named pipe."""
        pipe_name = self.pipe_name_input.text
        try:
            self.pipe = win32file.CreateFile(
                r'\\.\pipe\\' + pipe_name,
                win32file.GENERIC_READ | win32file.GENERIC_WRITE,
                0, None, win32file.OPEN_EXISTING, 0, None
            )
            self.pipe_connected = True
            logging.info("Connected to the pipe.")
            return True
        except Exception as e:
            logging.error(f"Failed to connect to pipe: {e}")
            self.pipe_connected = False
            return False

    def audio_buffering(self):
        """Continuously listen for audio data and schedule waveform updates."""
        while not self.stop_event.is_set():
            if not self.connect_to_pipe():
                time.sleep(2)
                continue

            try:
                while not self.stop_event.is_set():
                    audio_chunk = self.read_audio_chunk()
                    if audio_chunk is not None:
                        Clock.schedule_once(lambda dt: self.waveform_widget.update_waveform(audio_chunk))

                        # Convert chunk to pydub segment and append it
                        segment = AudioSegment(
                            audio_chunk.tobytes(),
                            frame_rate=44100,
                            sample_width=4,  # float32 -> 4 bytes per sample
                            channels=2
                        )
                        self.audio_segment += segment  # Append to the audio segment
                    else:
                        logging.warning("No audio data received. Reconnecting...")
                        break
            except Exception as e:
                logging.error(f"Buffering error: {e}")
                time.sleep(1)
            finally:
                self.close_pipe()

    def read_audio_chunk(self):
        """Read audio data from the pipe."""
        try:
            result, data = win32file.ReadFile(self.pipe, 1024 * 4)
            audio_chunk = np.frombuffer(data, dtype=np.float32)
            return audio_chunk
        except Exception as e:
            logging.error(f"Error reading from pipe: {e}")
            return None

    def save_to_mp3(self):
        """Save the collected audio to an MP3 file."""
        try:
            self.audio_segment.export(self.output_file, format="mp3")
            logging.info(f"Audio saved to {self.output_file}")
        except Exception as e:
            logging.error(f"Failed to save audio: {e}")

    def close_pipe(self):
        """Close the pipe connection."""
        try:
            if self.pipe:
                win32file.CloseHandle(self.pipe)
                self.pipe_connected = False
                logging.info("Pipe closed.")
        except Exception as e:
            logging.error(f"Failed to close pipe: {e}")

    def on_stop(self):
        """Handle app exit."""
        self.stop_audio_thread(None)
        self.close_pipe()

if __name__ == '__main__':
    AudioClientApp().run()
