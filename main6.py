import os
import logging
import time
import numpy as np
from threading import Thread, Event
from pydub import AudioSegment
import win32file
import win32con
from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.uix.boxlayout import BoxLayout
from kivy.clock import Clock
from kivy.graphics import Color, Line, Rectangle
from kivy.uix.widget import Widget

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class WaveformWidget(Widget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.audio_chunk = np.array([])  # Placeholder for audio data
        self.update_event = Event()
        self.bind(size=self.update_canvas)  # Update on size change

    def update_waveform(self, audio_chunk):
        """Update the waveform with new audio data."""
        if len(audio_chunk) <= 0:
            logging.warning("Received an empty audio chunk.")
            return

        self.audio_chunk = audio_chunk
        self.update_event.set()  # Signal to update the waveform

    def draw_waveform(self, dt):
        """Draw the waveform from the latest audio data."""
        if not self.update_event.is_set():
            return

        if self.audio_chunk is None or len(self.audio_chunk) == 0:
            logging.warning("Audio chunk is empty or None.")
            return

        # Normalize audio data for drawing
        max_value = np.max(np.abs(self.audio_chunk))
        if max_value == 0:  # Avoid division by zero
            logging.warning("Max value of audio chunk is 0, normalization skipped.")
            normalized_data = np.zeros_like(self.audio_chunk)  # Use zeros if there's no valid data
        else:
            normalized_data = ((self.audio_chunk - np.mean(self.audio_chunk)) / max_value) * (self.height / 2) + (self.height / 2)

        # Scale the x-axis to fit the widget's width
        step = max(1, len(normalized_data) // self.width)

        # Generate points to fit the widget's size dynamically
        points = []
        for i in range(0, len(normalized_data), step):
            x = i * (self.width / (len(normalized_data) // step))  # Scale x to widget width
            y = normalized_data[i] if not np.isnan(normalized_data[i]) else 0  # Ensure y is not NaN
            points.extend([x, y])

        # Clear the canvas before drawing the new line
        self.canvas.clear()

        # Draw the waveform
        with self.canvas:
            Color(1, 1, 1)  # Set color to white for the waveform line
            Line(points=points, width=2)  # Adjust line width for better visibility

        logging.info(f"Waveform drawn with {len(points)//2} points.")
        self.update_event.clear()  # Reset the update event

    def on_size(self, *args):
        """Update the canvas when the size changes."""
        self.draw_waveform(None)

    def update_canvas(self, *args):
        """Ensure the canvas is redrawn on size change."""
        self.canvas.clear()  # Clear the existing canvas
        if self.audio_chunk.size > 0:
            self.draw_waveform(None)

class AudioClientApp(App):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pipe_connected = False
        self.pipe = None
        self.stop_event = Event()
        self.buffered_audio = AudioSegment.silent(duration=0)  # Start with empty audio buffer
        self.output_file = "./tmp/audio_files/output.mp3"
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)

        # Thread to handle audio buffering
        self.audio_thread = None

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
        if self.audio_thread is None or not self.audio_thread.is_alive():
            self.stop_event.clear()
            self.audio_thread = Thread(target=self.audio_buffering, daemon=True)
            self.audio_thread.start()

            # Schedule waveform drawing updates every 0.1 seconds
            Clock.schedule_interval(self.waveform_widget.draw_waveform, 0.1)

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
        sample_width = 4  # 32-bit float audio (4 bytes per sample)
        channels = 2  # Stereo audio
        frame_rate = 48000  # Ensure this matches JUCE settings
        samples_per_channel = 480  # Number of samples per channel
        expected_chunk_size = samples_per_channel * channels  # 960 samples total (2 channels)
        expected_length = expected_chunk_size * sample_width  # 3840 bytes

        while not self.stop_event.is_set():
            if not self.connect_to_pipe():
                time.sleep(2)  # Try reconnecting if the pipe is not connected
                continue

            try:
                while not self.stop_event.is_set():
                    audio_chunk = self.read_audio_chunk(expected_length)  # Read audio chunk
                    if audio_chunk is not None and len(audio_chunk) > 0:
                        logging.info(f"Processing audio chunk with {len(audio_chunk)} samples.")

                        # Normalize the audio chunk
                        audio_chunk = self.normalize_audio_chunk(audio_chunk)

                        # Update waveform on the main thread
                        Clock.schedule_once(lambda dt: self.waveform_widget.update_waveform(audio_chunk))

                        logging.info("Audio chunk processed.")
                    else:
                        logging.warning("Received empty or invalid audio chunk.")
                    time.sleep(0.01)  # Sleep to yield control to the GUI
            except Exception as e:
                logging.error(f"Buffering error: {e}")
            finally:
                self.close_pipe()  # Ensure the pipe is closed on exit

    def read_audio_chunk(self, expected_length=48 * 1024):
        try:
            # Read data from the pipe
            result, data = win32file.ReadFile(self.pipe, expected_length)
            logging.info(f"Read {len(data)} bytes from pipe.")

            if len(data) == 0:
                logging.warning("Received zero-length data from pipe.")
                return np.array([])  # Return an empty array if no data is received

            # Convert bytes to numpy array
            audio_data = np.frombuffer(data, dtype=np.float32)  # Assuming float32
            return audio_data
        except Exception as e:
            logging.error(f"Error reading from pipe: {e}")
            return None

    def normalize_audio_chunk(self, audio_chunk):
        """Normalize the audio chunk."""
        # if audio_chunk.size == 0:
        #     return audio_chunk  # Return empty if no data
        #
        # # Normalize the audio data to the range [-1.0, 1.0]
        # max_value = np.max(np.abs(audio_chunk))
        #
        # threshold = 0.01  # Minimum threshold to avoid static line
        # if max_value < threshold:
        #     normalized_audio = np.zeros_like(audio_chunk)  # or handle differently
        # else:
        #     normalized_audio = audio_chunk / max_value

        #return normalized_audio
        return audio_chunk;

    def close_pipe(self):
        """Close the pipe safely."""
        if self.pipe is not None:
            win32file.CloseHandle(self.pipe)
            self.pipe_connected = False
            logging.info("Pipe closed.")

if __name__ == '__main__':
    AudioClientApp().run()
