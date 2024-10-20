import win32file
import win32con  # For GENERIC_READ
import os
from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.widget import Widget
from kivy.graphics import Line, Color
from threading import Thread, Event
import numpy as np
import logging
import time
from pydub import AudioSegment
from kivy.clock import Clock

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

        self.output_file = "./tmp/audio_files/output.mp3"
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
        sample_width = 4  # 32-bit float audio (4 bytes per sample)
        channels = 2  # Stereo audio
        frame_rate = 48000  # Ensure this matches JUCE settings
        samples_per_chunk = 480  # Number of samples per channel
        expected_chunk_size = samples_per_chunk * channels  # 960 samples total (2 channels)
        expected_length = expected_chunk_size * sample_width  # 3840 bytes

        while not self.stop_event.is_set():
            if not self.connect_to_pipe():
                time.sleep(2)  # Try reconnecting if the pipe is not connected
                continue

            try:
                while not self.stop_event.is_set():
                    audio_chunk = self.read_audio_chunk(expected_length)
                    if audio_chunk is not None and len(audio_chunk) > 0:
                        logging.info(f"Processing audio chunk with {len(audio_chunk)} samples.")

                        # Normalize audio chunk
                        audio_chunk = self.normalize_audio_chunk(audio_chunk)

                        # Check if padding is required (in case of partial chunks)
                        if len(audio_chunk) < expected_chunk_size:
                            padding_length = expected_chunk_size - len(audio_chunk)
                            logging.warning(f"Padding audio chunk by {padding_length} samples.")
                            audio_chunk = np.pad(audio_chunk, (0, padding_length), 'constant')

                        # Convert audio_chunk to bytes
                        audio_bytes = audio_chunk.tobytes()  # Ensure to convert to bytes

                        # Create an AudioSegment from the chunk and append it to the buffer
                        segment = AudioSegment(
                            audio_bytes,
                            frame_rate=frame_rate,
                            sample_width=sample_width,
                            channels=channels
                        )
                        self.buffered_audio += segment  # Append to buffered audio

                        # Update the waveform on the main thread
                        Clock.schedule_once(lambda dt: self.waveform_widget.update_waveform(audio_chunk))

                        logging.info("Audio chunk appended to buffer.")
                        logging.info(f"Audio frame rate: {segment.frame_rate}")
                    else:
                        logging.warning("Received empty or invalid audio chunk.")
            except Exception as e:
                logging.error(f"Buffering error: {e}")
            finally:
                self.close_pipe()

    def read_audio_chunk(self, expected_length=48 * 1024, num_channels=2):
        try:
            # Read data from the pipe
            result, data = win32file.ReadFile(self.pipe, expected_length)
            logging.info(f"Read {len(data)} bytes from pipe.")

            # Check for data integrity
            if len(data) % 4 != 0:
                logging.warning("Received data length is not a multiple of 4.")
                return None

            # Convert raw bytes to a float32 numpy array
            audio_chunk = np.frombuffer(data, dtype=np.float32)

            # Check if the array is empty
            if audio_chunk.size == 0:
                logging.warning("Received empty audio chunk.")
                return None

            # Calculate total samples and pad if necessary to ensure it can be reshaped
            total_samples = audio_chunk.size // num_channels
            if audio_chunk.size % num_channels != 0:
                padding_length = num_channels - (audio_chunk.size % num_channels)
                logging.warning(f"Padding audio chunk with {padding_length} zeros.")
                audio_chunk = np.pad(audio_chunk, (0, padding_length), 'constant')

            # Reshape the audio chunk to (num_samples, num_channels)
            audio_chunk = audio_chunk.reshape(-1, num_channels)
            logging.info(f"Reshaped audio chunk: {audio_chunk.shape}")

            return audio_chunk.flatten()  # Return a 1D array for processing
        except Exception as e:
            logging.error(f"Error reading audio chunk: {e}")
            return None

    def export_to_mp3(self, dt):
        """Export buffered audio to the MP3 file periodically."""
        if len(self.buffered_audio) > 0:
            try:
                if os.path.exists(self.output_file):
                    # Read the existing MP3 file
                    existing_audio = AudioSegment.from_file(self.output_file, format="mp3")
                    # Combine the existing audio with the buffered audio
                    combined_audio = existing_audio + self.buffered_audio
                else:
                    # If the file doesn't exist, use the buffered audio as is
                    combined_audio = self.buffered_audio

                # Normalize combined audio to avoid pitch shifts
                combined_audio = combined_audio.set_frame_rate(48000)  # Ensure correct frame rate
                combined_audio = combined_audio.set_channels(2)  # Ensure stereo output

                # Export the combined audio
                combined_audio.export(self.output_file, format="mp3", bitrate="192k")
                logging.info(f"Exported buffered audio to {self.output_file}")

                # Clear the buffer after exporting
                self.buffered_audio = AudioSegment.silent(duration=0)  # Reset buffer
            except Exception as e:
                logging.error(f"Failed to export to MP3: {e}")
        else:
            logging.warning("No audio data to export.")

    def normalize_audio_chunk(self, audio_chunk):
        """Normalize the audio chunk to avoid clipping and static noise."""
        max_val = np.max(np.abs(audio_chunk))
        if max_val > 0:
            audio_chunk = audio_chunk / max_val  # Normalize to -1.0 to 1.0
        return audio_chunk

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
        logging.info("Application stopped by user.")
