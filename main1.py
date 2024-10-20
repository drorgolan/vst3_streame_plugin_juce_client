import time
import win32file
import win32pipe
import numpy as np
import sounddevice as sd
from pydub import AudioSegment
from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.uix.boxlayout import BoxLayout
from threading import Thread, Event
import os
import logging
from scipy.signal import butter, lfilter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AudioClientApp(App):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pipe_connected = False
        self.pipe = None
        self.output_file = "./tmp/audio_files/output.mp3"

        # Ensure the directory for output files exists
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)

        # Initialize AudioSegment object to store audio
        self.audio_segment = AudioSegment.empty()

        # Event for stopping the audio buffering thread
        self.stop_event = Event()

    def build(self):
        # Create layout with buttons
        layout = BoxLayout(orientation='vertical')

        # Button to start audio buffering
        self.start_button = Button(text='Start Listening for Audio', on_press=self.start_audio_thread)
        layout.add_widget(self.start_button)

        # Button to stop audio buffering
        self.stop_button = Button(text='Stop Listening', on_press=self.stop_audio_thread)
        layout.add_widget(self.stop_button)

        # Button to play the saved MP3 file
        self.play_button = Button(text='Play Saved Audio', on_press=self.play_audio_file)
        layout.add_widget(self.play_button)

        # Input field for pipe name
        self.pipe_name_input = TextInput(text='Enter Pipe Name Here', multiline=False)
        layout.add_widget(self.pipe_name_input)

        return layout

    def start_audio_thread(self, instance):
        """Start the audio listening in a separate thread."""
        self.stop_event.clear()  # Clear the stop event
        self.audio_thread = Thread(target=self.audio_buffering, daemon=True)
        self.audio_thread.start()

    def stop_audio_thread(self, instance):
        """Stop the audio listening thread."""
        self.stop_event.set()  # Set the stop event to signal the thread to stop

    def audio_buffering(self):
        """Listen to audio data and save it to an MP3 file continuously."""
        total_samples = 44100 * 3 * 2  # 3 seconds of stereo audio
        audio_buffer = np.zeros(total_samples, dtype=np.float32)
        buffer_index = 0

        while not self.stop_event.is_set():
            if not self.connect_to_pipe():
                logging.info("Retrying connection in 2 seconds...")
                time.sleep(2)
                continue  # Try to connect again

            try:
                while not self.stop_event.is_set():
                    audio_chunk = self.read_audio_chunk()
                    if audio_chunk is not None:
                        chunk_size = len(audio_chunk)
                        if buffer_index + chunk_size <= total_samples:
                            audio_buffer[buffer_index:buffer_index + chunk_size] = audio_chunk
                            buffer_index += chunk_size
                        if buffer_index >= total_samples:  # Only save when buffer is full
                            logging.info("Buffer full, saving to MP3...")
                            segment = AudioSegment(
                                audio_buffer.tobytes(),
                                frame_rate=44100,
                                sample_width=4,
                                channels=2
                            )
                            self.audio_segment += segment
                            self.audio_segment.export(self.output_file, format="mp3")
                            logging.info(f"Saved to {self.output_file}")
                            buffer_index = 0  # Reset buffer index
                    else:
                        logging.warning("No audio data received. Attempting to reconnect...")
                        break  # Exit the inner while to reconnect
            except Exception as e:
                logging.error(f"Buffering error: {e}")
                time.sleep(1)  # Wait before attempting to reconnect
            finally:
                self.close_pipe()

    def connect_to_pipe(self):
        """Connect to an existing named pipe."""
        pipe_name = self.pipe_name_input.text  # Get the pipe name from the input
        try:
            self.pipe = win32file.CreateFile(
                r'\\.\pipe\\' + pipe_name,
                win32file.GENERIC_READ | win32file.GENERIC_WRITE,
                0,
                None,
                win32file.OPEN_EXISTING,
                0,
                None
            )
            self.pipe_connected = True
            logging.info("Connected to the pipe.")
            return True
        except Exception as e:
            logging.error(f"Failed to connect to pipe: {e}")
            self.pipe_connected = False
            return False



    def butter_highpass(cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
        return b, a

    def highpass_filter(data, cutoff=500.0, fs=44100.0, order=5):
        b, a = butter_highpass(cutoff, fs=fs, order=order)
        y = lfilter(b, a, data)
        return y

    def read_audio_chunk(self):
        """Read a small chunk of audio data from the pipe and apply high-pass filter."""
        if not self.pipe_connected:
            logging.error("Pipe is not connected. Cannot read data.")
            return None

        try:
            result, data = win32file.ReadFile(self.pipe, 1024 * 4)  # Read 4KB of data
            audio_chunk = np.frombuffer(data, dtype=np.float32)  # Convert byte data to float array

            # Apply high-pass filter
            audio_chunk_filtered = highpass_filter(audio_chunk)

            return audio_chunk_filtered
        except Exception as e:
            logging.error(f"Error reading from pipe: {e}")
            return None

    def play_audio_file(self, instance):
        """Play the saved MP3 file."""
        try:
            logging.info(f"Playing {self.output_file}...")
            audio = AudioSegment.from_mp3(self.output_file)
            sd.play(np.array(audio.get_array_of_samples()), samplerate=audio.frame_rate, channels=audio.channels)
            sd.wait()  # Wait for playback to finish
        except Exception as e:
            logging.error(f"Failed to play audio: {e}")

    def close_pipe(self):
        """Close the pipe connection."""
        try:
            if hasattr(self, 'pipe') and self.pipe:
                win32file.CloseHandle(self.pipe)
                self.pipe_connected = False  # Update connection status
                logging.info("Pipe closed.")
        except Exception as e:
            logging.error(f"Failed to close pipe: {e}")

    def on_stop(self):
        """Called when the application is stopping."""
        self.stop_audio_thread(None)  # Ensure the audio thread is stopped
        self.close_pipe()  # Close the pipe when the app stops

if __name__ == '__main__':
    AudioClientApp().run()
