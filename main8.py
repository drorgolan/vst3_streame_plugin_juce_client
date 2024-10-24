from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.slider import Slider
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
import numpy as np
import wave
import socket
import threading
import time
from kivy.uix.widget import Widget
from kivy.uix.floatlayout import FloatLayout
from kivy.graphics import Line, Color
from kivy.clock import Clock
from pydub import AudioSegment
import librosa

class WaveformDisplay(Widget):
    def __init__(self, **kwargs):
        super(WaveformDisplay, self).__init__(**kwargs)
        self.audio_buffer = None
        self.bind(pos=self.update_canvas, size=self.update_canvas)

    def set_audio_buffer(self, audio_buffer):
        self.audio_buffer = audio_buffer
        Clock.schedule_once(self.update_canvas)

    def update_canvas(self, *args):
        if self.audio_buffer is None or len(self.audio_buffer) == 0:
            return

        self.canvas.clear()

        with self.canvas:
            width, height = self.size
            mid_height = height / 2
            max_amplitude = np.max(np.abs(self.audio_buffer)) if np.max(np.abs(self.audio_buffer)) > 0 else 1
            scaled_waveform = (self.audio_buffer / max_amplitude) * (height / 2)

            points = []
            for i, sample in enumerate(scaled_waveform):
                x = (i / len(scaled_waveform)) * width
                y = mid_height + sample + 400
                points.extend([x, y])

            if points:
                Color(1, 0, 0)
                Line(points=points, width=1)

class UDPListenerApp(App):
    def __init__(self, **kwargs):
        super(UDPListenerApp, self).__init__(**kwargs)
        self.destination_ip = "127.0.0.1"
        self.destination_port = 5005
        self.threads = []
        self.keep_listening = False
        self.audio_buffers = []
        self.is_saving = False
        self.sample_rate = 48000
        self.buffer_size = 512
        self.wav_file = None
        self.lock = threading.Lock()
        self.semitones = -12  # Default semitone shift

    def build(self):
        layout = BoxLayout(orientation='vertical')

        self.info_label = Label(text="Waiting for input...", size_hint=(1, None), height=50)
        layout.add_widget(self.info_label)

        waveform_container = FloatLayout(size_hint=(1, None), height=100)
        self.waveform_display = WaveformDisplay(size_hint=(1, 1))
        waveform_container.add_widget(self.waveform_display)

        layout.add_widget(waveform_container)

        # Slider for semitones
        slider_layout = BoxLayout(orientation='horizontal', size_hint=(1, None), height=50)
        slider_label = Label(text="Semitone Shift:", size_hint=(0.5, 2), height=50)
        self.semitone_slider = Slider(min=-12, max=12, value=0, size_hint=(0.5, 2), height=50)
        self.semitone_slider.bind(value=self.on_semitone_change)
        slider_layout.add_widget(slider_label)
        slider_layout.add_widget(self.semitone_slider)
        layout.add_widget(slider_layout)

        start_button = Button(text="Start Listening", size_hint=(1, None), height=50)
        start_button.bind(on_press=self.start_listening)
        layout.add_widget(start_button)

        save_button = Button(text="Start Saving to WAV", size_hint=(1, None), height=50)
        save_button.bind(on_press=self.start_saving)
        layout.add_widget(save_button)

        stop_button = Button(text="Stop Saving to WAV", size_hint=(1, None), height=50)
        stop_button.bind(on_press=self.stop_saving)
        layout.add_widget(stop_button)

        return layout

    def on_semitone_change(self, instance, value):
        self.semitones = int(value)
        self.info_label.text = f"Semitone Shift: {self.semitones}"

    def log(self, message):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        print(log_message)

    def start_listening(self, instance):
        try:
            self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.udp_socket.bind((self.destination_ip, self.destination_port))

            self.info_label.text = f"Listening on {self.destination_ip}:{self.destination_port}"
            self.log(f"Listening started on {self.destination_ip}:{self.destination_port}")

            self.keep_listening = True
            listening_thread = threading.Thread(target=self.listen_for_data)
            listening_thread.daemon = True
            self.threads.append((listening_thread, self.udp_socket))
            listening_thread.start()

        except Exception as e:
            self.info_label.text = f"Error: {str(e)}"
            self.log(f"Error starting listener: {str(e)}")

    def listen_for_data(self):
        self.udp_socket.setblocking(0)
        while self.keep_listening:
            try:
                data, addr = self.udp_socket.recvfrom(4096)
                audio_data = np.frombuffer(data, dtype=np.float32)

                self.log(f"Raw audio data: {audio_data[:10]}...")

                if np.max(np.abs(audio_data)) > 0:
                    audio_data = (audio_data / np.max(np.abs(audio_data))).astype(np.float32)

                audio_data = self.correct_pitch(audio_data)
                Clock.schedule_once(lambda dt: self.waveform_display.set_audio_buffer(audio_data))

                if self.is_saving:
                    self.lock.acquire()
                    try:
                        self.audio_buffers.append(audio_data)
                    finally:
                        self.lock.release()

                self.log(f"Received data from {addr}")

            except BlockingIOError:
                time.sleep(0.01)
            except OSError as e:
                self.log(f"Error: {e}")
                break

    def correct_pitch(self, audio_data, semitones=None):
        if semitones is None:
            semitones = self.semitones

        # Normalize the audio data to float32 format
        audio_data = audio_data.astype(np.float32)

        # Convert to mono if needed
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)

        # Apply pitch shifting
        audio_data = librosa.effects.pitch_shift(audio_data, sr=self.sample_rate, n_steps=semitones)

        return audio_data

    def start_saving(self, instance):
        if self.is_saving:
            self.log("Already saving audio.")
            return

        self.is_saving = True
        output_filename = "output.wav"

        self.wav_file = wave.open(output_filename, 'wb')
        self.wav_file.setnchannels(2)
        self.wav_file.setsampwidth(2)
        self.wav_file.setframerate(self.sample_rate)

        saving_thread = threading.Thread(target=self.save_to_wav)
        saving_thread.daemon = True
        self.threads.append((saving_thread, None))
        saving_thread.start()

    def save_to_wav(self):
        while self.is_saving:
            if self.audio_buffers:
                audio_data = self.audio_buffers.pop(0)
                if not isinstance(audio_data, np.ndarray):
                    self.log("Error: audio_data is not a NumPy array.")
                    continue

                try:
                    audio_data_int16 = np.clip(audio_data * 32767, -32768, 32767).astype(np.int16)
                except Exception as e:
                    self.log(f"Error converting audio data to int16: {e}")
                    continue

                self.log(f"Audio data before saving: min {audio_data_int16.min()}, max {audio_data_int16.max()}")
                self.wav_file.writeframes(audio_data_int16.tobytes())

                time.sleep(0.01)

    def stop_saving(self, instance):
        if not self.is_saving:
            self.log("Not currently saving.")
            return

        self.is_saving = False
        if self.wav_file:
            self.wav_file.close()
            self.wav_file = None  # Clear the reference
        self.log(f"Audio saved to output.wav")

    def stop_all_threads(self):
        self.keep_listening = False
        for thread, udp_socket in self.threads:
            if udp_socket:
                udp_socket.close()
            thread.join()

        self.threads.clear()

    def on_stop(self):
        self.stop_all_threads()

if __name__ == '__main__':
    UDPListenerApp().run()
