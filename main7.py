import numpy as np
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.widget import Widget
from kivy.graphics import Line
from kivy.clock import Clock
import socket
import threading
import time


class WaveformDisplay(Widget):
    def __init__(self, **kwargs):
        super(WaveformDisplay, self).__init__(**kwargs)
        self.audio_buffer = None
        self.bind(pos=self.update_canvas, size=self.update_canvas)

    def set_audio_buffer(self, audio_buffer):
        """Store the audio buffer to visualize and update the canvas."""
        self.audio_buffer = self.low_pass_filter(audio_buffer)  # Apply low-pass filter
        Clock.schedule_once(self.update_canvas)

    def low_pass_filter(self, audio_data, window_size=5):
        """Apply a simple moving average filter to smooth the audio data."""
        if len(audio_data) < window_size:
            return audio_data  # If the buffer is smaller than the window size, return as is

        # Create a kernel for the moving average
        kernel = np.ones(window_size) / window_size
        filtered_data = np.convolve(audio_data, kernel, mode='valid')  # Apply the filter
        return filtered_data

    def update_canvas(self, *args):
        """Update the canvas to visualize the audio buffer as a waveform."""
        if self.audio_buffer is None or len(self.audio_buffer) == 0:
            return

        with self.canvas:
            self.canvas.clear()  # Clear previous drawings
            width, height = self.size
            mid_height = height / 2

            # Normalize the audio buffer to fit within the widget's height
            max_amplitude = np.max(np.abs(self.audio_buffer)) if np.max(np.abs(self.audio_buffer)) > 0 else 1
            scaled_waveform = (self.audio_buffer / max_amplitude) * (height / 2)

            # Create line points from the audio buffer
            points = []
            for i, sample in enumerate(scaled_waveform):
                x = (i / len(scaled_waveform)) * width
                y = mid_height + sample + 400
                points.extend([x, y])

            if points:
                Line(points=points, width=1)
            else:
                print("No points to draw.")

class UDPListenerApp(App):
    def build(self):
        self.destination_ip = "127.0.0.1"  # IP to listen on
        self.destination_port = 5005  # Port to listen on
        self.threads = []  # List to track all threads
        self.keep_listening = False  # Initialize keep_listening flag

        # Layout
        layout = BoxLayout(orientation='vertical')

        # Info label
        self.info_label = Label(text="Waiting for input...", size_hint=(1, None), height=50)
        layout.add_widget(self.info_label)

        # Create a container for the WaveformDisplay using FloatLayout
        waveform_container = FloatLayout(size_hint=(1, None), height=100)
        self.waveform_display = WaveformDisplay(size_hint=(1, 1))  # Set height to fill the container
        waveform_container.add_widget(self.waveform_display)

        layout.add_widget(waveform_container)

        # Start Listening button
        start_button = Button(text="Start Listening", size_hint=(1, None), height=50)
        start_button.bind(on_press=self.start_listening)  # Ensure this matches the method name
        layout.add_widget(start_button)

        return layout

    def log(self, message):
        """Log messages in the log output area."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}\n"
        # Here you could add code to display log messages if needed

    def start_listening(self, instance):
        """Start listening for UDP packets on the specified IP and port."""
        try:
            # Create UDP socket and bind to IP/Port
            self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # Enable SO_REUSEADDR
            self.udp_socket.bind((self.destination_ip, self.destination_port))

            self.info_label.text = f"Listening on {self.destination_ip}:{self.destination_port}"
            self.log(f"Listening started on {self.destination_ip}:{self.destination_port}")

            self.keep_listening = True  # Start listening
            # Start a thread to listen for incoming data
            listening_thread = threading.Thread(target=self.listen_for_data)
            listening_thread.daemon = True  # Daemonize thread to close with app
            self.threads.append((listening_thread, self.udp_socket))  # Store thread and socket
            listening_thread.start()

        except Exception as e:
            self.info_label.text = f"Error: {str(e)}"
            self.log(f"Error starting listener: {str(e)}")

    def listen_for_data(self):
        """Listen for incoming UDP data and update the waveform display."""
        self.udp_socket.setblocking(0)  # Set the socket to non-blocking mode
        while self.keep_listening:  # Continue while the flag is True
            try:
                # Receive data from the socket (adjust buffer size as needed)
                data, addr = self.udp_socket.recvfrom(4096)

                # Convert the received data to a NumPy array
                audio_data = np.frombuffer(data, dtype=np.float32)

                # Ensure the UI update is scheduled on the main thread
                Clock.schedule_once(lambda dt: self.waveform_display.set_audio_buffer(audio_data))

                # Log the reception of data
                self.log(f"Received data from {addr}")

            except BlockingIOError:
                # Ignore the error if no data is available
                time.sleep(0.01)  # Sleep briefly to avoid busy waiting
            except OSError as e:
                self.log(f"Error: {e}")
                break  # Break the loop if the socket is closed

    def stop_all_threads(self):
        """Stop all running threads and close their sockets."""
        for thread, udp_socket in self.threads:
            udp_socket.close()  # Close the socket to unblock recvfrom
            thread.join()  # Wait for the thread to finish

        self.threads.clear()  # Clear the thread list after stopping all

    def on_stop(self):
        """Clean up resources when the application is closing."""
        self.keep_listening = False  # Stop the listening loop
        self.stop_all_threads()  # Stop all threads

if __name__ == '__main__':
    UDPListenerApp().run()
