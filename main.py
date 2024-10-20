import socket
import numpy as np
from kivy.app import App
from kivy.graphics import Color, Line, Canvas
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.widget import Widget
from threading import Thread


class WaveformDisplay(Widget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data = []  # Store waveform data
        self.bind(size=self.update_canvas)  # Redraw on resize

    def update_canvas(self, *args):
        """Redraw the waveform on the canvas."""
        self.canvas.clear()  # Clear previous drawings
        with self.canvas:
            Color(1, 1, 1, 1)  # White color for waveform lines

            if self.data:
                step = self.width / len(self.data)  # Horizontal step per sample
                for i in range(len(self.data) - 1):
                    x1 = i * step
                    y1 = (self.data[i] + 1) * (self.height / 2)  # Scale Y values
                    x2 = (i + 1) * step
                    y2 = (self.data[i + 1] + 1) * (self.height / 2)
                    Line(points=[x1, y1, x2, y2], width=2)

    def update_data(self, new_data):
        """Update data and redraw the canvas."""
        self.data = new_data
        self.update_canvas()


class UDPReceiver(Thread):
    def __init__(self, ip, port, display):
        super().__init__()
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_socket.setblocking(0)  # Set non-blocking mode
        self.display = display
        self.running = True  # Control thread execution

        try:
            self.udp_socket.bind((ip, port))
            print(f"Listening for UDP packets on {ip}:{port}")
        except Exception as e:
            print("Failed to bind UDP socket:", e)

    def run(self):
        """Thread entry point to receive UDP data."""
        while self.running:
            try:
                data, _ = self.udp_socket.recvfrom(8192)  # Buffer size
                print(f"Data received: {len(data)} bytes")
                self.process_audio_data(data)
            except BlockingIOError:
                pass  # No data available yet
            except Exception as e:
                print("Error receiving data:", e)

    def stop(self):
        """Stop the thread."""
        self.running = False
        self.udp_socket.close()

    def process_audio_data(self, data):
        """Process received audio data."""
        num_samples = len(data) // 4  # Assuming float32 (4 bytes per sample)
        audio_data = np.frombuffer(data, dtype=np.float32)[:num_samples]

        if np.max(np.abs(audio_data)) != 0:
            audio_data = audio_data / np.max(np.abs(audio_data))  # Normalize

        # Update the waveform display with new data on the main thread
        self.display.update_data(audio_data)


class MyKivyApp(App):
    def build(self):
        self.receiver_thread = None  # Keep track of the UDP thread

        layout = BoxLayout(orientation='vertical')

        # Input fields for IP and port
        self.ip_input = TextInput(hint_text='Enter IP Address', multiline=False)
        self.port_input = TextInput(hint_text='Enter Port', multiline=False)
        start_button = Button(text='Start UDP Receiver')
        stop_button = Button(text='Stop UDP Receiver')

        start_button.bind(on_press=self.start_udp_receiver)
        stop_button.bind(on_press=self.stop_udp_receiver)

        # Waveform display widget
        self.waveform_display = WaveformDisplay()

        # Add widgets to the layout
        layout.add_widget(self.ip_input)
        layout.add_widget(self.port_input)
        layout.add_widget(start_button)
        layout.add_widget(stop_button)
        layout.add_widget(self.waveform_display)

        return layout

    def start_udp_receiver(self, instance):
        """Start the UDP receiver thread."""
        ip = self.ip_input.text
        port = int(self.port_input.text)

        if self.receiver_thread is None or not self.receiver_thread.is_alive():
            try:
                self.receiver_thread = UDPReceiver(ip, port, self.waveform_display)
                self.receiver_thread.start()
                print("UDP receiver started.")
            except socket.gaierror as e:
                print("Socket error:", e)
            except Exception as e:
                print("Unexpected error:", e)

    def stop_udp_receiver(self, instance):
        """Stop the UDP receiver thread."""
        if self.receiver_thread and self.receiver_thread.is_alive():
            self.receiver_thread.stop()
            self.receiver_thread.join()  # Ensure the thread has stopped
            print("UDP receiver stopped.")

    def on_stop(self):
        """Ensure the thread is stopped when the app exits."""
        if self.receiver_thread and self.receiver_thread.is_alive():
            self.receiver_thread.stop()
            self.receiver_thread.join()


if __name__ == '__main__':
    MyKivyApp().run()
