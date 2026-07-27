import json
import re
import time
from datetime import datetime

import serial


PORT = "COM12"
BAUDRATE = 115200
TIMEOUT = 1
OUTPUT_FILE = "sensor_data.json"


class ESPSerialReader:
    def __init__(self, port=PORT, baudrate=BAUDRATE, timeout=TIMEOUT):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.connection = None

    def connect(self):
        try:
            self.connection = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout,
            )
            time.sleep(2)
            return True
        except serial.SerialException as e:
            print(f"Serial connection error: {e}")
            self.connection = None
            return False

    def is_connected(self):
        return self.connection is not None and self.connection.is_open

    def read_line(self):
        if not self.is_connected():
            return None
        try:
            line = self.connection.readline().decode("utf-8", errors="ignore").strip()
            return line if line else None
        except Exception as e:
            print(f"Read error: {e}")
            return None

    def parse_line(self, line):
        if not line:
            return None

        try:
            data = json.loads(line)
            if "temperature" in data or "humidity" in data:
                return {
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "temperature": float(data.get("temperature", 0)),
                    "humidity": float(data.get("humidity", 0)),
                }
        except Exception:
            pass

        temp_match = re.search(r"Temp\s*:\s*([\d.\-]+)", line, re.IGNORECASE)
        hum_match = re.search(r"Humidity\s*:\s*([\d.\-]+)", line, re.IGNORECASE)

        if temp_match or hum_match:
            return {
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "temperature": float(temp_match.group(1)) if temp_match else 0.0,
                "humidity": float(hum_match.group(1)) if hum_match else 0.0,
            }

        return None

    def close(self):
        if self.is_connected():
            self.connection.close()


def save_data(data, output_file=OUTPUT_FILE):
    payload = {
        "timestamp": data.get("timestamp", datetime.now().strftime("%H:%M:%S")),
        "temperature": float(data.get("temperature", 0.0)),
        "humidity": float(data.get("humidity", 0.0)),
    }
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main():
    reader = ESPSerialReader()
    if not reader.connect():
        return

    print(f"Connected to {reader.port}")
    try:
        while True:
            line = reader.read_line()
            if not line:
                continue

            print(f"RAW: {line}")
            data = reader.parse_line(line)
            if data:
                save_data(data)
                print(f"Saved: {data}")
    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        reader.close()


if __name__ == "__main__":
    main()
