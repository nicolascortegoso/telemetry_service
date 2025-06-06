import unittest
import yaml
from unittest.mock import MagicMock
from datetime import datetime, timedelta

from src.core.generator import generate_clean_telemetry_data
from src.core.telemetry import Telemetry


class TestGenerateCleanTelemetryData(unittest.TestCase):
    
    def setUp(self):
        # Create a mock telemetry instance
        self.mock_telemetry = Telemetry(vehicle=None)
        # Mock the generate_telemetry_point method
        self.mock_telemetry.generate_telemetry_point = MagicMock(side_effect=lambda ts: {
            "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%S") + "Z",
            "wheel_rpm": 100.0,
            "speed": 80.0,
            "distance": 22.22,
        })

    @classmethod
    def setUpClass(cls):
        # Runs once before all test methods
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
        cls.min_minutes = config['timeseries']['min_minutes']

    def test_length_of_data(self):
        # Request less than 60 minutes to test minimum duration
        data = generate_clean_telemetry_data(self.mock_telemetry, minutes=30)
        self.assertEqual(len(data), self.min_minutes * 60)

        data = generate_clean_telemetry_data(self.mock_telemetry, minutes=120)
        self.assertEqual(len(data), (self.min_minutes * 2) * 60)  # 120 minutes

    def test_timestamps_increment(self):
        start = datetime(2025, 6, 5, 12, 0, 0)
        data = generate_clean_telemetry_data(self.mock_telemetry, minutes=1, start_time=start)

        # Check timestamps increment by one second
        for i in range(len(data)):
            expected_timestamp = (start + timedelta(seconds=i)).strftime("%Y-%m-%dT%H:%M:%S") + "Z"
            self.assertEqual(data[i]["timestamp"], expected_timestamp)

    def test_data_content(self):
        # Just verify that data contains expected keys and values come from mock
        data = generate_clean_telemetry_data(self.mock_telemetry, minutes=1)
        sample = data[0]
        self.assertIn("timestamp", sample)
        self.assertIn("wheel_rpm", sample)
        self.assertIn("speed", sample)
        self.assertIn("distance", sample)
        self.assertEqual(sample["wheel_rpm"], 100.0)
        self.assertEqual(sample["speed"], 80.0)
        self.assertEqual(sample["distance"], 22.22)


if __name__ == "__main__":
    unittest.main()
