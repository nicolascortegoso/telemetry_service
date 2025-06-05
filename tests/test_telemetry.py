import unittest
from unittest.mock import MagicMock
from datetime import datetime
import math

from src.core.telemetry import Telemetry
from src.core.vehicles import Vehicle


class DummyVehicle(Vehicle):
    """A deterministic vehicle for testing."""
    def __init__(self, speed=72.0):  # 72 km/h = 20 m/s
        super().__init__(
            initial_speed=speed,
            min_speed=0.0,
            max_speed=150.0,
            speed_variation=0.0,  # no randomness
            wheel_diameter=0.6
        )

    def update_speed(self):
        pass  # Override to keep speed constant for test predictability


class TestTelemetry(unittest.TestCase):

    def test_generate_telemetry_point_output(self):
        vehicle = DummyVehicle()
        telemetry = Telemetry(vehicle)
        timestamp = datetime(2024, 1, 1, 12, 0, 0)

        result = telemetry.generate_telemetry_point(timestamp)

        expected_speed = 72.0  # in km/h
        expected_speed_mps = expected_speed * 1000 / 3600  # = 20.0
        expected_wheel_rpm = (expected_speed_mps / (math.pi * 0.6)) * 60

        self.assertEqual(result["timestamp"], "2024-01-01T12:00:00Z")
        self.assertAlmostEqual(result["speed"], expected_speed, places=2)
        self.assertAlmostEqual(result["distance"], expected_speed_mps, places=2)
        self.assertAlmostEqual(result["wheel_rpm"], round(expected_wheel_rpm, 2), places=2)

    def test_wheel_rpm_zero_when_diameter_zero(self):
        vehicle = DummyVehicle()
        vehicle.wheel_diameter = 0
        telemetry = Telemetry(vehicle)
        timestamp = datetime.utcnow()

        result = telemetry.generate_telemetry_point(timestamp)
        self.assertEqual(result["wheel_rpm"], 0.0)


if __name__ == '__main__':
    unittest.main()
