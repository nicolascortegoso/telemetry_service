from typing import Dict
from datetime import datetime
import math

from src.core.vehicles import Vehicle


class Telemetry:
    """
    A class responsible for simulating and generating telemetry data for a vehicle.

    Attributes:
        vehicle (Vehicle): An instance of a vehicle for which telemetry is being generated.
    """

    def __init__(self, vehicle: Vehicle):
        self.vehicle = vehicle

    def generate_telemetry_point(self, timestamp: datetime) -> Dict[str, float]:
        """
        Generate a single telemetry data point for the vehicle at a given timestamp.

        The method updates the vehicle's speed, computes the wheel revolutions per minute (RPM)
        based on wheel diameter, and calculates the distance traveled in one second.

        Args:
            timestamp (datetime): The time at which the telemetry data is generated.

        Returns:
            Dict[str, float]: A dictionary containing:
                - "timestamp" (str): ISO-formatted UTC timestamp.
                - "wheel_rpm" (float): Wheel revolutions per minute based on speed and wheel size.
                - "speed" (float): Current speed of the vehicle in km/h.
                - "distance" (float): Distance traveled in meters (for 1 second at current speed).
        """

        self.vehicle.update_speed()

        # Convert speed from km/h to m/s
        speed_mps = self.vehicle.speed * 1000 / 3600

        # Calculate wheel RPM based on circumference
        circumference = math.pi * self.vehicle.wheel_diameter  # in meters
        wheel_rpm = (speed_mps / circumference) * 60 if circumference > 0 else 0

        return {
            "timestamp": timestamp.strftime("%Y-%m-%dT%H:%M:%S") + "Z",
            "wheel_rpm": round(wheel_rpm, 2),
            "speed": round(self.vehicle.speed, 2),
            "distance": round(speed_mps, 2),
        }
