from abc import ABC
import random


class Vehicle(ABC):
    """
    Abstract base class representing a generic vehicle.

    Attributes:
        speed (float): The current speed of the vehicle in km/h.
        min_speed (float): The minimum allowable speed.
        max_speed (float): The maximum allowable speed.
        speed_variation (float): The range of random variation that can be applied to the speed.
    """
        
    def __init__(
        self,
        initial_speed: float,
        min_speed: float,
        max_speed: float,
        speed_variation: float,
        wheel_diameter: float
    ):
        self.speed = initial_speed              # Current speed in km/h
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.speed_variation = speed_variation
        self.wheel_diameter = wheel_diameter    # In meters


    def update_speed(self):
        """Apply random delta to the speed."""
        delta = random.uniform(-self.speed_variation, self.speed_variation)
        self.speed = max(min(self.speed + delta, self.max_speed), self.min_speed)


def get_vehicle_classes():
    """
    Retrieve all classes that directly inherit from the Vehicle base class.

    Returns:
        dict: A dictionary mapping subclass names to the subclass types.
    """

    return {cls.__name__: cls for cls in Vehicle.__subclasses__()}


class Car(Vehicle):
    """
    A concrete implementation of Vehicle representing a car.
    """

    def __init__(self, initial_speed: float = 80.0):
        super().__init__(
            initial_speed=initial_speed,
            min_speed=0.0,
            max_speed=140.0,
            speed_variation=2.0,
            wheel_diameter=0.65
        )

class Truck(Vehicle):
    """
    A concrete implementation of Vehicle representing a truck.
    """
        
    def __init__(self, initial_speed: float = 60.0):
        super().__init__(
            initial_speed=initial_speed,
            min_speed=0.0,
            max_speed=100.0,
            speed_variation=1.5,
            wheel_diameter=1.0
        )

class ElectricVehicle(Vehicle):
    """
    A concrete implementation of Vehicle representing an electric vehicle.
    """
        
    def __init__(self, initial_speed: float = 90.0):
        super().__init__(
            initial_speed=initial_speed,
            min_speed=0.0,
            max_speed=160.0,
            speed_variation=2.5,
            wheel_diameter=0.7
        )
