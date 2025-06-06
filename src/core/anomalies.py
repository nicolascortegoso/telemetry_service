import random
from abc import ABC, abstractmethod
from typing import Dict


class Anomaly(ABC):
    """
    Abstract base class for telemetry anomalies.

    Attributes:
        duration (int): Duration (in seconds) the anomaly should remain active once triggered.
        probability (float): Probability of the anomaly being triggered at any given moment.
        active_counter (int): Counter to track how many seconds the anomaly remains active.
    """

    def __init__(self, duration: int, probability: float, ):
        self.duration = duration
        self.probability = probability / duration   # We divide the probability by duration to mantain the proportion
        self.active_counter = 0                     # Tracks how many seconds the anomaly should still apply

    def should_apply(self) -> bool:
        if self.active_counter > 0:
            return True
        return random.random() < self.probability

    @abstractmethod
    def apply(self, data_point: Dict) -> Dict:
        pass


def get_anomaly_classes():
    """
    Retrieve all subclasses of the Anomaly class.

    Returns:
        Dict[str, Type[Anomaly]]: A dictionary mapping class names to Anomaly subclasses.
    """
    
    return {cls.__name__: cls for cls in Anomaly.__subclasses__()}


class WheelSlip(Anomaly):
    """
    Anomaly that simulates wheel slip by doubling the wheel RPM.
    """

    def __init__(self, duration: int, probability: float):
        super().__init__(duration, probability)

    def apply(self, data_point: Dict) -> Dict:
        # Double the wheel_rpm to simulate wheel slip
        if self.active_counter == 0:
            self.active_counter = self.duration
        data_point["wheel_rpm"] *= 2
        self.active_counter -= 1

        return data_point
    

class GPSLoss(Anomaly):
    """
    Anomaly that simulates GPS signal loss by setting the distance to zero.
    """

    def __init__(self, duration: int, probability: float):
        super().__init__(duration, probability)

    def apply(self, data_point: Dict) -> Dict:
        # Set distance to zero to simulate GPS loss
        if self.active_counter == 0:
            self.active_counter = self.duration
        
        data_point["distance"] = 0
        self.active_counter -= 1
        
        return data_point


class SpeedSensorFreeze(Anomaly):
    """
    Anomaly that simulates a frozen speed sensor by holding the speed constant.
    """
    
    def __init__(self, duration: int, probability: float):
        super().__init__(duration, probability)
        self.frozen_speed = None

    def apply(self, data_point: Dict) -> Dict:
        if self.active_counter == 0:
            self.active_counter = self.duration
            self.frozen_speed = data_point["speed"]

        data_point["speed"] = self.frozen_speed
        self.active_counter -= 1

        return data_point
    