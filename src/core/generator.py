from datetime import datetime, timedelta
from typing import Dict, List

from src.core.telemetry import Telemetry


def generate_clean_telemetry_data(telemetry: Telemetry, minutes: int, start_time: datetime = None) -> List[Dict[str, float]]:
    """
    Generate clean (anomaly-free) telemetry data over a given duration.

    Args:
        telemetry (Telemetry): An instance of the Telemetry class to generate data from.
        minutes (int): Number of minutes to simulate telemetry data. 
                       Minimum duration is 60 minutes.
        start_time (datetime, optional): Starting timestamp for the simulation. 
                                         Defaults to the current datetime.

    Returns:
        List[Dict[str, float]]: A list of telemetry data points, one per simulated second.
                                Each data point includes timestamp, speed, distance, and wheel RPM.
    """

    # If no time is provided it is set to the current time
    if start_time is None:
        start_time = datetime.now()
    
    # Generates telemetry at intervals of at least 60 minutes
    minutes = minutes if minutes > 60 else 60
    # Intervals ​​are simulated in 1 second increments.
    intervals = 60 * minutes

    data = []
    for i in range(intervals):
        timestamp = start_time + timedelta(seconds=i)
        data.append(telemetry.generate_telemetry_point(timestamp))

    return data
