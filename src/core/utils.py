import pandas as pd
from typing import Dict, List

from src.core.anomalies import Anomaly


def introduce_anomalies(telemetry_data: List[Dict], anomalies: List[Anomaly], labels: bool) -> None:
    """
    Introduce anomalies into telemetry data by applying anomaly effects to each data entry.

    For each telemetry data point, this function checks whether each anomaly should apply based on its probability.
    If so, it modifies the data point accordingly and sets the 'is_anomaly' label to True; otherwise, sets it to False.

    Args:
        telemetry_data (List[Dict]): List of telemetry data points (each a dictionary) to modify.
        anomalies (List[Anomaly]): List of anomaly instances to consider applying.
        labels (bool): Whether to add an 'is_anomaly' key to each data point indicating if an anomaly was applied.

    Returns:
        None: The function modifies telemetry_data in place.
    """

    for entry in telemetry_data:
        for anomaly in anomalies:
            if anomaly.should_apply():
                anomaly.apply(entry)
                if labels:
                    entry['is_anomaly'] = True
            else:
                if labels:
                    entry['is_anomaly'] = False


def create_file_csv(data: List[Dict], output_file: str) -> None:
    """
    Write a list of dictionaries to a CSV file.

    The first dictionary's keys are used as CSV headers. Each dictionary in the list represents a row.
    All values are converted to strings before writing.

    Args:
        data (List[Dict]): List of dictionaries containing data to write to the CSV file.
        output_file (str): Path to the output CSV file.

    Returns:
        None: Creates or overwrites the CSV file at the specified location.
    """
        
    with open(output_file, 'w') as file:
        # Write the header
        headers = data[0].keys()
        file.write(','.join(headers) + '\n')

        # Write each dictionary as a row
        for row in data:
            # Convert each value to a string
            values = [str(value) for value in row.values()]
            file.write(','.join(values) + '\n')
    

def find_true_sublists(boolean_list: List[bool]) -> List:
    """
    Identifies contiguous sequences of True values in a boolean list.

    This function scans a list of boolean values and returns a list of tuples, 
    where each tuple represents the start and end indices (inclusive) of a 
    contiguous sublist of True values.
    """

    result = []
    start_idx = None
    
    for i, value in enumerate(boolean_list):
        if value and start_idx is None:
            # Start of a new True sequence
            start_idx = i
        elif not value and start_idx is not None:
            # End of a True sequence
            result.append((start_idx, i - 1))
            start_idx = None
    
    # Handle case where the list ends with a True sequence
    if start_idx is not None:
        result.append((start_idx, len(boolean_list) - 1))

    return result


def construct_time_intervals(timestamps: pd.Series, intervals: List) -> List[tuple]:
    """
    Constructs a list of timestamp intervals from a pandas Series based on given index intervals.

    For each interval in the provided list of index tuples, this function extracts the corresponding
    timestamp range from the input Series and returns a list of tuples containing the first and last
    timestamps of each range.
    """

    timestamps_intervals = list()

    for interval in intervals:
        start, end = interval
        timestamp_interval = timestamps[start:end + 2].to_list()
        lower_upper_bounds = (timestamp_interval[0], timestamp_interval[-1])
        timestamps_intervals.append(lower_upper_bounds)
    
    return timestamps_intervals
