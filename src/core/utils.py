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
    