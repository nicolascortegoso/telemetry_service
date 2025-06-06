import argparse

from src.core.generator import generate_clean_telemetry_data
from src.core.vehicles import get_vehicle_classes
from src.core.telemetry import Telemetry
from src.core.anomalies import get_anomaly_classes
from src.core.utils import introduce_anomalies, create_file_csv
from src.core.config import get_config


config = get_config()


def parse_vehicle_arg(value: str):
    """Parse and validate the --vehicle argument (format: VehicleName:InitialSpeed)."""
    try:
        vehicle_name, initial_speed_str = value.split(":")
        initial_speed = float(initial_speed_str)
    except ValueError:
        raise argparse.ArgumentTypeError(
            "Vehicle must be in format VehicleName:InitialSpeed, e.g., Car:80"
        )

    available = get_vehicle_classes()
    if vehicle_name not in available:
        raise argparse.ArgumentTypeError(
            f"Unknown vehicle '{vehicle_name}'. Available vehicles: {', '.join(available)}"
        )

    return vehicle_name, initial_speed


def parse_minutes_arg(value):
    """Parse and validate the --minutes argument."""
    try:
        ivalue = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("Minutes must be an integer.")
    if ivalue < config['timeseries']['min_minutes']:
        raise argparse.ArgumentTypeError("Minimum value for --minutes is 60.")
    return ivalue


def parse_anomalies_arg(arg):
    """Parse and validate the --anomalies argument (format: AnomalyName:Duration:Probability)."""
    try:
        name, duration_str, probability_str = arg.split(":")
        duration = int(duration_str)
        probability = float(probability_str)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Anomalies must be in format AnomalyName:Duration:Probability. Invalid: '{arg}'"
        )
    anomaly_classes = get_anomaly_classes()
    if name not in anomaly_classes:
        raise argparse.ArgumentTypeError(
            f"Unknown anomaly '{name}'. Available anomalies: {', '.join(anomaly_classes)}"
        )
    if not (1 <= duration <= 60):
        raise argparse.ArgumentTypeError(
            f"The duration for anomaly '{name}' must be between 1 and 60."
        )
    if not (0.0 <= probability <= 1.0):
        raise argparse.ArgumentTypeError(
            f"Probability for anomaly '{name}' must be between 0.0 and 1.0."
        )
    return (name, duration, probability)


def main():
    """
    Command-line interface for generating synthetic telemetry data.

    This function parses command-line arguments to generate the data for a specified vehicle type over a given duration.
    It optionally introduces anomalies (e.g., wheel slip, GPS loss) with user-defined probabilities
    and durations. The final dataset is saved as a CSV file with optional labels.
    """

    parser = argparse.ArgumentParser(
        description="Produces synthetic telemetry data for a type of vehicle.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Parse vehicle argument
    vehicle_classes = get_vehicle_classes()
    help_text_vehicle = (
        "Vehicle type and optional initial speed in the format: "
        "<vehicle_name>:<initial_speed>. "
        "Example: Car:80 . "
        "Available vehicles: " + ", ".join(vehicle_classes)
    )
    parser.add_argument(
        '--vehicle',
        type=parse_vehicle_arg,
        required=True,
        help=help_text_vehicle
    )

    # Parse time period argument
    parser.add_argument(
        '--minutes',
        type=parse_minutes_arg,
        default=60,
        help="Duration of the generated data in minutes (minimum: 60)."
    )

    # Parse anomalies arguments
    anomaly_classes = get_anomaly_classes()
    help_text_anomalies = (
        "Space-separated list of anomaly types with their probabilities, in the format: "
        "<anomaly_name>:<duration>:<probability>. "
        "Example: WheelSlip:0.01 GPSLoss:0.02 . "                    
        "Available anomalies: " + ", ".join(anomaly_classes) + ". "
        "If no anomalies are passed, a telemetry will not contain distorted data."
    )
    parser.add_argument(
        '--anomalies',
        nargs='+',
        type=parse_anomalies_arg,
        help=help_text_anomalies
    )

    # Parse labels argument
    parser.add_argument('--labels', action='store_true', help='Enable anomaly labels.')


    # Parse output file argument
    parser.add_argument(
        "--output",
        type=str,
        help='Path to the CSV file where the dataset will be saved.', 
        default="synthetic_dataset.csv"
    )
    
    args = parser.parse_args()

    # Instantiate vehicle type
    vehicle_name, init_speed = args.vehicle
    vehicle_classes = get_vehicle_classes()
    vehicle = vehicle_classes[vehicle_name](initial_speed=init_speed)

    # Instantiate telemetry
    telemetry = Telemetry(vehicle)

    # Create synthetic data
    telemetry_data = generate_clean_telemetry_data(telemetry=telemetry, minutes=args.minutes)

    # Label data
    include_labels =  args.labels

    # Introduce anomalies
    if args.anomalies:
        anomaly_classes = get_anomaly_classes()

        anomalies = [
            anomaly_classes[anomaly](duration=duration, probability=probability)
            for anomaly, duration, probability in args.anomalies
        ]
        # Modify the data in place
        introduce_anomalies(telemetry_data, anomalies, include_labels)

    # Create csv file
    create_file_csv(telemetry_data, output_file=args.output)
    

if __name__ == "__main__":
    main()
