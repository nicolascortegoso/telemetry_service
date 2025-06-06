import argparse

from src.core.detector import AnomalyDetector 
from src.core.config import get_config


def main():
    """
    Command-line interface for detecting anomalies in a multivariate time series.

    This function parses command-line arguments to read a CSV file containing 
    time series data, runs an anomaly detection model on it, and optionally 
    saves the output to a file.
    """

    parser = argparse.ArgumentParser(
        description="Analyses a time series.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Parse output file argument
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help='Path to the CSV file containing the time series.'
    )

    # Parse output file argument
    parser.add_argument(
        "--output",
        type=str,
        help='Path to the file where the predictions results will be saved.', 
        default=None
    )
    
    args = parser.parse_args()

    # Load configuration from environment variables
    config = get_config()
    input_dim = config['model']['input_dim']
    hidden_dim = config['model']['hidden_dim']
    num_layers = config['model']['num_layers']
    model_weights_path = config['paths']['model_weights']
    scaler_path = config['paths']['scaler']
    threshold_path = config['paths']['threshold']
    threshold_coefficient = config['threshold']['coefficient']
    window_size = config['model']['window_size']
    device = config['inference']['device']

    detector = AnomalyDetector(
        input_dim,
        hidden_dim,
        num_layers,
        model_weights_path,
        scaler_path,
        threshold_path,
        threshold_coefficient,
        window_size,
        device
    )

    if args.output:
        detector.create_file(args.input, args.output)
    else:
        output = detector.anomaly_intervals(args.input)
        # Explicit print for CLI
        print(output)


if __name__ == "__main__":
    main()