import argparse

from src.core.detector import AnomalyDetector 


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

    detector = AnomalyDetector()

    if args.output:
        detector.create_file(args.input, args.output)
    else:
        output = detector.anomaly_intervals(args.input)
        # Explicit print for CLI
        print(output)


if __name__ == "__main__":
    main()