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
    
    # Passed arguments
    args = parser.parse_args()
    input_file = args.input
    output_file = args.output

    config = get_config()
    model_path = config['paths']['model_save_path']
    batch_size = config['inference']['batch_size']
    device = config['inference']['device']

    detector = AnomalyDetector(model_path, batch_size, device)

    _, _, _, dataframe, anomaly_intervals = detector.inference(input_file)

    # Check if an output file path is provided
    if output_file:
        # If output_file is provided, save the DataFrame to a CSV file
        dataframe.to_csv(output_file, index=False)
        print(f'Saved CSV file to {output_file}')
    
    else:
        # If no output_file is provided, print the anomaly_intervals instead
        print(anomaly_intervals)


if __name__ == "__main__":
    main()