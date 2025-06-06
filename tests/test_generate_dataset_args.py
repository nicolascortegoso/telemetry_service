import unittest
from unittest.mock import patch
import argparse
import yaml

from src.generate_dataset import parse_vehicle_arg, parse_minutes_arg, parse_anomalies_arg


class TestArgumentParsing(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Runs once before all test methods
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
        cls.min_minutes = config['timeseries']['min_minutes']

    @patch('src.generate_dataset.get_vehicle_classes')
    def test_parse_vehicle_arg_valid(self, mock_get_vehicles):
        mock_get_vehicles.return_value = {"Car": object(), "Bike": object()}
        result = parse_vehicle_arg("Car:80")
        self.assertEqual(result, ("Car", 80.0))

    @patch('src.generate_dataset.get_vehicle_classes')
    def test_parse_vehicle_arg_invalid_format(self, mock_get_vehicles):
        mock_get_vehicles.return_value = {"Car": object()}
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_vehicle_arg("Car-80")  # wrong delimiter

    @patch('src.generate_dataset.get_vehicle_classes')
    def test_parse_vehicle_arg_unknown_vehicle(self, mock_get_vehicles):
        mock_get_vehicles.return_value = {"Car": object()}
        with self.assertRaises(argparse.ArgumentTypeError) as context:
            parse_vehicle_arg("Plane:100")
        self.assertIn("Unknown vehicle", str(context.exception))

    def test_parse_minutes_arg_valid(self):
        self.assertEqual(parse_minutes_arg(f"{self.min_minutes}"), self.min_minutes)
        self.assertEqual(parse_minutes_arg(f"{self.min_minutes * 2}"), self.min_minutes * 2)

    def test_parse_minutes_arg_invalid_integer(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_minutes_arg("abc")

    def test_parse_minutes_arg_too_small(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_minutes_arg(f"{self.min_minutes - 1}")

    @patch('src.generate_dataset.get_anomaly_classes')
    def test_parse_anomalies_arg_valid(self, mock_get_anomalies):
        mock_get_anomalies.return_value = {"WheelSlip": object()}
        result = parse_anomalies_arg("WheelSlip:10:0.5")
        self.assertEqual(result, ("WheelSlip", 10, 0.5))

    @patch('src.generate_dataset.get_anomaly_classes')
    def test_parse_anomalies_arg_invalid_format(self, mock_get_anomalies):
        mock_get_anomalies.return_value = {"WheelSlip": object()}
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_anomalies_arg("WheelSlip-10-0.5")

    @patch('src.generate_dataset.get_anomaly_classes')
    def test_parse_anomalies_arg_unknown_anomaly(self, mock_get_anomalies):
        mock_get_anomalies.return_value = {"GPSLoss": object()}
        with self.assertRaises(argparse.ArgumentTypeError) as context:
            parse_anomalies_arg("WheelSlip:10:0.5")
        self.assertIn("Unknown anomaly", str(context.exception))

    @patch('src.generate_dataset.get_anomaly_classes')
    def test_parse_anomalies_arg_duration_out_of_range(self, mock_get_anomalies):
        mock_get_anomalies.return_value = {"WheelSlip": object()}
        with self.assertRaises(argparse.ArgumentTypeError) as context:
            parse_anomalies_arg("WheelSlip:0:0.5")
        self.assertIn("duration", str(context.exception))

        with self.assertRaises(argparse.ArgumentTypeError) as context:
            parse_anomalies_arg("WheelSlip:61:0.5")
        self.assertIn("duration", str(context.exception))

    @patch('src.generate_dataset.get_anomaly_classes')
    def test_parse_anomalies_arg_probability_out_of_range(self, mock_get_anomalies):
        mock_get_anomalies.return_value = {"WheelSlip": object()}
        with self.assertRaises(argparse.ArgumentTypeError) as context:
            parse_anomalies_arg("WheelSlip:10:-0.1")
        self.assertIn("Probability", str(context.exception))

        with self.assertRaises(argparse.ArgumentTypeError) as context:
            parse_anomalies_arg("WheelSlip:10:1.1")
        self.assertIn("Probability", str(context.exception))


if __name__ == "__main__":
    unittest.main()
