import unittest
from unittest.mock import patch
from src.core.anomalies import Anomaly, WheelSlip, GPSLoss, SpeedSensorFreeze, get_anomaly_classes


class TestAnomalies(unittest.TestCase):
    def setUp(self):
        self.data_point = {
            "wheel_rpm": 100,
            "distance": 1000,
            "speed": 60
        }

    @patch("random.random", return_value=0.001)
    def test_should_apply_true_when_random_triggered(self, mock_random):
        anomaly = WheelSlip(duration=2, probability=0.01)  # per-second prob = 0.005
        anomaly.active_counter = 0  # ensures random logic is used
        self.assertTrue(anomaly.should_apply())
        mock_random.assert_called_once()

    def test_should_apply_true_when_active_counter_positive(self):
        anomaly = GPSLoss(duration=5, probability=0.1)
        anomaly.active_counter = 3
        self.assertTrue(anomaly.should_apply())

    def test_get_anomaly_classes(self):
        classes = get_anomaly_classes()
        self.assertIn("WheelSlip", classes)
        self.assertIn("GPSLoss", classes)
        self.assertIn("SpeedSensorFreeze", classes)
        self.assertTrue(issubclass(classes["WheelSlip"], Anomaly))

    def test_wheel_slip_apply(self):
        anomaly = WheelSlip(duration=2, probability=0.01)
        anomaly.active_counter = 0
        result = anomaly.apply(self.data_point.copy())
        self.assertEqual(result["wheel_rpm"], 200)
        self.assertEqual(anomaly.active_counter, 1)

    def test_gps_loss_apply(self):
        anomaly = GPSLoss(duration=3, probability=0.05)
        anomaly.active_counter = 0
        result = anomaly.apply(self.data_point.copy())
        self.assertEqual(result["distance"], 0)
        self.assertEqual(anomaly.active_counter, 2)

    def test_speed_sensor_freeze_apply(self):
        anomaly = SpeedSensorFreeze(duration=4, probability=0.02)
        anomaly.active_counter = 0
        result = anomaly.apply(self.data_point.copy())
        self.assertEqual(result["speed"], 60)
        self.assertEqual(anomaly.frozen_speed, 60)
        self.assertEqual(anomaly.active_counter, 3)

        # Apply again to simulate frozen speed
        new_data = {"speed": 999}
        result = anomaly.apply(new_data)
        self.assertEqual(result["speed"], 60)
        self.assertEqual(anomaly.active_counter, 2)

    def test_active_counter_decrement_and_transformation(self):
        anomaly = WheelSlip(duration=2, probability=0.01)
        anomaly.active_counter = 2
        data = self.data_point.copy()

        # Apply twice
        data = anomaly.apply(data)
        self.assertEqual(data["wheel_rpm"], 200)
        self.assertEqual(anomaly.active_counter, 1)

        data = anomaly.apply(data)
        self.assertEqual(data["wheel_rpm"], 400)  # Doubled again
        self.assertEqual(anomaly.active_counter, 0)


if __name__ == "__main__":
    unittest.main()
