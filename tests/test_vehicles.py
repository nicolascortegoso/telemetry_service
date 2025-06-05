import unittest
from src.core.vehicles import Vehicle, Car, Truck, ElectricVehicle, get_vehicle_classes

class TestVehicleBase(unittest.TestCase):

    def test_update_speed_within_bounds(self):
        car = Car(initial_speed=80.0)
        for _ in range(100):
            old_speed = car.speed
            car.update_speed()
            self.assertGreaterEqual(car.speed, car.min_speed)
            self.assertLessEqual(car.speed, car.max_speed)
            # Optionally check the delta is not too large
            self.assertLessEqual(abs(car.speed - old_speed), car.speed_variation)

class TestVehicleSubclasses(unittest.TestCase):

    def test_car_initialization(self):
        car = Car()
        self.assertEqual(car.speed, 80.0)
        self.assertEqual(car.min_speed, 0.0)
        self.assertEqual(car.max_speed, 140.0)
        self.assertEqual(car.speed_variation, 2.0)
        self.assertEqual(car.wheel_diameter, 0.65)

    def test_truck_initialization(self):
        truck = Truck()
        self.assertEqual(truck.speed, 60.0)
        self.assertEqual(truck.max_speed, 100.0)
        self.assertEqual(truck.speed_variation, 1.5)
        self.assertEqual(truck.wheel_diameter, 1.0)

    def test_electric_vehicle_initialization(self):
        ev = ElectricVehicle()
        self.assertEqual(ev.speed, 90.0)
        self.assertEqual(ev.max_speed, 160.0)
        self.assertEqual(ev.speed_variation, 2.5)
        self.assertEqual(ev.wheel_diameter, 0.7)

    def test_get_vehicle_classes(self):
        classes = get_vehicle_classes()
        self.assertIn("Car", classes)
        self.assertIn("Truck", classes)
        self.assertIn("ElectricVehicle", classes)
        self.assertTrue(issubclass(classes["Car"], Vehicle))


if __name__ == '__main__':
    unittest.main()
