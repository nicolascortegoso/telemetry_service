
# Поиск аномалий в телеметрии транспортного средства


## Исходные параметры

Телеметрия состоит из трёх измеряемых признаков:
- wheel_rpm — число оборотов колеса в минуту.
- speed — скорость движения (км/ч).
- distance — пройденное расстояние по GPS (в метрах).

Значения симулируются с шагом в 1 секунду.
wheel_rpm и speed снимаются с телеметрии автомобиля. distance снимается независимо с GPS треккера.

## 1. Генератор синтетических телеметрических данных

Этот скрипт генерирует синтетические телеметрические данные для указанного типа транспортного средства на заданный промежуток времени.
По желанию могут быть добавлены аномалии (например, пробуксовка колёс, потеря сигнала GPS) с заданной пользователем вероятностью и продолжительностью.
Итоговый набор данных сохраняется в файл CSV с возможностью добавления меток аномалий.


Пример использования:


```
usage: python src/generate_dataset.py [-h] --vehicle VEHICLE [--minutes MINUTES] [--anomalies ANOMALIES [ANOMALIES ...]] [--labels]
                           [--output OUTPUT]

Produces synthetic telemetry data for a type of vehicle.

options:
  -h, --help            show this help message and exit
  --vehicle VEHICLE     Vehicle type and optional initial speed in the format: <vehicle_name>:<initial_speed>. Example:
                        Car:80 . Available vehicles: Car, Truck, ElectricVehicle (default: None)
  --minutes MINUTES     Duration of the generated data in minutes (minimum: 60). (default: 60)
  --anomalies ANOMALIES [ANOMALIES ...]
                        Space-separated list of anomaly types with their probabilities, in the format:
                        <anomaly_name>:<duration>:<probability>. Example: WheelSlip:0.01 GPSLoss:0.02 . Available
                        anomalies: WheelSlip, GPSLoss, SpeedSensorFreeze. If no anomalies are passed, a telemetry will
                        not contain distorted data. (default: None)
  --labels              Enable anomaly labels. (default: False)
  --output OUTPUT       Path to the CSV file where the dataset will be saved. (default: synthetic_dataset.csv)
```