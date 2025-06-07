from fastapi import Response

from src.core.vehicles import get_vehicle_classes
from src.core.anomalies import get_anomaly_classes
from src.core.generator import generate_clean_telemetry_data
from src.core.utils import introduce_anomalies
from src.core.telemetry import Telemetry

from api.core.schemas import CreateSyntheticTelemetry
from api.core.utils import create_csv_in_memory


def generate_dataset_response(data: CreateSyntheticTelemetry) -> Response:
    vehicle_type = data.vehicle.type
    initial_speed = data.vehicle.initial_speed

    vehicle_classes = get_vehicle_classes()
    vehicle = vehicle_classes[vehicle_type](initial_speed=initial_speed)

    telemetry = Telemetry(vehicle)
    telemetry_data = generate_clean_telemetry_data(telemetry=telemetry, minutes=data.minutes)

    if data.anomalies:
        anomaly_classes = get_anomaly_classes()
        anomalies = [
            anomaly_classes[anomaly.type](duration=anomaly.duration, probability=anomaly.probability)
            for anomaly in data.anomalies
        ]
        introduce_anomalies(telemetry_data, anomalies, data.labels)

    csv_content = create_csv_in_memory(telemetry_data)

    headers = {
        'Content-Disposition': f'attachment; filename="{data.output}"',
        'Content-Type': 'text/csv',
    }

    return Response(content=csv_content, headers=headers, media_type='text/csv')
