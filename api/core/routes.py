from fastapi import APIRouter, Response, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse

from typing import Optional
from io import BytesIO

from src.core.vehicles import get_vehicle_classes
from src.core.anomalies import get_anomaly_classes
from src.core.telemetry import Telemetry
from src.core.generator import generate_clean_telemetry_data
from src.core.detector import AnomalyDetector
from src.core.utils import introduce_anomalies

from api.core.models import CreateSyntheticTelemetry
from api.core.utils import create_csv_in_memory, return_csv_buffer


router = APIRouter()


@router.post("/generate_dataset/", summary="Synthetic dataset creation")
def generate_dataset(data: CreateSyntheticTelemetry):

    # Instantiate vehicle type
    vehicle_type = data.vehicle.type
    initial_speed = data.vehicle.initial_speed
    vehicle_classes = get_vehicle_classes()
    vehicle = vehicle_classes[vehicle_type](initial_speed=initial_speed)

    # Instantiate telemetry
    telemetry = Telemetry(vehicle)

    # Create synthetic data
    minutes = data.minutes
    telemetry_data = generate_clean_telemetry_data(telemetry=telemetry, minutes=minutes)

    # Label data
    include_labels = data.labels

    # Introduce anomalies
    if data.anomalies:
        anomaly_classes = get_anomaly_classes()

        anomalies = [
            anomaly_classes[anomaly.type](duration=anomaly.duration, probability=anomaly.probability)
            for anomaly in data.anomalies
        ]
        # Modify the data in place
        introduce_anomalies(telemetry_data, anomalies, include_labels)

    # Create csv file
    csv_content = create_csv_in_memory(telemetry_data)
    
    headers = {
        'Content-Disposition': f'attachment; filename="{data.output}"',
        'Content-Type': 'text/csv',
    }
    
    return Response(content=csv_content, headers=headers, media_type='text/csv')


@router.post("/anomaly_detector/", summary="Anomaly detection")
async def anomaly_detector(
    file: UploadFile = File(...),
    output: Optional[str] = Form(None)
):

    # Validate only if output is not None
    if output is not None and not output.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Output filename must end with .csv")
    
    # Read uploaded file into a temporary file
    contents = await file.read()
    input_buffer = BytesIO(contents)

    # Instantiate the anomaly detector
    detector = AnomalyDetector()

    if output:
        # Use decorated version of the method
        buffer = return_csv_buffer(detector.process_csv_file)(input_buffer)
        return StreamingResponse(
            buffer,
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={output}"}
        )
    else:
        output = detector.anomaly_intervals(input_buffer)        
        return {"anomaly_intervals": output}