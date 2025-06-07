from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import Optional

from api.core.schemas import CreateSyntheticTelemetry
from api.core.services.anomaly_detector import handle_anomaly_detection
from api.core.services.dataset_generator import generate_dataset_response

from src.core.config import get_config


router = APIRouter()


config = get_config()


@router.post("/generate_dataset/", 
             summary=config['endpoint_generate_dataset']['summary'],
             description=config['endpoint_generate_dataset']['description']
)
def generate_dataset(data: CreateSyntheticTelemetry):
    return generate_dataset_response(data)


@router.post("/anomaly_detector/",
             summary=config['endpoint_anomaly_detector']['summary'],
             description=config['endpoint_anomaly_detector']['description']
)
async def anomaly_detector(
    file: UploadFile = File(...),
    output: Optional[str] = Form(None)
):
    try:
        return await handle_anomaly_detection(file, output)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))