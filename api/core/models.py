from pydantic import BaseModel, validator, conint
from typing import List, Optional

from src.core.vehicles import get_vehicle_classes
from src.core.anomalies import get_anomaly_classes


class Vehicle(BaseModel):
    type: str
    initial_speed: int

    @validator("type")
    def type_must_be_allowed(cls, v):
        allowed = get_vehicle_classes()
        if v not in allowed:
            raise ValueError(
                f"Invalid vehicle type: {v}. Available types of vehicle {', '.join(get_vehicle_classes().keys())}"
            )
        return v


class Anomaly(BaseModel):
    type: str
    duration: int
    probability: float

    @validator("type")
    def type_must_be_allowed(cls, v):
        allowed = get_anomaly_classes()
        if v not in allowed:
            raise ValueError(
                f"Invalid anomaly type: {v}. Available types of anomaly {', '.join(get_anomaly_classes().keys())}"
            )
        return v


class CreateSyntheticTelemetry(BaseModel):
    minutes: conint(gt=59) = 60                         # default 60, must be > 59
    vehicle: Vehicle
    anomalies: Optional[List[Anomaly]] = None           # Optional field with default None
    labels: Optional[bool] = None                       # Optional field with default None
    output: Optional[str] = "synthetic_dataset.csv"     # Optional field with default filename
