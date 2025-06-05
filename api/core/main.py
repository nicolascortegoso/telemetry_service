from fastapi import FastAPI
from api.core.routes import router as generate_dataset


app = FastAPI(
    title="Vehicle Telemetry API",
    description="API for generating synthetic telemetry data and detecting anomalies in using an LSTM Autoencoder."
)

# Include routers
app.include_router(generate_dataset, prefix="/api/v1", tags=["Available services"])