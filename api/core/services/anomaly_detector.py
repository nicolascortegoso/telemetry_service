# services/anomaly.py
from fastapi.responses import StreamingResponse
from io import BytesIO, StringIO

from src.core.detector import AnomalyDetector
from src.core.config import get_config

# Load configuration from environment variables
config = get_config()
model_path = config['paths']['model_save_path']
batch_size = config['inference']['batch_size']
device = config['inference']['device']

# Load configuration from environment variables
config = get_config()
model_path = config['paths']['model_save_path']
batch_size = config['inference']['batch_size']
device = config['inference']['device']

detector = AnomalyDetector(model_path, batch_size, device)


async def handle_anomaly_detection(file, output: str):
    if output and not output.endswith(".csv"):
        raise ValueError("Output filename must end with .csv")

    contents = await file.read()
    input_buffer = BytesIO(contents)

    _, _, _, output_df, anomaly_intervals = detector.inference(input_buffer)

    if output:
        buffer = StringIO()
        output_df.to_csv(buffer, index=False)
        buffer.seek(0)
        return StreamingResponse(
            buffer,
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={output}"}
        )
    else:
        return {"anomaly_intervals": anomaly_intervals}
