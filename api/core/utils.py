import csv
import io
import pandas as pd
from typing import List, Dict
from functools import wraps


def create_csv_in_memory(data: List[Dict]) -> str:
    """Generate CSV content as a string from list of dicts."""
    output = io.StringIO()
    if not data:
        return ""
    
    writer = csv.DictWriter(output, fieldnames=data[0].keys())
    writer.writeheader()
    writer.writerows(data)
    
    return output.getvalue()


def return_csv_buffer(func):
    """
    Decorator that converts a pandas DataFrame returned by a function
    into an in-memory CSV buffer (StringIO).
    """
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        df = func(*args, **kwargs)
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Expected the function to return a DataFrame.")
        buffer = io.StringIO()
        df.to_csv(buffer, index=False)
        buffer.seek(0)
        return buffer
    return wrapper