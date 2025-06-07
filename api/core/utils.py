import csv
import io
from typing import List, Dict


def create_csv_in_memory(data: List[Dict]) -> str:
    """Generate CSV content as a string from list of dicts."""
    output = io.StringIO()
    if not data:
        return ""
    
    writer = csv.DictWriter(output, fieldnames=data[0].keys())
    writer.writeheader()
    writer.writerows(data)
    
    return output.getvalue()