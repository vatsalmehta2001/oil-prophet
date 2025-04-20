# Oil Prophet API

A RESTful API for the Oil Prophet forecasting system that provides oil price predictions with sentiment analysis integration.

## Features

- **Oil Price Forecasting**: Get price predictions for Brent and WTI oil using multiple models
- **Sentiment Analysis**: Analyze market sentiment from Reddit discussions about oil
- **Model Performance Metrics**: Compare different forecasting models
- **Pipeline Execution**: Run the complete forecasting pipeline programmatically
- **Flexible Data Options**: Support for daily, weekly, and monthly data frequencies
- **Secure Access**: API key authentication for endpoint protection

## Installation

### Prerequisites

- Python 3.8+ 
- TensorFlow 2.9+
- PyTorch 1.11+
- FastAPI
- All dependencies listed in requirements.txt

### Option 1: Standard Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/oil-prophet.git
   cd oil-prophet
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the API server:
   ```bash
   uvicorn api:app --host 0.0.0.0 --port 8000
   ```

### Option 2: Docker Installation

1. Build and run with Docker Compose:
   ```bash
   docker-compose up -d
   ```

2. Set API key through environment variable:
   ```bash
   export OIL_PROPHET_API_KEY=your_secure_key_here
   docker-compose up -d
   ```

## API Endpoints

### Forecast Generation

`POST /api/v1/forecast`

Generate oil price forecasts using specified model and parameters.

**Request Body**:
```json
{
  "oil_type": "brent",
  "frequency": "daily",
  "model": "ensemble",
  "forecast_horizon": 14,
  "include_history": true,
  "history_periods": 30
}
```

### Sentiment Analysis

`GET /api/v1/sentiment?timeframe=30d&include_data=false`

Get market sentiment analysis for oil-related discussions.

### Model Performance

`GET /api/v1/models/performance?oil_type=brent&frequency=daily`

Get performance metrics comparing different forecasting models.

### Available Models

`GET /api/v1/models`

List all available forecasting models.

### Pipeline Execution

`POST /api/v1/pipeline/run`

Run the forecast pipeline in the background.

## Authentication

All API endpoints require API key authentication. Include the API key in the request header:

```
X-API-Key: your_api_key_here
```

The API key can be set as an environment variable:

```bash
export OIL_PROPHET_API_KEY=your_secure_key_here
```

## Documentation

Interactive API documentation is available at `/docs` when the server is running.

## Usage Examples

### Python Client

```python
import requests
import json

API_URL = "http://localhost:8000"
API_KEY = "your_api_key_here"
HEADERS = {
    "X-API-Key": API_KEY,
    "Content-Type": "application/json"
}

# Generate a forecast
forecast_req = {
    "oil_type": "brent",
    "frequency": "daily",
    "model": "sentiment",
    "forecast_horizon": 7,
    "include_history": True,
    "history_periods": 30
}

response = requests.post(
    f"{API_URL}/api/v1/forecast", 
    headers=HEADERS,
    json=forecast_req
)

# Print forecast results
forecast_data = response.json()
print(json.dumps(forecast_data, indent=2))
```

### cURL Example

```bash
# Generate a forecast
curl -X POST "http://localhost:8000/api/v1/forecast" \
  -H "X-API-Key: your_api_key_here" \
  -H "Content-Type: application/json" \
  -d '{
    "oil_type": "brent",
    "frequency": "daily",
    "model": "ensemble",
    "forecast_horizon": 7
  }'
```

## Error Handling

The API uses standard HTTP status codes:

- `200 OK`: Request succeeded
- `202 Accepted`: Request accepted for background processing
- `400 Bad Request`: Invalid parameters
- `401 Unauthorized`: Missing or invalid API key
- `404 Not Found`: Resource not found
- `500 Internal Server Error`: Server error

Error responses include a JSON object with error details:

```json
{
  "error": "Error message",
  "code": 400,
  "details": "Additional information about the error"
}
```

## License

This project is licensed under the MIT License