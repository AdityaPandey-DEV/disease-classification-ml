# Disease Classification ML Comparison

This project demonstrates the application of various machine learning algorithms to classify different diseases, including diabetes, brain stroke, and heart disease. It compares algorithm performances to provide insights into which models are most effective for different medical conditions.

## Project Structure

```
.
├── backend/
│   ├── data/           # Dataset storage
│   ├── models/         # ML model implementations
│   └── app.py          # Flask API server
└── frontend/
    ├── public/         # Static files
    └── src/            # React source code
```

## Features

- Analysis of multiple disease datasets (diabetes, brain stroke, heart disease)
- Comparison of different ML algorithms:
  - Logistic Regression
  - Random Forest
  - Support Vector Machine (SVM)
  - Neural Network
- Interactive visualization of model performance metrics
- RESTful API for model training and evaluation

## Setup Instructions

### Backend Setup

1. Navigate to the backend directory:
   ```
   cd backend
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python3 -m venv venv
   venv\Scripts\activate  # On Windows
   source venv/bin/activate  # On macOS/Linux
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Run the Flask server:
   ```
   python app.py
   ```
   The server will start on http://localhost:5000

### Frontend Setup

1. Navigate to the frontend directory:
   ```
   cd frontend
   ```

2. Install dependencies:
   ```
   npm install
   ```

3. Start the development server:
   ```
   npm start
   ```
   The application will open in your browser at http://localhost:3000

## Usage

1. Select one or more diseases to analyze
2. Choose the machine learning algorithms to compare
3. Click "Train & Compare" to start the analysis
4. View the results in the visualization dashboard

## API Endpoints

- `GET /api/diseases` - List available diseases
- `GET /api/algorithms` - List available algorithms
- `POST /api/train` - Train models on a single disease
- `POST /api/compare` - Compare models across multiple diseases 
