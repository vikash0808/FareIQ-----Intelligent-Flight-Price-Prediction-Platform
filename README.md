# ✈️ FareIQ — Intelligent Flight Price Prediction Platform

**FareIQ** is a machine learning-powered platform that predicts Indian domestic flight fares using XGBoost, optimized hyperparameters, and historical flight data. It combines a high-performance FastAPI backend with an interactive Streamlit interface to deliver fast and accessible flight price predictions.

## 🚀 Key Features

- **High-Accuracy Predictions:** Achieved an R² score of **97.35%** with a Mean Absolute Error (MAE) of ₹2,084 on the test set.
- **Large-Scale Dataset:** Trained on **300,000+ Indian domestic flight records**.
- **Hyperparameter Optimization:** Used Optuna to optimize XGBoost and improve performance over a Random Forest baseline.
- **Fast Inference:** Loads serialized model artifacts into memory during FastAPI startup for sub-100ms predictions under the tested conditions.
- **Interactive Interface:** Provides a user-friendly Streamlit interface for entering flight details and viewing predicted fares.
- **Cloud Deployment:** Deployed on Hugging Face Spaces.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| XGBoost | Flight fare prediction model |
| Optuna | Hyperparameter optimization |
| FastAPI | REST API and model inference |
| Streamlit | Interactive frontend |
| Pandas & NumPy | Data processing |
| Scikit-learn | Model evaluation and baseline comparison |
| Joblib / Pickle | Model serialization |
| Hugging Face Spaces | Deployment and hosting |

## 📊 Model Performance

| Metric | Result |
|---|---:|
| Training dataset | 300,000+ records |
| R² Score | 97.35% |
| Mean Absolute Error (MAE) | ₹2,084 |
| Prediction latency | Under 100 ms |
| Baseline model | Random Forest |

The optimized XGBoost model outperformed the Random Forest baseline by **1.13 percentage points in R²** and reduced MAE by **₹363**, based on the reported evaluation results.

*Note: Performance metrics are based on the project's test set and measured deployment conditions. Actual results may vary with input data and runtime environment.*

## 🏗️ System Architecture

```text
             User
              |
              v
     Streamlit Interface
              |
              v
        FastAPI Backend
              |
              v
     Feature Preprocessing
              |
              v
      Trained XGBoost Model
       (Loaded into RAM)
              |
              v
      Predicted Flight Fare
              |
              v
     Result to the User
```

### How It Works

1. **Data Processing:** Flight records are cleaned and transformed into features suitable for machine learning.
2. **Model Training:** XGBoost learns patterns from historical flight fares.
3. **Hyperparameter Tuning:** Optuna searches for effective model configurations.
4. **Model Evaluation:** Performance is evaluated using R² and MAE, with Random Forest as the baseline.
5. **Model Serialization:** The trained model and required preprocessing artifacts are saved for inference.
6. **API Initialization:** FastAPI loads the saved artifacts into memory at startup.
7. **Prediction:** The API processes input features and returns the predicted fare.
8. **Deployment:** The application is hosted on Hugging Face Spaces.

## 💻 Getting Started

### Prerequisites

- Python 3.10 or a compatible version
- pip
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/FareIQ.git
cd FareIQ
```

Replace `YOUR_USERNAME` with your GitHub username.

### 2. Create a Virtual Environment

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

If your repository contains a `requirements.txt` file:

```bash
pip install -r requirements.txt
```

Otherwise, install the core dependencies:

```bash
pip install pandas numpy scikit-learn xgboost optuna fastapi uvicorn streamlit joblib
```

### 4. Configure Model Artifacts

Ensure that the trained XGBoost model and any required preprocessing objects are available at the file paths expected by your application.

For example:

```text
FareIQ/
├── app.py
├── main.py
├── requirements.txt
├── models/
│   └── xgboost_model.pkl
└── README.md
```

*The filenames above are illustrative. Adjust them to match your actual repository structure.*

### 5. Run the FastAPI Backend

If your FastAPI application is defined in `main.py` as `app`:

```bash
uvicorn main:app --reload
```

Open the interactive API documentation at:

```text
http://127.0.0.1:8000/docs
```

### 6. Run the Streamlit Interface

If your Streamlit application is defined in `app.py`:

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal to access the application.

## 📁 Project Structure

An example structure for the project is:

```text
FareIQ/
├── app.py                  # Streamlit frontend
├── main.py                 # FastAPI backend
├── requirements.txt        # Python dependencies
├── models/                 # Saved model artifacts
├── notebooks/              # Data analysis and experiments
├── data/                   # Dataset (if distributable)
└── README.md               # Project documentation
```

Your actual folder structure may differ. Update this section to reflect the files present in your repository.

## 🔬 Future Improvements

- Integrate live flight fare data from supported aviation or booking APIs.
- Add historical fare trends and price comparison dashboards.
- Expand coverage to additional routes and airlines.
- Introduce prediction intervals to communicate model uncertainty.
- Add automated model retraining and monitoring.
- Containerize the application with Docker.

## 🌐 Live Demo

**Hugging Face Spaces:** [Add your deployed FareIQ URL here](https://huggingface.co/spaces)

**GitHub Repository:** [Add your repository URL here](https://github.com/)

## 👨‍💻 Author

**Vikash Dhillan**

B.Tech Computer Science
