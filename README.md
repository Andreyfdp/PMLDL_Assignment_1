# PMLDL Assignment 1: Deployment

A simple automated MLOps pipeline for Palmer Penguins species classification.

The project consists of three stages:

1. Data Engineering
2. Model Engineering
3. Deployment

The pipeline is managed with DVC and automatically checked every 5 minutes.

## Technologies

- Python
- Pandas
- Scikit-learn
- DVC
- ClearML
- FastAPI
- Streamlit
- Docker
- Docker Compose

## Dataset

The project uses the Palmer Penguins dataset.

Target:

`species`

Classes:

- Adelie
- Chinstrap
- Gentoo

Features:

- bill_length_mm
- bill_depth_mm
- flipper_length_mm
- body_mass_g
- island
- sex

## Pipeline

```text
prepare
   |
   v
train
   |
   v
deploy
```

### Stage 1: Data Engineering

The `prepare` stage:

- loads the raw dataset;
- handles missing values;
- removes duplicates;
- removes outliers using the IQR method;
- removes unused columns;
- splits the data into training and testing datasets.

Outputs:

```text
data/processed/train.csv
data/processed/test.csv
```

### Stage 2: Model Engineering

The `train` stage performs feature engineering and model training.

Numerical features are processed using `StandardScaler`.

Categorical features are processed using `OneHotEncoder`.

The model is Logistic Regression.

Parameters:

```text
C = 1.0
class_weight = balanced
max_iter = 2000
random_state = 42
```

The following test metrics are calculated:

- Accuracy
- Precision Macro
- Recall Macro
- F1 Macro

Training experiments and metrics are logged with ClearML.

Outputs:

```text
models/penguins_model.joblib
metrics/metrics.json
```

### Stage 3: Deployment

The trained model is deployed using two separate Docker containers.

The first container runs the FastAPI model API.

The second container runs the Streamlit web application.

The Streamlit application sends prediction requests to the FastAPI service.

## Installation

Create a virtual environment:

```bash
python -m venv .venv
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## ClearML Setup

Configure ClearML credentials:

```bash
clearml-init
```

## Run the Pipeline

Run the complete DVC pipeline:

```bash
dvc repro
```

View the pipeline graph:

```bash
dvc dag
```

DVC executes only the stages whose dependencies have changed.

## Automatic Execution

Run the scheduler:

```bash
python scheduler.py
```

The scheduler runs:

```bash
dvc repro
```

every 5 minutes.

If nothing has changed, DVC skips all unnecessary stages.

## Web Application

Open:

```text
http://localhost:8501
```

Enter penguin features and press `Predict`.

The application displays the predicted species and class probabilities.

## API

FastAPI documentation is available at:

```text
http://localhost:8000/docs
```

Available endpoints:

```text
GET /health
POST /predict
```

## Docker

Start the API and application manually:

```bash
docker compose -f code/deployment/docker-compose.yml up -d --build
```

Check running containers:

```bash
docker compose -f code/deployment/docker-compose.yml ps
```

Stop the containers:

```bash
docker compose -f code/deployment/docker-compose.yml down
```

## Project Structure

```text
PMLDL_Assignment_1/
|
├── code/
│   ├── datasets/
│   │   └── prepare_data.py
│   ├── models/
│   │   └── train_model.py
│   └── deployment/
│       ├── api/
│       │   ├── main.py
│       │   └── Dockerfile
│       ├── app/
│       │   ├── app.py
│       │   └── Dockerfile
│       └── docker-compose.yml
|
├── data/
│   ├── raw/
│   │   └── penguins.csv
│   └── processed/
|
├── models/
├── metrics/
|
├── dvc.yaml
├── dvc.lock
├── scheduler.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Application Workflow

```text
User
 |
 v
Streamlit
 |
 | POST /predict
 v
FastAPI
 |
 v
Logistic Regression
 |
 v
Prediction
 |
 v
Streamlit
```