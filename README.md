# ITU BDS MLOps'25 - Exam Project

## Group Members

- Dmytro Stavnychyi
- Vitus Andreas Moustgaard
- Ada Yıldız

## Project Description

In this project, we have restructured machine learning code to adhere to concepts of MLOps such as:

- **Data versioning** with DVC
- **Experiment tracking** with MLflow
- **Container orchestration** with Dagger (Go)
- **CI/CD automation** with GitHub Actions

The best performing model (Logistic Regression, F1~0.78) is automatically trained, validated, and deployed as a GitHub artifact.

## Project Structure

```
.
├── .github/workflows/       # GitHub Actions CI/CD
│   └── train-model.yml     # Model training workflow
├── mlops_project/          # Main Python package
│   ├── data/              # Data loading and preprocessing
│   │   ├── load.py        # DVC data pulling and loading
│   │   └── preprocess.py  # Data cleaning and preprocessing
│   ├── deployment/
│   │   ├── mlflow_utils.py  # Util functions for mlflow
│   ├── features/          # Feature engineering
│   │   └── build_features.py
│   ├── models/            # Model training and evaluation
│   │   ├── train.py       # XGBoost & Logistic Regression training
│   │   └── evaluate.py    # Model evaluation and comparison
│   └── scripts/
│       └── main.py        # Main pipeline orchestration
│   └── utils/
│       └── helpers.py     # Helpers for numeric columns and missing values
│   └── config.py
├── notebooks/             # Original exploratory notebooks
│   └── artifacts/         # Raw data (DVC tracked)
├── output/                # Generated artifacts (gitignored)
│   ├── model              # Best model (for GitHub workflow)
│   ├── results.json       # Evaluation metrics
│   └── *.pkl              # Serialized models
├── mlruns/                # MLflow experiment tracking
├── dagger-pipeline.go     # Dagger orchestration (Go)
├── requirements.txt       # Python dependencies
├── .dvc/                  # DVC configuration
└── README.md
```

**Note:** The project follows the [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/) template structure. Directories not listed above (`models/`, `tests/`, `reports/`, `references/`, `docs/`) are part of the original template but were not actively used in this implementation, as they were not required for the core MLOps pipeline.

## Prerequisites

- **Python 3.10**
- **Go 1.25**
- **Dagger CLI**
- **DVC**
- **Git**

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Vitus-A-M/MLOps-project.git
cd MLOps-project
```

### 2. Set up Python environment

```bash
python3.10 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Install DVC

```bash
pip install dvc
```

### 4. Pull data

```bash
dvc pull
```

## How to Run

### Option 1: Direct Python Execution

```bash
# Activate virtual environment
source .venv/bin/activate

# Run the pipeline
python mlops_project/scripts/main.py
```

The model artifacts will be saved in `output/`:

- `output/model` - Best model (no extension)
- `output/xgboost_model.pkl` - XGBoost model
- `output/logistic_regression_model.pkl` - Logistic Regression model
- `output/results.json` - Evaluation metrics
- `output/best_model.txt` - Model metadata

### Option 2: Using Dagger (Containerized)

```bash
# Install Dagger CLI
curl -L https://dl.dagger.io/dagger/install.sh | sh

# Run pipeline in container
dagger run go run dagger-pipeline.go
```

### Option 3: Automated with GitHub Actions

Push to `main` branch or any `feature/*` branch:

```bash
git add .
git commit -m "feat: your changes"
git push origin main
```

GitHub Actions will automatically:

1. Pull data with DVC
2. Run Dagger pipeline
3. Train models
4. Upload `model` artifact


## Development Workflow

### Branch Strategy

- `main` - Production-ready code
- `feature/*` - Feature development -`debug/*` - Debugging existing features

### Commit messages

While we don't strictly follow conventional commit prefixes (`feat:`, `fix:`), our messages reasonably describe what was changed.

**Examples from our project:**

- "workflows merged correctly - syntax issue solved"
- "Merge branch 'main' into feature/githubworkflow"
- "train and test worflow seperated."
