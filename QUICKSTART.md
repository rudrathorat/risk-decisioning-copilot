# Quick Start Guide

## Setup

```bash
# 1. Navigate to project directory
cd risk-decisioning-copilot

# 2. Create and activate virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # macOS/Linux
# OR
venv\Scripts\activate     # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

## Running the Project

**Option 1: Run Python Script (Quick Execution)**
```bash
python run.py
```
Outputs results to `results/` folder with CSV files and metrics.

**Option 2: Run Notebook (Interactive with Visualizations)**
```bash
jupyter lab notebooks/02_enhanced_credit_risk_model.ipynb
```
Or open in VS Code/Cursor and run cells interactively.

**Option 3: Use Custom Modules**
```python
from src.feature_engineering import FeatureEngineer
from src.models import ModelTrainer
# ... use the modules
```

## Verify Installation

```python
import pandas as pd
import numpy as np
from src.feature_engineering import FeatureEngineer
print("✓ Setup complete")
```

## Troubleshooting

**Import errors**: Ensure virtual environment is activated and dependencies installed.

**XGBoost errors**: XGBoost is optional. The project works without it. If you need it on Mac, install OpenMP: `brew install libomp`

**SHAP installation**: `pip install shap --no-cache-dir`

**Jupyter not found**: `pip install jupyter jupyterlab`
