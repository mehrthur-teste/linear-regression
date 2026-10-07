# Linear Regression Starter

A small Python project that trains a linear regression model, saves it to disk, and loads it later to make predictions for new input values.

> **Note:** The current training data is for demonstration only. It follows the artificial relationship `y = 2x + 1`, so its predictions are not meaningful for a real-world problem. Replace the sample `X` and `y` values in `linear_regression.py` with representative real data and train the model again.

## Requirements

- Python 3
- The packages listed in `requirements.txt`

## Setup

From the project directory, create and activate a virtual environment, then install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Whenever you open a new Terminal, activate the environment again before running the scripts:

```bash
source .venv/bin/activate
```

## Train and save the model

Run:

```bash
python linear_regression.py
```

The script splits the sample data into training and test sets, trains a `LinearRegression` model, prints evaluation metrics (MAE, MSE, and R²), and saves the trained model as `linear_regression.joblib`.

To train on your own data, replace `X` and `y` near the top of `linear_regression.py`:

- `X` contains the input feature values. It must be a two-dimensional array, with one row per example.
- `y` contains the target values to predict, with one value for each row in `X`.

For example, if `X` represents house areas and `y` represents prices, keep the same units and row order in both arrays. After changing the data, run the training script again to create an updated model file.

## Make a prediction

After training, run:

```bash
python predict.py
```

Enter a new input value when prompted. The script loads `linear_regression.joblib` and prints the model's prediction. The input must use the same feature and units as the training data.

Run `linear_regression.py` before `predict.py` if the model file does not exist yet.

## Project files

- `linear_regression.py` — trains, evaluates, and saves the model.
- `predict.py` — loads the saved model and predicts a target for a new input.
- `requirements.txt` — lists the Python package dependencies.
- `linear_regression.joblib` — generated model file; recreated when the training script runs.
