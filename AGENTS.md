# AGENTS.md

Rossmann Store Sales forecasting (Kaggle competition). All work lives in Jupyter notebooks under `notebooks/`; there is no src package, no manifest (`pyproject.toml`/`requirements.txt`), and no tests.

## Critical: how the code runs
- Notebooks are written for **Google Colab**, not local execution. Each mounts Drive:
  `drive.mount('/content/drive')` and reads data from
  `/content/drive/MyDrive/data/store-sales-forecasting/train.csv`.
- The `data/` directory is **intentionally empty** (gitignored). The real dataset is only on the author's Colab Drive. Running any notebook locally will fail at the read step unless that file exists.
- Don't expect the notebooks to be runnable in this repo as-is; they are a record of analysis, not an executable pipeline.

## Dependencies (Colab-installed, not declared anywhere)
`pandas`, `matplotlib`, `seaborn`, `numpy`, `sklearn`, `xgboost`, `mlflow`, `dagshub`.
Notebooks `%pip install` (`xgboost`, `dagshub`, `mlflow`) inside cells. There is no `pyproject.toml`/`requirements.txt` to keep in sync.

## Experiment tracking
Models log metrics/params to **MLflow via DagsHub** (`dagshub.init(repo_owner='PigStep', repo_name='store-sales-forecast', mlflow=True)` in `Baseline.ipynb`). The GitHub remote is also `PigStep/store-sales-forecast`. Requires the author's DagsHub credentials to fully run; keep the init call intact.

## Domain/data gotchas
- Sales data spans 1,115 stores; forecasting horizon is ~6 weeks.
- `Sales` is the target. Some rows have `Sales == 0` (e.g. stores closed Sundays, some Sunday hours). New/open labeling matters — not all stores are closed Sundays.
- Preprocessing convention in `EDA.ipynb`: set `Date` as index, `pd.to_datetime`.
- `Baseline.ipynb` builds features (`create_basic_features`: DayOfWeek, Month, Year, Open, Promo, StateHoliday, etc., `FEAT_COLS`) and adds lag features (`add_lag_features`), then trains `XGBRegressor` with `TimeSeriesSplit` CV. Follow this pattern for new modeling work.

## Content per notebook
- `EDA.ipynb` — distribution/seasonality exploration.
- `Closed_shop_analysis.ipynb` — stores temporarily closed for refurbishment; Sunday-opening store analysis.
- `Baseline.ipynb` — XGB baseline + lag features + MLflow/DagsHub logging.

## Conventions
- Write analysis as notebook markdown cells with plain prose around plots (see existing notebooks for the established style).
- Git history uses `feat:`/`ref:`/`fix:` conventional prefixes; single `main` branch.
