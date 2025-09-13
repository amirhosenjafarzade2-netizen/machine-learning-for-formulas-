# data_utils.py
import pandas as pd
import numpy as np
import structlog
import io
from config import Config
from sklearn.preprocessing import StandardScaler

logger = structlog.get_logger()

class MissingDataError(Exception):
    """Raised when insufficient data is provided."""
    pass

class InvalidPressureError(Exception):
    """Raised when pressure column is invalid."""
    pass

def handle_outliers(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Handle outliers using IQR method."""
    for param in columns:
        if df[param].dtype not in [np.float64, np.int64]:
            logger.warning(f"Skipping outlier handling for non-numeric column {param}")
            continue
        Q1 = df[param].quantile(0.25)
        Q3 = df[param].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[param] = df[param].clip(lower=lower_bound, upper=upper_bound)
        logger.info("Clipped outliers", param=param, lower=lower_bound, upper=upper_bound)
    return df

def impute_missing_values(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Impute missing values with median and drop columns with all NaNs or non-numeric data."""
    valid_columns = []
    for param in columns:
        if df[param].isna().all():
            logger.warning(f"Dropping column {param}: all values are NaN")
            continue
        if df[param].dtype not in [np.float64, np.int64]:
            logger.warning(f"Dropping column {param}: non-numeric data")
            continue
        missing_ratio = df[param].isna().sum() / len(df)
        if missing_ratio > 0.5:
            logger.warning(f"Column {param} has {missing_ratio:.2%} missing values")
        median_value = df[param].median()
        if np.isnan(median_value):
            logger.warning(f"Dropping column {param}: median is NaN")
            continue
        df[param] = df[param].fillna(median_value)
        if df[param].isna().any():
            logger.warning(f"Dropping column {param}: still contains NaNs after imputation")
            continue
        valid_columns.append(param)
    df = df[valid_columns]
    logger.info("Imputed missing values", valid_columns=valid_columns)
    return df

def create_sample_data(rng: np.random.Generator) -> pd.DataFrame:
    """Create sample data with wider P range to ensure balanced regimes."""
    n_samples = 100
    return pd.DataFrame({
        'Bo': 1.2 + rng.normal(0, 0.05, n_samples),
        'Rs': 500 + rng.normal(0, 50, n_samples),
        'P': 1900 + rng.uniform(-400, 400, n_samples),  # Uniform for balanced regimes
        'stock tank oil gravity': 30 + rng.normal(0, 2, n_samples),
        'gas SG': 0.7 + rng.normal(0, 0.02, n_samples),
        't': 150 + rng.normal(0, 5, n_samples)
    })

def load_and_preprocess_data(uploaded_files, config: Config, n_rows: int | None = None, rng: np.random.Generator = None) -> tuple[pd.DataFrame, list[str]]:
    """Load and preprocess data from Excel files or use sample data."""
    if rng is None:
        rng = np.random.default_rng(config.random_seed)
    
    potential_params = [
        'Bo', 'Rs', 'gas SG', 'oil SG', 't', 'corrected gas gravity', 
        'P', 'oil density', 'oil viscosity', 'stock tank oil gravity'
    ]
    
    if not uploaded_files:
        logger.info("No files uploaded. Using sample data.")
        return create_sample_data(rng), potential_params
    
    dfs = []
    for file in uploaded_files:
        try:
            df_temp = pd.read_excel(io.BytesIO(file.getvalue()), sheet_name=0, engine='openpyxl')
            all_columns = set(df_temp.columns)
            params = [p for p in potential_params if p in all_columns]
            if not params:
                logger.warning("No valid parameters found", file=file.name)
                continue
            df_temp = df_temp[params].select_dtypes(include=[np.float64, np.int64])
            if df_temp.empty:
                logger.warning("No numeric columns found", file=file.name)
                continue
            df_temp = handle_outliers(df_temp, df_temp.columns)
            df_temp = impute_missing_values(df_temp, df_temp.columns)
            if df_temp.empty:
                logger.warning("No valid columns after preprocessing", file=file.name)
                continue
            if n_rows is not None and n_rows < len(df_temp):
                df_temp = df_temp.sample(n=n_rows, random_state=rng.integers(0, 2**32))
                logger.info("Sampled rows", file=file.name, n_rows=n_rows)
            dfs.append(df_temp)
        except Exception as e:
            logger.error("Error processing file", file=file.name, error=str(e))
    
    if not dfs:
        logger.info("No valid files loaded. Using sample data.")
        return create_sample_data(rng), potential_params
    
    merged_df = pd.concat(dfs, ignore_index=True)
    if len(merged_df) < config.min_rows:
        raise MissingDataError(f"Merged dataset has {len(merged_df)} rows, but minimum is {config.min_rows}")
    
    if 'oil SG' in merged_df.columns and 'stock tank oil gravity' not in merged_df.columns:
        try:
            merged_df['stock tank oil gravity'] = 141.5 / merged_df['oil SG'] - 131.5
            logger.info("Calculated stock tank oil gravity")
        except Exception as e:
            logger.warning("Failed to calculate stock tank oil gravity", error=str(e))
    
    params = list(merged_df.select_dtypes(include=[np.float64, np.int64]).columns)
    if not params:
        raise MissingDataError("No valid numeric parameters found after preprocessing")
    merged_df = impute_missing_values(merged_df, params)
    logger.info("Detected parameters", params=params)
    return merged_df, params

def classify_regimes(df: pd.DataFrame, pressure_col: str = 'P', bubble_point: float = 2000) -> pd.DataFrame:
    """Classify data into saturated/undersaturated regimes based on pressure."""
    if pressure_col not in df.columns:
        raise InvalidPressureError(f"Pressure column '{pressure_col}' not found")
    df['Regime'] = np.where(df[pressure_col] >= bubble_point, 'Saturated', 'Undersaturated')
    logger.info("Classified regimes", pressure_col=pressure_col, bubble_point=bubble_point, 
                saturated=len(df[df['Regime'] == 'Saturated']), 
                undersaturated=len(df[df['Regime'] == 'Undersaturated']), 
                p_min=df[pressure_col].min(), p_max=df[pressure_col].max(), 
                p_mean=df[pressure_col].mean(), p_std=df[pressure_col].std())
    return df
