import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import structlog
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from data_utils import classify_regimes
from models import compute_shap_values
from config import Config

logger = structlog.get_logger()

def compute_mi_score(param1: str, param2: str, df: pd.DataFrame, random_seed: int = 42) -> float:
    """Compute mutual information score for a parameter pair."""
    if param1 == param2:
        return 1.0
    X = df[[param1]].fillna(df[param1].median()).values  # Shape: (n_samples, 1)
    y = df[param2].fillna(df[param2].median()).values.ravel()  # Shape: (n_samples,)
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        logger.error(f"NaN values detected in {param1} or {param2} after imputation", 
                     x_nans=np.sum(np.isnan(X)), y_nans=np.sum(np.isnan(y)))
        raise ValueError(f"Input contains NaN for {param1} or {param2}")
    rng = np.random.default_rng(random_seed)
    score = mutual_info_regression(X, y, random_state=rng.integers(0, 2**32))[0]
    logger.debug("Computed MI score", param1=param1, param2=param2, score=score)
    return score

def compute_mi_matrix(df: pd.DataFrame, params: list[str], regime: str | None = None, config: Config | None = None) -> pd.DataFrame:
    """Compute mutual information matrix sequentially."""
    if config is None:
        config = Config()
    if len(df) < config.min_rows:
        logger.warning(f"Dataset has {len(df)} rows, minimum is {config.min_rows}, returning empty matrix")
        return pd.DataFrame(np.nan, index=params, columns=params)
    
    mi_matrix = np.zeros((len(params), len(params)))
    param_pairs = [(i, j, params[i], params[j]) for i in range(len(params)) for j in range(i, len(params))]
    failed_pairs = []
    
    for i, j, param1, param2 in param_pairs:
        try:
            score = compute_mi_score(param1, param2, df, config.random_seed)
            mi_matrix[i, j] = score
            mi_matrix[j, i] = score
        except ValueError as e:
            logger.error("MI computation failed", param1=param1, param2=param2, error=str(e))
            failed_pairs.append((param1, param2))
            mi_matrix[i, j] = 0.0
            mi_matrix[j, i] = 0.0
    
    mi_matrix = pd.DataFrame(mi_matrix, index=params, columns=params)
    max_score = mi_matrix.max().max()
    if max_score > 0:
        mi_matrix = mi_matrix / max_score
    else:
        logger.warning(f"MI matrix ({regime or 'all'}) has max score 0, skipping normalization")
    valid_entries = np.sum((mi_matrix.values != 0) & (~mi_matrix.isna().values))
    logger.info("Computed MI matrix", regime=regime or "all", shape=mi_matrix.shape, 
                any_nans=mi_matrix.isna().any().any(), valid_entries=valid_entries, 
                failed_pairs=failed_pairs, mean_score=mi_matrix.values.mean(), 
                min_score=mi_matrix.values.min(), max_score=mi_matrix.values.max(), 
                sample=mi_matrix.iloc[:2, :2].to_dict())
    return mi_matrix

def plot_heatmap(data: pd.DataFrame, x_labels: list[str], y_labels: list[str], title: str, x_title: str, y_title: str) -> go.Figure:
    """Generate an interactive heatmap with Plotly."""
    if data.empty or data.isna().all().all() or not np.any(data.values != 0):
        logger.warning(f"Heatmap data for '{title}' is empty, all NaNs, or all zeros", 
                       shape=data.shape, any_nans=data.isna().any().any(), non_zero=np.sum(data.values != 0))
        return go.Figure()
    logger.info("Generating heatmap", title=title, shape=data.shape, any_nans=data.isna().any().any(), 
                non_zero=np.sum(data.values != 0), mean=data.values.mean(), min=data.values.min(), 
                max=data.values.max(), sample=data.iloc[:2, :2].to_dict())
    fig = px.imshow(
        data, x=x_labels, y=y_labels, title=title,
        labels={'x': x_title, 'y': y_title}, color_continuous_scale='Viridis',
        text_auto='.2f', width=800, height=800
    )
    fig.update_layout(
        title_font_size=16,
        xaxis_title_font_size=12,
        yaxis_title_font_size=12,
        font=dict(size=12)
    )
    return fig

def plot_pairplot(df: pd.DataFrame, params: list[str], title: str) -> plt.Figure:
    """Generate a pair plot with seaborn."""
    logger.info("Generating pairplot", title=title, params=params)
    sns_plot = sns.pairplot(df[params])
    fig = sns_plot.figure
    fig.suptitle(title, y=1.02, fontsize=16)
    return fig

def plot_parallel_coordinates(df: pd.DataFrame, params: list[str], title: str) -> go.Figure:
    """Generate a parallel coordinates plot with Plotly."""
    logger.info("Generating parallel coordinates", title=title, params=params)
    fig = px.parallel_coordinates(
        df, dimensions=params, title=title,
        color=params[0], color_continuous_scale='Viridis'
    )
    fig.update_layout(title_font_size=16, width=800, height=600)
    return fig

def plot_boxplot(df: pd.DataFrame, params: list[str], title: str) -> go.Figure:
    """Generate a box plot with Plotly."""
    logger.info("Generating boxplot", title=title, params=params)
    fig = go.Figure()
    for param in params:
        fig.add_trace(go.Box(y=df[param], name=param))
    fig.update_layout(title=title, title_font_size=16, width=800, height=600, yaxis_title="Value")
    return fig

def plot_pca(df: pd.DataFrame, params: list[str], title: str) -> go.Figure:
    """Generate a 2D PCA plot with Plotly."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[params])
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    explained_variance = pca.explained_variance_ratio_
    logger.info("Generating PCA plot", title=title, explained_variance=explained_variance.tolist())
    fig = px.scatter(
        x=X_pca[:, 0], y=X_pca[:, 1], title=title,
        labels={'x': f'PC1 ({explained_variance[0]:.2%})', 'y': f'PC2 ({explained_variance[1]:.2%})'}
    )
    fig.update_layout(title_font_size=16, width=800, height=600)
    return fig

def plot_pca_3d(df: pd.DataFrame, params: list[str], title: str) -> go.Figure:
    """Generate a 3D PCA plot with Plotly."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[params])
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)
    explained_variance = pca.explained_variance_ratio_
    logger.info("Generating PCA 3D plot", title=title, explained_variance=explained_variance.tolist())
    fig = px.scatter_3d(
        x=X_pca[:, 0], y=X_pca[:, 1], z=X_pca[:, 2], title=title,
        labels={
            'x': f'PC1 ({explained_variance[0]:.2%})',
            'y': f'PC2 ({explained_variance[1]:.2%})',
            'z': f'PC3 ({explained_variance[2]:.2%})'
        }
    )
    fig.update_layout(title_font_size=16, width=800, height=600)
    return fig

def plot_rf_importance(rf_importance: dict[str, pd.DataFrame], title: str) -> plt.Figure:
    """Generate a bar plot for RF importance."""
    fig = plt.figure(figsize=(10, 6))
    for target, importance_df in rf_importance.items():
        plt.bar(importance_df['Feature'], importance_df['Importance'], label=target, alpha=0.5)
    plt.title(title, fontsize=16)
    plt.xlabel('Features', fontsize=12)
    plt.ylabel('Importance Score', fontsize=12)
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    logger.info("Generating RF importance plot", title=title)
    return fig

def plot_shap_summary(shap_values: np.ndarray, X_sample: pd.DataFrame, title: str) -> plt.Figure:
    """Generate SHAP summary plot."""
    fig = plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.title(title, fontsize=16)
    logger.info("Generated SHAP summary plot", title=title)
    return fig

def analyze_relationships(df: pd.DataFrame, params: list[str], bubble_point: float, config: Config, mi_params: list[str] | None = None) -> list[plt.Figure | go.Figure]:
    """Perform relationship analysis between parameters."""
    df = classify_regimes(df, 'P', bubble_point)
    
    mi_params = mi_params or params
    if not all(p in params for p in mi_params):
        raise ValueError("mi_params must be a subset of params")
    if len(mi_params) > 10:
        logger.warning("Large number of MI parameters may be slow", count=len(mi_params))
    
    # Compute correlations and MI
    corr_matrix = df[params].corr(method='spearman')
    logger.info("Computed correlation matrix", shape=corr_matrix.shape, any_nans=corr_matrix.isna().any().any(), 
                non_zero=np.sum(corr_matrix.values != 0), mean=corr_matrix.values.mean(), 
                min=corr_matrix.values.min(), max=corr_matrix.values.max(), sample=corr_matrix.iloc[:2, :2].to_dict())
    
    mi_matrix = pd.DataFrame(np.nan, index=mi_params, columns=mi_params)
    try:
        mi_matrix = compute_mi_matrix(df, mi_params, config=config)
    except Exception as e:
        logger.warning("Failed to compute MI matrix", error=str(e))
    
    mi_saturated = pd.DataFrame(np.nan, index=mi_params, columns=mi_params)
    try:
        mi_saturated = compute_mi_matrix(df[df['Regime'] == 'Saturated'], mi_params, regime='saturated', config=config)
    except Exception as e:
        logger.warning("Failed to compute MI saturated matrix", error=str(e))
    
    mi_undersaturated = pd.DataFrame(np.nan, index=mi_params, columns=mi_params)
    try:
        mi_undersaturated = compute_mi_matrix(df[df['Regime'] == 'Undersaturated'], mi_params, regime='undersaturated', config=config)
    except Exception as e:
        logger.warning("Failed to compute MI undersaturated matrix", error=str(e))
    
    # Random Forest importance
    rf_importance: dict[str, pd.DataFrame] = {}
    shap_values_dict: dict[str, tuple[np.ndarray, pd.DataFrame]] = {}
    rng = np.random.default_rng(config.random_seed)
    for target in params:
        X = df[params].drop(columns=[target])
        y = df[target]
        if y.isna().any():
            logger.warning(f"Skipping RF importance for {target}: contains NaN values")
            continue
        if len(X) < config.min_rows:
            logger.info("Skipping RF importance", target=target, rows=len(X))
            continue
        rf = RandomForestRegressor(n_estimators=100, random_state=rng.integers(0, 2**32))
        rf.fit(X, y)
        perm_importance = permutation_importance(rf, X, y, n_repeats=10, random_state=rng.integers(0, 2**32))
        rf_importance[target] = pd.DataFrame({
            'Feature': X.columns,
            'Importance': perm_importance.importances_mean
        }).sort_values('Importance', ascending=False)
        if 'shap' in config.visualizations:
            try:
                shap_values, X_sample = compute_shap_values(rf, X, config.shap_sample_size, config.random_seed)
                shap_values_dict[target] = (shap_values, X_sample)
            except Exception as e:
                logger.error(f"Failed to compute SHAP values for {target}", error=str(e))
    
    # Visualizations
    figs: list[plt.Figure | go.Figure] = []
    selected_viz = config.visualizations
    
    if 'correlation' in selected_viz:
        fig = plot_heatmap(corr_matrix, params, params, 'Spearman Correlation Heatmap', 'Parameters', 'Parameters')
        figs.append(fig)
    
    if 'mi' in selected_viz:
        fig = plot_heatmap(mi_matrix, mi_params, mi_params, 'Normalized Mutual Information Heatmap', 'Parameters', 'Parameters')
        figs.append(fig)
    
    if 'mi_saturated' in selected_viz:
        fig = plot_heatmap(mi_saturated, mi_params, mi_params, 'Normalized MI Heatmap (Saturated Regime)', 'Parameters', 'Parameters')
        figs.append(fig)
    
    if 'mi_undersaturated' in selected_viz:
        fig = plot_heatmap(mi_undersaturated, mi_params, mi_params, 'Normalized MI Heatmap (Undersaturated Regime)', 'Parameters', 'Parameters')
        figs.append(fig)
    
    if 'pairplot' in selected_viz:
        fig = plot_pairplot(df, params, 'Pair Plot of Parameters')
        figs.append(fig)
    
    if 'parallel' in selected_viz:
        fig = plot_parallel_coordinates(df, params, 'Parallel Coordinates of Parameters')
        figs.append(fig)
    
    if 'boxplot' in selected_viz:
        fig = plot_boxplot(df, params, 'Box Plot of Parameters')
        figs.append(fig)
    
    if 'pca' in selected_viz:
        fig = plot_pca(df, params, 'PCA 2D Plot of Parameters')
        figs.append(fig)
    
    if 'pca_3d' in selected_viz:
        fig = plot_pca_3d(df, params, 'PCA 3D Plot of Parameters')
        figs.append(fig)
    
    if 'rf_importance' in selected_viz and rf_importance:
        fig = plot_rf_importance(rf_importance, 'Random Forest Feature Importance')
        figs.append(fig)
    
    if 'shap' in selected_viz and shap_values_dict:
        for target, (shap_values, X_sample) in shap_values_dict.items():
            fig = plot_shap_summary(shap_values, X_sample, f'SHAP Summary Plot for {target}')
            figs.append(fig)
    
    logger.info("Generated figures", count=len(figs), types=[type(fig).__name__ for fig in figs])
    return figs
