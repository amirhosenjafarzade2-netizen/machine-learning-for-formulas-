# optimization.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import structlog
from deap import base, creator, tools
from skopt import gp_minimize
from skopt.space import Real
from joblib import Memory
from models import build_neural_network, train_neural_network, predict_nn, compute_shap_values
from analysis import plot_heatmap
from config import Config
from data_utils import MissingDataError

logger = structlog.get_logger()
memory = Memory("cache_dir", verbose=0)

def setup_genetic_algorithm(creator_class_name: str, fitness_weights: tuple[float, ...]) -> None:
    """Set up genetic algorithm classes."""
    if not hasattr(creator, creator_class_name):
        creator.create(creator_class_name, base.Fitness, weights=fitness_weights)
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=getattr(creator, creator_class_name))
    logger.info("Set up genetic algorithm", class_name=creator_class_name)

@memory.cache
def evaluate_individual_target(individual: tuple[float, ...], model, scaler, feature_columns: list[str]) -> float:
    """Cached evaluation for genetic algorithm in optimize_target."""
    X_test = pd.DataFrame([individual], columns=feature_columns)
    X_test_scaled = scaler.transform(X_test)
    return predict_nn(model, X_test_scaled)[0][0]

def optimize_target_ga(df: pd.DataFrame, params: list[str], target_name: str, model, scaler, config: Config) -> tuple[pd.DataFrame, list[plt.Figure | go.Figure]]:
    """Optimize target using genetic algorithm."""
    features = df[params].drop(columns=[target_name])
    target = df[target_name]
    
    setup_genetic_algorithm("FitnessMax", (1.0,))
    
    rng = np.random.default_rng(config.random_seed)
    
    def create_individual() -> creator.Individual:
        return creator.Individual([rng.uniform(df[p].min(), df[p].max()) for p in features.columns])
    
    toolbox = base.Toolbox()
    toolbox.register("individual", create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_individual_target, model=model, scaler=scaler, feature_columns=features.columns)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    population = toolbox.population(n=config.pop_size)
    
    figs: list[plt.Figure | go.Figure] = []
    best_fitness_history = []
    for gen in range(config.n_generations):
        offspring = toolbox.select(population, len(population))
        offspring = [toolbox.clone(ind) for ind in offspring]
        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            if rng.random() < 0.7:
                toolbox.mate(ind1, ind2)
                del ind1.fitness.values
                del ind2.fitness.values
        for ind in offspring:
            if rng.random() < 0.2:
                toolbox.mutate(ind)
                for i in range(len(ind)):
                    ind[i] = max(df[features.columns[i]].min(), min(df[features.columns[i]].max(), ind[i]))
                del ind.fitness.values
        X_batch = np.array([list(ind) for ind in offspring])
        fitnesses = [evaluate_individual_target(tuple(ind), model, scaler, features.columns) for ind in X_batch]
        for ind, fit in zip(offspring, fitnesses):
            ind.fitness.values = (fit,)
        population[:] = offspring
        best_ind = tools.selBest(population, 1)[0]
        best_fitness_history.append(best_ind.fitness.values[0])
    
    best_ind = tools.selBest(population, 1)[0]
    
    fig = plt.figure(figsize=(8, 5))
    plt.plot(range(1, config.n_generations + 1), best_fitness_history, marker='o')
    plt.title(f'Optimization Progress for {target_name} (GA)')
    plt.xlabel('Generation')
    plt.ylabel(f'Predicted {target_name} (Scaled)')
    plt.grid(True)
    figs.append(fig)
    
    optimal_params = best_ind
    optimal_df = pd.DataFrame([optimal_params], columns=features.columns)
    optimal_df[target_name] = predict_nn(model, scaler.transform(optimal_df))[0][0] * target.std() + target.mean()
    
    return optimal_df, figs

def optimize_target_bayesian(df: pd.DataFrame, params: list[str], target_name: str, model, scaler, config: Config) -> tuple[pd.DataFrame, list[plt.Figure | go.Figure]]:
    """Optimize target using Bayesian optimization."""
    features = df[params].drop(columns=[target_name])
    target = df[target_name]
    
    search_space = [Real(df[p].min(), df[p].max(), name=p) for p in features.columns]
    
    @memory.cache
    def objective(params: list[float]) -> float:
        X_test = pd.DataFrame([params], columns=features.columns)
        X_test_scaled = scaler.transform(X_test)
        return -predict_nn(model, X_test_scaled)[0][0]
    
    figs: list[plt.Figure | go.Figure] = []
    rng = np.random.default_rng(config.random_seed)
    res = gp_minimize(
        objective,
        search_space,
        n_calls=config.n_calls,
        random_state=rng.integers(0, 2**32)
    )
    
    fig = plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(res.func_vals) + 1), -np.array(res.func_vals), marker='o')
    plt.title(f'Optimization Progress for {target_name} (Bayesian)')
    plt.xlabel('Iteration')
    plt.ylabel(f'Predicted {target_name} (Scaled)')
    plt.grid(True)
    figs.append(fig)
    
    optimal_params = res.x
    optimal_df = pd.DataFrame([optimal_params], columns=features.columns)
    optimal_df[target_name] = -res.fun * target.std() + target.mean()
    
    return optimal_df, figs

def optimize_target(df: pd.DataFrame, params: list[str], target_name: str, config: Config, optimizer: str = 'ga', progress_bar=None) -> tuple[pd.DataFrame, list[plt.Figure | go.Figure], tuple[np.ndarray, pd.DataFrame] | None]:
    """Optimize a target parameter using neural network and specified optimizer."""
    if target_name not in params:
        raise ValueError(f"Target '{target_name}' not in parameters: {', '.join(params)}")
    
    features = df[params].drop(columns=[target_name])
    target = df[target_name]
    
    model, scaler = train_neural_network(features, (target - target.mean()) / target.std(), config, progress_bar)
    
    weights = model.layers[0].get_weights()[0]
    feature_importance = np.mean(np.abs(weights), axis=1)
    importance_df = pd.DataFrame({'Feature': features.columns, 'Importance': feature_importance})
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    figs: list[plt.Figure | go.Figure] = []
    fig = plt.figure(figsize=(10, 6))
    plt.bar(importance_df['Feature'], importance_df['Importance'], color='skyblue')
    plt.title(f'Neural Network Feature Importance for {target_name}', fontsize=16)
    plt.xlabel('Features', fontsize=12)
    plt.ylabel('Importance Score', fontsize=12)
    plt.xticks(rotation=45)
    figs.append(fig)
    
    importance_matrix = pd.DataFrame(0.0, index=[target_name], columns=features.columns)
    for i, feature in enumerate(features.columns):
        importance_matrix.loc[target_name, feature] = feature_importance[i]
    fig = plot_heatmap(importance_matrix, features.columns, [target_name], f'Feature Importance for {target_name}', 'Features', 'Target')
    figs.append(fig)
    
    shap_values, X_sample = None, None
    if 'shap' in config.visualizations:
        try:
            shap_values, X_sample = compute_shap_values(model, features, config.shap_sample_size, config.random_seed)
        except Exception as e:
            logger.error(f"Failed to compute SHAP values for neural network", error=str(e))
    
    if optimizer == 'ga':
        optimal_df, opt_figs = optimize_target_ga(df, params, target_name, model, scaler, config)
    else:
        optimal_df, opt_figs = optimize_target_bayesian(df, params, target_name, model, scaler, config)
    
    figs.extend(opt_figs)
    return optimal_df, figs, (shap_values, X_sample)
