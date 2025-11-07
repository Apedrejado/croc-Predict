# ==========================
# src/visualization/feature_importance.py - Importância de Features
# ==========================
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from config import Config
import warnings
warnings.filterwarnings('ignore')

def plot_feature_importance(model, feature_names, top_n=20):
    """
    Plota importância das features
    
    Args:
        model: Modelo treinado (sklearn)
        feature_names: Nomes das features
        top_n: Número de features mais importantes a mostrar
    
    Returns:
        str: Nome do arquivo gerado ou None
    """
    charts_folder = Config.CHARTS_FOLDER
    
    # Obter importância das features
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # Para modelos lineares
        importances = np.abs(model.coef_).mean(axis=0)
    else:
        return None
    
    # Criar DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False).head(top_n)
    
    # Plot
    plt.figure(figsize=(12, 8))
    colors = sns.color_palette('viridis', len(importance_df))
    
    plt.barh(
        range(len(importance_df)),
        importance_df['importance'].values,
        color=colors
    )
    
    plt.yticks(range(len(importance_df)), importance_df['feature'].values)
    plt.xlabel('Importância')
    plt.title(f'Top {top_n} Features Mais Importantes', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    filename = 'feature_importance.png'
    plt.savefig(charts_folder / filename, dpi=100, bbox_inches='tight')
    plt.close()
    
    return filename

def plot_feature_importance_comparison(models_dict, feature_names, top_n=15):
    """
    Compara importância de features entre múltiplos modelos
    
    Args:
        models_dict: Dict com {nome_modelo: modelo_treinado}
        feature_names: Nomes das features
        top_n: Número de features a comparar
    
    Returns:
        str: Nome do arquivo gerado ou None
    """
    charts_folder = Config.CHARTS_FOLDER
    
    # Coletar importâncias
    importance_data = {}
    
    for model_name, model in models_dict.items():
        if hasattr(model, 'feature_importances_'):
            importance_data[model_name] = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance_data[model_name] = np.abs(model.coef_).mean(axis=0)
    
    if not importance_data:
        return None
    
    # Criar DataFrame
    df = pd.DataFrame(importance_data, index=feature_names)
    
    # Selecionar top N features (média entre modelos)
    df['mean'] = df.mean(axis=1)
    df_top = df.nlargest(top_n, 'mean').drop('mean', axis=1)
    
    # Plot
    plt.figure(figsize=(14, 8))
    df_top.plot(kind='barh', figsize=(14, 8), width=0.8)
    
    plt.xlabel('Importância')
    plt.ylabel('Feature')
    plt.title(f'Comparação de Importância - Top {top_n} Features', fontsize=14, fontweight='bold')
    plt.legend(title='Modelo', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    filename = 'feature_importance_comparison.png'
    plt.savefig(charts_folder / filename, dpi=100, bbox_inches='tight')
    plt.close()
    
    return filename

def plot_permutation_importance(model, X_test, y_test, feature_names, top_n=20):
    """
    Calcula e plota importância por permutação
    
    Args:
        model: Modelo treinado
        X_test: Features de teste
        y_test: Labels de teste
        feature_names: Nomes das features
        top_n: Número de features a mostrar
    
    Returns:
        str: Nome do arquivo gerado
    """
    from sklearn.inspection import permutation_importance
    
    charts_folder = Config.CHARTS_FOLDER
    
    # Calcular permutation importance
    perm_importance = permutation_importance(
        model, X_test, y_test,
        n_repeats=10,
        random_state=Config.RANDOM_STATE,
        n_jobs=-1
    )
    
    # Criar DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance_mean': perm_importance.importances_mean,
        'importance_std': perm_importance.importances_std
    }).sort_values('importance_mean', ascending=False).head(top_n)
    
    # Plot
    plt.figure(figsize=(12, 8))
    
    plt.barh(
        range(len(importance_df)),
        importance_df['importance_mean'].values,
        xerr=importance_df['importance_std'].values,
        color='teal',
        alpha=0.7
    )
    
    plt.yticks(range(len(importance_df)), importance_df['feature'].values)
    plt.xlabel('Importância (Permutation)')
    plt.title(f'Top {top_n} Features - Permutation Importance', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    filename = 'permutation_importance.png'
    plt.savefig(charts_folder / filename, dpi=100, bbox_inches='tight')
    plt.close()
    
    return filename