# ==========================
# src/utils/helpers.py - Funções Auxiliares
# ==========================
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import joblib
from config import Config

def format_bytes(bytes_size):
    """
    Formata tamanho em bytes para formato legível
    
    Args:
        bytes_size: Tamanho em bytes
    
    Returns:
        str: Tamanho formatado (ex: "1.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f} TB"

def format_percentage(value, decimals=2):
    """
    Formata valor como porcentagem
    
    Args:
        value: Valor entre 0 e 1
        decimals: Número de casas decimais
    
    Returns:
        str: Porcentagem formatada
    """
    try:
        return f"{float(value) * 100:.{decimals}f}%"
    except (ValueError, TypeError):
        return "N/A"

def format_number(value, decimals=2):
    """
    Formata número com casas decimais
    
    Args:
        value: Número a formatar
        decimals: Número de casas decimais
    
    Returns:
        str: Número formatado
    """
    try:
        return f"{float(value):.{decimals}f}"
    except (ValueError, TypeError):
        return "N/A"

def get_timestamp():
    """
    Retorna timestamp formatado
    
    Returns:
        str: Timestamp (formato: YYYYMMDD_HHMMSS)
    """
    return datetime.now().strftime('%Y%m%d_%H%M%S')

def get_datetime_str():
    """
    Retorna data/hora formatada para exibição
    
    Returns:
        str: Data/hora (formato: DD/MM/YYYY HH:MM:SS)
    """
    return datetime.now().strftime('%d/%m/%Y %H:%M:%S')

def save_model_metadata(model_path, metadata):
    """
    Salva metadados do modelo em arquivo JSON
    
    Args:
        model_path: Caminho do arquivo .pkl do modelo
        metadata: Dict com metadados
    """
    model_path = Path(model_path)
    json_path = model_path.with_suffix('.json')
    
    # Adicionar timestamp
    metadata['saved_at'] = get_datetime_str()
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

def load_model_metadata(model_path):
    """
    Carrega metadados do modelo
    
    Args:
        model_path: Caminho do arquivo .pkl do modelo
    
    Returns:
        dict: Metadados ou None se não existir
    """
    model_path = Path(model_path)
    json_path = model_path.with_suffix('.json')
    
    if json_path.exists():
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

def get_dataset_stats(df):
    """
    Retorna estatísticas resumidas do dataset
    
    Args:
        df: DataFrame
    
    Returns:
        dict: Estatísticas
    """
    stats = {
        'n_rows': len(df),
        'n_cols': len(df.columns),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024),
        'missing_values_total': df.isna().sum().sum(),
        'missing_percentage': (df.isna().sum().sum() / (len(df) * len(df.columns))) * 100
    }
    
    # Estatísticas de colunas numéricas
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        stats['numeric_cols'] = len(numeric_cols)
        stats['numeric_summary'] = {
            col: {
                'mean': float(df[col].mean()),
                'std': float(df[col].std()),
                'min': float(df[col].min()),
                'max': float(df[col].max())
            } for col in numeric_cols
        }
    
    # Estatísticas de colunas categóricas
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        stats['categorical_cols'] = len(categorical_cols)
        stats['categorical_summary'] = {
            col: {
                'unique_values': int(df[col].nunique()),
                'most_common': str(df[col].mode()[0]) if len(df[col].mode()) > 0 else None
            } for col in categorical_cols
        }
    
    return stats

def clean_filename(filename):
    """
    Limpa nome de arquivo removendo caracteres inválidos
    
    Args:
        filename: Nome do arquivo
    
    Returns:
        str: Nome limpo
    """
    import re
    # Remover caracteres especiais, manter apenas letras, números, ponto e underscore
    clean_name = re.sub(r'[^\w\.-]', '_', filename)
    return clean_name

def get_model_info(model_path):
    """
    Retorna informações sobre modelo salvo
    
    Args:
        model_path: Caminho do modelo
    
    Returns:
        dict: Informações do modelo
    """
    model_path = Path(model_path)
    
    if not model_path.exists():
        return None
    
    info = {
        'filename': model_path.name,
        'size': format_bytes(model_path.stat().st_size),
        'created': datetime.fromtimestamp(model_path.stat().st_ctime).strftime('%d/%m/%Y %H:%M:%S'),
        'modified': datetime.fromtimestamp(model_path.stat().st_mtime).strftime('%d/%m/%Y %H:%M:%S')
    }
    
    # Carregar metadados se existirem
    metadata = load_model_metadata(model_path)
    if metadata:
        info['metadata'] = metadata
    
    return info

def create_summary_dict(title, data, format_type='table'):
    """
    Cria dicionário formatado para exibição em templates
    
    Args:
        title: Título do resumo
        data: Dados a formatar
        format_type: Tipo de formato ('table', 'list', 'text')
    
    Returns:
        dict: Dados formatados
    """
    return {
        'title': title,
        'data': data,
        'format': format_type,
        'timestamp': get_datetime_str()
    }

def get_available_models_info():
    """
    Retorna informações sobre modelos disponíveis
    
    Returns:
        dict: Info dos modelos
    """
    return {
        'random_forest': {
            'name': 'Random Forest',
            'description': 'Ensemble de árvores de decisão. Robusto e preciso.',
            'pros': ['Alta precisão', 'Lida bem com features categóricas', 'Menos propenso a overfitting'],
            'cons': ['Pode ser lento com muitos dados', 'Difícil de interpretar'],
            'best_for': 'Datasets balanceados com muitas features'
        },
        'gradient_boosting': {
            'name': 'Gradient Boosting',
            'description': 'Boosting sequencial de árvores. Alta performance.',
            'pros': ['Excelente precisão', 'Flexível', 'Lida bem com dados desbalanceados'],
            'cons': ['Risco de overfitting', 'Requer tuning cuidadoso'],
            'best_for': 'Competições e alta performance'
        },
        'svm': {
            'name': 'Support Vector Machine',
            'description': 'Classificação por hiperplanos. Eficiente em alta dimensionalidade.',
            'pros': ['Eficaz em espaços de alta dimensão', 'Versátil (múltiplos kernels)'],
            'cons': ['Lento com grandes datasets', 'Sensível à escala'],
            'best_for': 'Datasets pequenos/médios com muitas features'
        }
    }

def calculate_class_weights(y):
    """
    Calcula pesos das classes para lidar com desbalanceamento
    
    Args:
        y: Labels
    
    Returns:
        dict: Pesos por classe
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    
    return dict(zip(classes, weights))