# ==========================
# src/utils/validators.py - Validações
# ==========================
import pandas as pd
import numpy as np
from pathlib import Path
from config import Config

def validate_file_extension(filename):
    """
    Valida extensão do arquivo
    
    Args:
        filename: Nome do arquivo
    
    Returns:
        bool: True se extensão válida
    """
    if not filename:
        return False
    
    extension = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
    return extension in Config.ALLOWED_EXTENSIONS

def validate_file_size(file_size):
    """
    Valida tamanho do arquivo
    
    Args:
        file_size: Tamanho em bytes
    
    Returns:
        tuple: (is_valid: bool, message: str)
    """
    max_size = Config.MAX_CONTENT_LENGTH
    
    if file_size > max_size:
        max_mb = max_size / (1024 * 1024)
        return False, f"Arquivo muito grande. Máximo: {max_mb}MB"
    
    return True, "Tamanho válido"

def validate_dataframe_columns(df, required_columns=None):
    """
    Valida colunas do DataFrame
    
    Args:
        df: DataFrame a validar
        required_columns: Lista de colunas obrigatórias (default: Config.REQUIRED_COLUMNS)
    
    Returns:
        tuple: (is_valid: bool, missing_columns: list)
    """
    if required_columns is None:
        required_columns = Config.REQUIRED_COLUMNS
    
    missing = set(required_columns) - set(df.columns)
    
    return len(missing) == 0, list(missing)

def validate_numeric_column(df, column_name):
    """
    Valida se coluna contém valores numéricos válidos
    
    Args:
        df: DataFrame
        column_name: Nome da coluna
    
    Returns:
        tuple: (is_valid: bool, message: str)
    """
    if column_name not in df.columns:
        return False, f"Coluna '{column_name}' não encontrada"
    
    # Verificar se é numérica
    if not pd.api.types.is_numeric_dtype(df[column_name]):
        # Tentar converter
        try:
            pd.to_numeric(df[column_name], errors='raise')
        except:
            return False, f"Coluna '{column_name}' não contém valores numéricos válidos"
    
    # Verificar se há valores válidos
    valid_values = df[column_name].notna().sum()
    if valid_values == 0:
        return False, f"Coluna '{column_name}' não possui valores válidos"
    
    # Verificar se há valores negativos (não faz sentido para comprimento/peso)
    if (df[column_name] < 0).any():
        return False, f"Coluna '{column_name}' contém valores negativos"
    
    return True, "Coluna válida"

def validate_categorical_column(df, column_name, min_categories=1):
    """
    Valida coluna categórica
    
    Args:
        df: DataFrame
        column_name: Nome da coluna
        min_categories: Mínimo de categorias diferentes
    
    Returns:
        tuple: (is_valid: bool, message: str)
    """
    if column_name not in df.columns:
        return False, f"Coluna '{column_name}' não encontrada"
    
    # Verificar se há valores
    valid_values = df[column_name].notna().sum()
    if valid_values == 0:
        return False, f"Coluna '{column_name}' não possui valores"
    
    # Verificar número de categorias
    n_categories = df[column_name].nunique()
    if n_categories < min_categories:
        return False, f"Coluna '{column_name}' tem apenas {n_categories} categoria(s)"
    
    return True, "Coluna válida"

def validate_model_params(model_type, params):
    """
    Valida parâmetros do modelo
    
    Args:
        model_type: Tipo do modelo ('random_forest', 'gradient_boosting', 'svm')
        params: Dict com parâmetros
    
    Returns:
        tuple: (is_valid: bool, message: str)
    """
    if model_type not in Config.AVAILABLE_MODELS:
        return False, f"Modelo '{model_type}' não disponível"
    
    # Validações específicas por modelo
    if model_type == 'random_forest':
        if 'n_estimators' in params:
            if not isinstance(params['n_estimators'], int) or params['n_estimators'] < 1:
                return False, "n_estimators deve ser inteiro positivo"
        
        if 'max_depth' in params and params['max_depth'] is not None:
            if not isinstance(params['max_depth'], int) or params['max_depth'] < 1:
                return False, "max_depth deve ser inteiro positivo ou None"
    
    elif model_type == 'gradient_boosting':
        if 'n_estimators' in params:
            if not isinstance(params['n_estimators'], int) or params['n_estimators'] < 1:
                return False, "n_estimators deve ser inteiro positivo"
        
        if 'learning_rate' in params:
            if not isinstance(params['learning_rate'], (int, float)) or params['learning_rate'] <= 0:
                return False, "learning_rate deve ser número positivo"
    
    elif model_type == 'svm':
        if 'C' in params:
            if not isinstance(params['C'], (int, float)) or params['C'] <= 0:
                return False, "C deve ser número positivo"
        
        if 'kernel' in params:
            valid_kernels = ['linear', 'poly', 'rbf', 'sigmoid']
            if params['kernel'] not in valid_kernels:
                return False, f"kernel deve ser um de: {', '.join(valid_kernels)}"
    
    return True, "Parâmetros válidos"

def validate_prediction_input(input_data):
    """
    Valida dados de entrada para predição
    
    Args:
        input_data: Dict com dados de entrada
    
    Returns:
        tuple: (is_valid: bool, message: str)
    """
    required_fields = [
        'Observed Length (m)',
        'Observed Weight (kg)',
        'Conservation Status',
        'Country/Region',
        'Habitat Type',
        'Genus',
        'Family',
        'Age Class',
        'Sex'
    ]
    
    # Verificar campos obrigatórios
    missing = [field for field in required_fields if field not in input_data]
    if missing:
        return False, f"Campos faltando: {', '.join(missing)}"
    
    # Validar valores numéricos
    try:
        length = float(input_data['Observed Length (m)'])
        if length <= 0:
            return False, "Comprimento deve ser maior que zero"
    except (ValueError, TypeError):
        return False, "Comprimento inválido"
    
    try:
        weight = float(input_data['Observed Weight (kg)'])
        if weight <= 0:
            return False, "Peso deve ser maior que zero"
    except (ValueError, TypeError):
        return False, "Peso inválido"
    
    # Validar campos categóricos (não podem estar vazios)
    for field in ['Conservation Status', 'Country/Region', 'Habitat Type', 'Genus', 'Family', 'Age Class', 'Sex']:
        if not input_data[field] or str(input_data[field]).strip() == '':
            return False, f"Campo '{field}' não pode estar vazio"
    
    return True, "Dados válidos"