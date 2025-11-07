# ==========================
# src/data/loader.py - Carregamento e Validação de Dados
# ==========================
import pandas as pd
from pathlib import Path
from config import Config

def load_dataset(filepath):
    """
    Carrega dataset CSV ou Excel
    
    Args:
        filepath: Caminho do arquivo (str ou Path)
    
    Returns:
        DataFrame com os dados carregados
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {filepath}")
    
    # Carregar de acordo com a extensão
    if filepath.suffix == '.csv':
        df = pd.read_csv(filepath)
    elif filepath.suffix in ['.xlsx', '.xls']:
        df = pd.read_excel(filepath)
    else:
        raise ValueError(f"Formato não suportado: {filepath.suffix}")
    
    return df

def validate_dataset(df):
    """
    Valida se o dataset possui as colunas obrigatórias
    
    Args:
        df: DataFrame a ser validado
    
    Returns:
        tuple: (is_valid: bool, message: str)
    """
    required_cols = Config.REQUIRED_COLUMNS
    
    # Verificar se DataFrame está vazio
    if df.empty:
        return False, "Dataset está vazio"
    
    # Verificar colunas obrigatórias
    missing_cols = set(required_cols) - set(df.columns)
    
    if missing_cols:
        return False, f"Colunas faltando: {', '.join(missing_cols)}"
    
    # Verificar se há valores na coluna target
    if df['Common Name'].isna().all():
        return False, "Coluna 'Common Name' não possui valores"
    
    # Verificar se há valores numéricos válidos
    numeric_cols = ['Observed Length (m)', 'Observed Weight (kg)']
    for col in numeric_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except:
                return False, f"Coluna '{col}' deve conter valores numéricos"
        
        if df[col].isna().all():
            return False, f"Coluna '{col}' não possui valores válidos"
    
    return True, "Dataset válido"

def get_dataset_info(df):
    """
    Retorna informações básicas do dataset
    
    Args:
        df: DataFrame
    
    Returns:
        dict com informações do dataset
    """
    return {
        'n_rows': len(df),
        'n_cols': len(df.columns),
        'n_species': df['Common Name'].nunique(),
        'columns': list(df.columns),
        'missing_values': df.isna().sum().to_dict(),
        'dtypes': df.dtypes.astype(str).to_dict()
    }