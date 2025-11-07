# ==========================
# src/data/preprocessor.py - Pré-processamento de Dados
# ==========================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from config import Config
from src.data.feature_engineering import create_features

def preprocess_data(df):
    """
    Preprocessa dados para treinamento
    
    Args:
        df: DataFrame original
    
    Returns:
        X_train, X_test, y_train, y_test, label_encoder, feature_names
    """
    # Fazer cópia para não modificar original
    df = df.copy()
    
    # Remover linhas com valores faltantes críticos
    critical_cols = ['Common Name', 'Observed Length (m)', 'Observed Weight (kg)']
    df = df.dropna(subset=critical_cols)
    
    # Separar features e target
    y = df['Common Name']
    X = df.drop('Common Name', axis=1)
    
    # Aplicar feature engineering
    X = create_features(X)
    
    # Codificar variáveis categóricas com one-hot encoding
    categorical_cols = Config.CATEGORICAL_FEATURES
    numerical_cols = Config.NUMERICAL_FEATURES
    
    # Selecionar apenas colunas que existem
    available_cat = [col for col in categorical_cols if col in X.columns]
    available_num = [col for col in numerical_cols if col in X.columns]
    
    X_encoded = pd.get_dummies(X[available_cat + available_num], drop_first=False)
    
    # Codificar target
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Dividir em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, 
        y_encoded, 
        test_size=Config.TEST_SIZE,
        random_state=Config.RANDOM_STATE,
        stratify=y_encoded
    )
    
    return X_train, X_test, y_train, y_test, le, X_encoded.columns.tolist()

def get_data_summary(df):
    """
    Gera resumo estatístico dos dados
    
    Args:
        df: DataFrame
    
    Returns:
        dict com estatísticas descritivas
    """
    summary = {}
    
    # Informações gerais
    summary['n_rows'] = len(df)
    summary['n_cols'] = len(df.columns)
    summary['n_species'] = df['Common Name'].nunique() if 'Common Name' in df.columns else 0
    
    # Estatísticas numéricas
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        summary['numeric_stats'] = df[numeric_cols].describe().to_dict()
    
    # Distribuição de espécies
    if 'Common Name' in df.columns:
        summary['species_distribution'] = df['Common Name'].value_counts().to_dict()
    
    # Distribuição por habitat
    if 'Habitat Type' in df.columns:
        summary['habitat_distribution'] = df['Habitat Type'].value_counts().to_dict()
    
    # Distribuição por região
    if 'Country/Region' in df.columns:
        summary['region_distribution'] = df['Country/Region'].value_counts().head(10).to_dict()
    
    # Status de conservação
    if 'Conservation Status' in df.columns:
        summary['conservation_status'] = df['Conservation Status'].value_counts().to_dict()
    
    # Valores faltantes
    summary['missing_values'] = df.isna().sum().to_dict()
    summary['missing_percentage'] = (df.isna().sum() / len(df) * 100).to_dict()
    
    return summary

def clean_data(df):
    """
    Limpa dados removendo duplicatas e outliers extremos
    
    Args:
        df: DataFrame
    
    Returns:
        DataFrame limpo
    """
    df = df.copy()
    
    # Remover duplicatas
    df = df.drop_duplicates()
    
    # Remover outliers extremos de comprimento e peso (IQR method)
    numeric_cols = ['Observed Length (m)', 'Observed Weight (kg)']
    
    for col in numeric_cols:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # Definir limites (mais permissivo: 3 * IQR)
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            
            # Filtrar outliers extremos
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    
    return df

def handle_missing_values(df):
    """
    Trata valores faltantes
    
    Args:
        df: DataFrame
    
    Returns:
        DataFrame com valores faltantes tratados
    """
    df = df.copy()
    
    # Para variáveis categóricas: preencher com 'Unknown'
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = df[col].fillna('Unknown')
    
    # Para variáveis numéricas: preencher com mediana
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
    
    return df