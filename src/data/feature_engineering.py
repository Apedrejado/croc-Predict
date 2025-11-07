# ==========================
# src/data/feature_engineering.py - Criação de Features
# ==========================
import pandas as pd
import numpy as np

def create_features(df):
    """
    Cria features derivadas a partir dos dados originais
    
    Args:
        df: DataFrame com dados originais
    
    Returns:
        DataFrame com features adicionais
    """
    df = df.copy()
    
    # Feature 1: Peso por comprimento (densidade relativa)
    if 'Observed Length (m)' in df.columns and 'Observed Weight (kg)' in df.columns:
        df['Weight_per_Length'] = df['Observed Weight (kg)'] / df['Observed Length (m)']
        df['Weight_per_Length'] = df['Weight_per_Length'].replace([np.inf, -np.inf], np.nan)
        df['Weight_per_Length'] = df['Weight_per_Length'].fillna(df['Weight_per_Length'].median())
    
    # Feature 2: Classificação de tamanho
    if 'Observed Length (m)' in df.columns:
        df['Size_Class'] = pd.cut(
            df['Observed Length (m)'],
            bins=[0, 2.0, 4.0, float('inf')],
            labels=['Small', 'Medium', 'Large'],
            include_lowest=True
        )
        df['Size_Class'] = df['Size_Class'].astype(str)
    
    # Feature 3: Agrupamento geográfico
    if 'Country/Region' in df.columns:
        region_map = {
            'Mexico': 'Central America',
            'Belize': 'Central America',
            'Guatemala': 'Central America',
            'Honduras': 'Central America',
            'Costa Rica': 'Central America',
            'Nicaragua': 'Central America',
            'Panama': 'Central America',
            'Cuba': 'Caribbean',
            'Jamaica': 'Caribbean',
            'Haiti': 'Caribbean',
            'Dominican Republic': 'Caribbean',
            'Philippines': 'Southeast Asia',
            'Indonesia': 'Southeast Asia',
            'Malaysia': 'Southeast Asia',
            'Thailand': 'Southeast Asia',
            'Vietnam': 'Southeast Asia',
            'Cambodia': 'Southeast Asia',
            'Papua New Guinea': 'Oceania',
            'Australia': 'Oceania',
            'Solomon Islands': 'Oceania',
            'Venezuela': 'South America',
            'Colombia': 'South America',
            'Brazil': 'South America',
            'Peru': 'South America',
            'Ecuador': 'South America',
            'India': 'South Asia',
            'Bangladesh': 'South Asia',
            'Sri Lanka': 'South Asia',
            'Myanmar': 'South Asia',
            'United States': 'North America',
            'China': 'East Asia'
        }
        df['Region_Group'] = df['Country/Region'].map(region_map).fillna('Other')
    
    # Feature 4: Agrupamento de habitat
    if 'Habitat Type' in df.columns:
        habitat_map = {
            'Rivers': 'Freshwater',
            'Lakes': 'Freshwater',
            'Swamps': 'Wetland',
            'Freshwater Wetlands': 'Wetland',
            'Marshes': 'Wetland',
            'Estuaries': 'Brackish',
            'Mangroves': 'Brackish',
            'Coastal': 'Marine',
            'Marine': 'Marine'
        }
        df['Habitat_Type_Group'] = df['Habitat Type'].map(habitat_map).fillna('Other')
    
    # Feature 5: Refinar status de conservação
    if 'Conservation Status' in df.columns:
        status_map = {
            'Data Deficient': 'Unknown',
            'Not Evaluated': 'Unknown'
        }
        df['Conservation_Status_Group'] = df['Conservation Status'].map(status_map).fillna(
            df['Conservation Status']
        )
    
    # Feature 6: IMC do crocodilo (Body Mass Index adaptado)
    if 'Observed Length (m)' in df.columns and 'Observed Weight (kg)' in df.columns:
        # BMI = peso / comprimento^2
        df['Croc_BMI'] = df['Observed Weight (kg)'] / (df['Observed Length (m)'] ** 2)
        df['Croc_BMI'] = df['Croc_BMI'].replace([np.inf, -np.inf], np.nan)
        df['Croc_BMI'] = df['Croc_BMI'].fillna(df['Croc_BMI'].median())
    
    # Feature 7: Categoria de peso
    if 'Observed Weight (kg)' in df.columns:
        df['Weight_Class'] = pd.cut(
            df['Observed Weight (kg)'],
            bins=[0, 50, 200, 500, float('inf')],
            labels=['Light', 'Medium', 'Heavy', 'Very Heavy'],
            include_lowest=True
        )
        df['Weight_Class'] = df['Weight_Class'].astype(str)
    
    # Feature 8: Indicador de água salgada/doce
    if 'Habitat_Type_Group' in df.columns:
        df['Is_Saltwater'] = df['Habitat_Type_Group'].isin(['Brackish', 'Marine']).astype(int)
    
    return df

def get_feature_importance_names():
    """
    Retorna lista de nomes descritivos para features
    
    Returns:
        dict: mapeamento de features para nomes descritivos
    """
    return {
        'Observed Length (m)': 'Comprimento Observado',
        'Observed Weight (kg)': 'Peso Observado',
        'Weight_per_Length': 'Densidade Relativa (Peso/Comprimento)',
        'Croc_BMI': 'Índice de Massa Corporal',
        'Size_Class_Small': 'Tamanho: Pequeno',
        'Size_Class_Medium': 'Tamanho: Médio',
        'Size_Class_Large': 'Tamanho: Grande',
        'Region_Group_Central America': 'Região: América Central',
        'Region_Group_South America': 'Região: América do Sul',
        'Region_Group_Southeast Asia': 'Região: Sudeste Asiático',
        'Region_Group_Oceania': 'Região: Oceania',
        'Habitat_Type_Group_Freshwater': 'Habitat: Água Doce',
        'Habitat_Type_Group_Wetland': 'Habitat: Pântano',
        'Habitat_Type_Group_Brackish': 'Habitat: Água Salobra',
        'Habitat_Type_Group_Marine': 'Habitat: Marinho',
        'Is_Saltwater': 'Habitat de Água Salgada'
    }