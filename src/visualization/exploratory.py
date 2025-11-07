# ==========================
# src/visualization/exploratory.py - Gráficos de Análise Exploratória
# ==========================
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from config import Config
import warnings
warnings.filterwarnings('ignore')

# Configurar estilo
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10

def generate_exploratory_plots(df):
    """
    Gera gráficos de análise exploratória
    
    Args:
        df: DataFrame com dados
    
    Returns:
        list: Lista de nomes dos arquivos gerados
    """
    chart_files = []
    charts_folder = Config.CHARTS_FOLDER
    
    # Limpar pasta de gráficos
    for file in charts_folder.glob('*.png'):
        file.unlink()
    
    # 1. Distribuição de espécies
    if 'Common Name' in df.columns:
        plt.figure(figsize=(12, 6))
        species_counts = df['Common Name'].value_counts().head(15)
        sns.barplot(x=species_counts.values, y=species_counts.index, palette='viridis')
        plt.title('Top 15 Espécies Mais Comuns no Dataset', fontsize=14, fontweight='bold')
        plt.xlabel('Quantidade de Observações')
        plt.ylabel('Espécie')
        plt.tight_layout()
        
        filename = 'species_distribution.png'
        plt.savefig(charts_folder / filename, dpi=100, bbox_inches='tight')
        plt.close()
        chart_files.append(filename)
    
    # 2. Distribuição de comprimento
    if 'Observed Length (m)' in df.columns:
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        df['Observed Length (m)'].hist(bins=30, color='steelblue', edgecolor='black', alpha=0.7)
        plt.title('Distribuição de Comprimento', fontweight='bold')
        plt.xlabel('Comprimento (m)')
        plt.ylabel('Frequência')
        
        plt.subplot(1, 2, 2)
        df.boxplot(column='Observed Length (m)', patch_artist=True)
        plt.title('Boxplot - Comprimento', fontweight='bold')
        plt.ylabel('Comprimento (m)')
        
        plt.tight_layout()
        filename = 'length_distribution.png'
        plt.savefig(charts_folder / filename, dpi=100, bbox_inches='tight')
        plt.close()
        chart_files.append(filename)
    
    # 3. Distribuição de peso
    if 'Observed Weight (kg)' in df.columns:
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        df['Observed Weight (kg)'].hist(bins=30, color='coral', edgecolor='black', alpha=0.7)
        plt.title('Distribuição de Peso', fontweight='bold')
        plt.xlabel('Peso (kg)')
        plt.ylabel('Frequência')
        
        plt.subplot(1, 2, 2)
        df.boxplot(column='Observed Weight (kg)', patch_artist=True)
        plt.title('Boxplot - Peso', fontweight='bold')
        plt.ylabel('Peso (kg)')
        
        plt.tight_layout()
        filename = 'weight_distribution.png'
        plt.savefig(charts_folder / filename, dpi=100, bbox_inches='tight')
        plt.close()
        chart_files.append(filename)
    
    # 4. Relação Peso x Comprimento
    if 'Observed Length (m)' in df.columns and 'Observed Weight (kg)' in df.columns:
        plt.figure(figsize=(10, 6))
        
        if 'Common Name' in df.columns:
            # Colorir por espécie (top 5)
            top_species = df['Common Name'].value_counts().head(5).index
            df_plot = df[df['Common Name'].isin(top_species)]
            
            for species in top_species:
                species_data = df_plot[df_plot['Common Name'] == species]
                plt.scatter(
                    species_data['Observed Length (m)'],
                    species_data['Observed Weight (kg)'],
                    label=species,
                    alpha=0.6,
                    s=50
                )
            plt.legend(title='Espécie', bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            plt.scatter(
                df['Observed Length (m)'],
                df['Observed Weight (kg)'],
                alpha=0.6,
                color='green',
                s=50
            )
        
        plt.title('Relação entre Comprimento e Peso', fontsize=14, fontweight='bold')
        plt.xlabel('Comprimento (m)')
        plt.ylabel('Peso (kg)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filename = 'length_vs_weight.png'
        plt.savefig(charts_folder / filename, dpi=100, bbox_inches='tight')
        plt.close()
        chart_files.append(filename)
    
    # 5. Distribuição por Habitat
    if 'Habitat Type' in df.columns:
        plt.figure(figsize=(10, 6))
        habitat_counts = df['Habitat Type'].value_counts()
        colors = sns.color_palette('Set2', len(habitat_counts))
        plt.pie(
            habitat_counts.values,
            labels=habitat_counts.index,
            autopct='%1.1f%%',
            startangle=90,
            colors=colors
        )
        plt.title('Distribuição por Tipo de Habitat', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        filename = 'habitat_distribution.png'
        plt.savefig(charts_folder / filename, dpi=100, bbox_inches='tight')
        plt.close()
        chart_files.append(filename)
    
    # 6. Distribuição por Região (Top 10)
    if 'Country/Region' in df.columns:
        plt.figure(figsize=(12, 6))
        region_counts = df['Country/Region'].value_counts().head(10)
        sns.barplot(x=region_counts.values, y=region_counts.index, palette='coolwarm')
        plt.title('Top 10 Países/Regiões', fontsize=14, fontweight='bold')
        plt.xlabel('Quantidade de Observações')
        plt.ylabel('País/Região')
        plt.tight_layout()
        
        filename = 'region_distribution.png'
        plt.savefig(charts_folder / filename, dpi=100, bbox_inches='tight')
        plt.close()
        chart_files.append(filename)
    
    # 7. Status de Conservação
    if 'Conservation Status' in df.columns:
        plt.figure(figsize=(10, 6))
        status_counts = df['Conservation Status'].value_counts()
        colors = ['#2ecc71', '#f39c12', '#e74c3c', '#95a5a6', '#3498db'][:len(status_counts)]
        plt.barh(status_counts.index, status_counts.values, color=colors)
        plt.title('Status de Conservação', fontsize=14, fontweight='bold')
        plt.xlabel('Quantidade de Observações')
        plt.ylabel('Status')
        plt.tight_layout()
        
        filename = 'conservation_status.png'
        plt.savefig(charts_folder / filename, dpi=100, bbox_inches='tight')
        plt.close()
        chart_files.append(filename)
    
    # 8. Matriz de Correlação
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) >= 2:
        plt.figure(figsize=(10, 8))
        correlation_matrix = df[numeric_cols].corr()
        sns.heatmap(
            correlation_matrix,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            square=True,
            linewidths=1,
            cbar_kws={"shrink": 0.8}
        )
        plt.title('Matriz de Correlação - Features Numéricas', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        filename = 'correlation_matrix.png'
        plt.savefig(charts_folder / filename, dpi=100, bbox_inches='tight')
        plt.close()
        chart_files.append(filename)
    
    return chart_files