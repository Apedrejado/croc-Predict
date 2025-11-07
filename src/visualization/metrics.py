# ==========================
# src/visualization/metrics.py - Visualização de Métricas dos Modelos
# ==========================
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from config import Config
import warnings
warnings.filterwarnings('ignore')

def generate_metrics_plots(model, X_test, y_test, class_names, feature_names=None):
    """
    Gera gráficos de métricas do modelo
    
    Args:
        model: Modelo treinado (sklearn)
        X_test: Features de teste
        y_test: Labels de teste
        class_names: Nomes das classes
        feature_names: Nomes das features (opcional)
    
    Returns:
        list: Lista de nomes dos arquivos gerados
    """
    chart_files = []
    charts_folder = Config.CHARTS_FOLDER
    
    # Fazer predições
    y_pred = model.predict(X_test)
    
    # 1. Matriz de Confusão
    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(y_test, y_pred)
    
    # Normalizar matriz de confusão
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm_normalized,
        display_labels=class_names
    )
    disp.plot(cmap='Blues', values_format='.2f', ax=plt.gca())
    plt.title('Matriz de Confusão (Normalizada)', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    filename = 'confusion_matrix.png'
    plt.savefig(charts_folder / filename, dpi=100, bbox_inches='tight')
    plt.close()
    chart_files.append(filename)
    
    # 2. Matriz de Confusão (valores absolutos)
    plt.figure(figsize=(12, 10))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )
    disp.plot(cmap='Greens', values_format='d', ax=plt.gca())
    plt.title('Matriz de Confusão (Contagens)', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    filename = 'confusion_matrix_counts.png'
    plt.savefig(charts_folder / filename, dpi=100, bbox_inches='tight')
    plt.close()
    chart_files.append(filename)
    
    # 3. Acurácia por Classe
    plt.figure(figsize=(12, 8))
    
    # Calcular acurácia por classe
    class_accuracy = []
    for i in range(len(class_names)):
        if cm[i].sum() > 0:
            acc = cm[i, i] / cm[i].sum()
            class_accuracy.append(acc)
        else:
            class_accuracy.append(0)
    
    # Ordenar por acurácia
    sorted_indices = np.argsort(class_accuracy)
    sorted_classes = [class_names[i] for i in sorted_indices]
    sorted_accuracy = [class_accuracy[i] for i in sorted_indices]
    
    colors = ['#e74c3c' if acc < 0.5 else '#f39c12' if acc < 0.7 else '#2ecc71' 
              for acc in sorted_accuracy]
    
    plt.barh(sorted_classes, sorted_accuracy, color=colors)
    plt.xlabel('Acurácia')
    plt.title('Acurácia por Espécie', fontsize=14, fontweight='bold')
    plt.xlim(0, 1)
    
    # Adicionar linha de referência
    plt.axvline(x=0.7, color='orange', linestyle='--', alpha=0.5, label='70%')
    plt.axvline(x=0.9, color='green', linestyle='--', alpha=0.5, label='90%')
    plt.legend()
    
    plt.tight_layout()
    
    filename = 'accuracy_per_class.png'
    plt.savefig(charts_folder / filename, dpi=100, bbox_inches='tight')
    plt.close()
    chart_files.append(filename)
    
    # 4. Distribuição de Predições (se houver predict_proba)
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)
        
        # Pegar a confiança máxima de cada predição
        max_proba = np.max(y_proba, axis=1)
        
        plt.figure(figsize=(10, 6))
        plt.hist(max_proba, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
        plt.xlabel('Confiança da Predição (Probabilidade Máxima)')
        plt.ylabel('Frequência')
        plt.title('Distribuição de Confiança das Predições', fontsize=14, fontweight='bold')
        plt.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='50%')
        plt.axvline(x=0.8, color='green', linestyle='--', alpha=0.5, label='80%')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filename = 'prediction_confidence.png'
        plt.savefig(charts_folder / filename, dpi=100, bbox_inches='tight')
        plt.close()
        chart_files.append(filename)
    
    return chart_files

def plot_training_history(train_scores, val_scores=None):
    """
    Plota histórico de treinamento (útil para Gradient Boosting)
    
    Args:
        train_scores: Scores de treino por iteração
        val_scores: Scores de validação por iteração (opcional)
    
    Returns:
        str: Nome do arquivo gerado
    """
    charts_folder = Config.CHARTS_FOLDER
    
    plt.figure(figsize=(10, 6))
    
    iterations = range(1, len(train_scores) + 1)
    plt.plot(iterations, train_scores, label='Treino', color='blue', linewidth=2)
    
    if val_scores is not None:
        plt.plot(iterations, val_scores, label='Validação', color='orange', linewidth=2)
    
    plt.xlabel('Iteração')
    plt.ylabel('Score')
    plt.title('Histórico de Treinamento', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filename = 'training_history.png'
    plt.savefig(charts_folder / filename, dpi=100, bbox_inches='tight')
    plt.close()
    
    return filename