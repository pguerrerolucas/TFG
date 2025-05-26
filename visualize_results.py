import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Directorios
RESULTS_DIR = 'results'
MODEL_DIR = 'models/trained'

def load_training_metrics():
    """Carga las métricas de entrenamiento de los modelos."""
    metrics = {}
    
    # Modelos a cargar
    model_names = ['dqn', 'ddpg', 'a2c']
    
    for model_name in model_names:
        metric_file = os.path.join(RESULTS_DIR, f'{model_name}_metrics.pkl')
        if os.path.exists(metric_file):
            try:
                with open(metric_file, 'rb') as f:
                    metrics[model_name] = pickle.load(f)
                print(f"Métricas cargadas para {model_name}")
            except Exception as e:
                print(f"Error al cargar métricas para {model_name}: {e}")
        else:
            print(f"No se encontraron métricas para {model_name}")
    
    return metrics

def load_evaluation_results():
    """Carga los resultados de evaluación comparativa."""
    eval_file = os.path.join(RESULTS_DIR, 'evaluation_results.json')
    
    if os.path.exists(eval_file):
        try:
            with open(eval_file, 'r') as f:
                results = json.load(f)
            print("Resultados de evaluación cargados correctamente")
            return results
        except Exception as e:
            print(f"Error al cargar resultados de evaluación: {e}")
    else:
        print("No se encontraron resultados de evaluación")
    
    return None

def plot_training_rewards(metrics):
    """Gráfica la recompensa durante el entrenamiento para cada modelo."""
    plt.figure(figsize=(12, 6))
    
    for model_name, model_metrics in metrics.items():
        if 'rewards' in model_metrics:
            rewards = model_metrics['rewards']
            episodes = range(1, len(rewards) + 1)
            plt.plot(episodes, rewards, label=f"{model_name.upper()}")
    
    plt.title('Recompensa promedio durante entrenamiento')
    plt.xlabel('Episodio')
    plt.ylabel('Recompensa promedio')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'training_rewards_comparison.png'))
    plt.show()

def plot_model_comparison(eval_results):
    """Gráfica la comparación de los modelos según diferentes métricas."""
    if not eval_results:
        print("No hay resultados de evaluación para comparar")
        return
    
    # Extraer datos para gráficas
    models = list(eval_results.keys())
    metrics = ['mean_reward', 'mean_diversity', 'mean_relevance']
    metric_labels = ['Recompensa', 'Diversidad', 'Relevancia']
    
    # Crear figura con subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        values = [eval_results[m][metric] for m in models]
        
        # Barplot para la métrica actual
        axes[i].bar(models, values, color=sns.color_palette("pastel")[i])
        axes[i].set_title(f'{label} por modelo')
        axes[i].set_ylabel(label)
        axes[i].set_ylim(0, 1)
        axes[i].grid(True, linestyle='--', alpha=0.5)
        
        # Añadir valores encima de las barras
        for j, v in enumerate(values):
            axes[i].text(j, v + 0.02, f"{v:.3f}", ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'model_comparison_detailed.png'))
    plt.show()

def print_metrics_summary(metrics):
    """Imprime un resumen de las métricas de entrenamiento."""
    print("\n===== RESUMEN DE MÉTRICAS DE ENTRENAMIENTO =====")
    
    for model_name, model_metrics in metrics.items():
        print(f"\nModelo: {model_name.upper()}")
        
        if 'rewards' in model_metrics:
            rewards = model_metrics['rewards']
            print(f"  • Recompensa final: {rewards[-1]:.4f}")
            print(f"  • Recompensa promedio: {np.mean(rewards):.4f}")
            print(f"  • Recompensa máxima: {np.max(rewards):.4f}")
        
        if 'losses' in model_metrics:
            losses = model_metrics['losses']
            print(f"  • Pérdida final: {losses[-1]:.4f}")
            print(f"  • Pérdida promedio: {np.mean(losses):.4f}")

def print_evaluation_summary(eval_results):
    """Imprime un resumen de los resultados de evaluación."""
    if not eval_results:
        return
    
    print("\n===== RESUMEN DE RESULTADOS DE EVALUACIÓN =====")
    
    # Crear tabla para visualización
    data = []
    
    for model_name, results in eval_results.items():
        row = {
            'Modelo': model_name.upper(),
            'Recompensa': f"{results['mean_reward']:.4f}",
            'Diversidad': f"{results['mean_diversity']:.4f}",
            'Relevancia': f"{results['mean_relevance']:.4f}"
        }
        data.append(row)
    
    # Convertir a DataFrame para mejor visualización
    df = pd.DataFrame(data)
    print(df.to_string(index=False))
    
    # Identificar el mejor modelo
    best_model = max(eval_results.items(), key=lambda x: x[1]['mean_reward'])
    print(f"\nMejor modelo según recompensa: {best_model[0].upper()} ({best_model[1]['mean_reward']:.4f})")

def main():
    """Función principal."""
    print("=== Visualización de Resultados del Sistema de Recomendación ===\n")
    
    # Cargar métricas y resultados
    training_metrics = load_training_metrics()
    eval_results = load_evaluation_results()
    
    # Verificar si se cargaron datos
    if not training_metrics:
        print("No se encontraron métricas de entrenamiento")
        return
    
    # Mostrar resúmenes
    print_metrics_summary(training_metrics)
    print_evaluation_summary(eval_results)
    
    # Visualizar gráficas
    print("\nGenerando gráficas...")
    plot_training_rewards(training_metrics)
    
    if eval_results:
        plot_model_comparison(eval_results)

if __name__ == "__main__":
    main()
