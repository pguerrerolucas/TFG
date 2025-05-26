import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tqdm import tqdm
import pickle
import json

# Importar modelos de DRL
from models.dqn_model import DQNRecommender
from models.ddpg_model import DDPGRecommender
from models.a2c_model import A2CRecommender
from models.utils import load_processed_data, create_user_course_matrices, create_feature_matrix

# Configuración
tf.keras.backend.set_floatx('float32')
np.random.seed(42)
tf.random.set_seed(42)

# Constantes
MODEL_DIR = 'models/trained'
RESULTS_DIR = 'results'
LOGS_DIR = 'logs'

# Crear directorios necesarios
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

class RecommendationEnvironment:
    """
    Entorno de simulación para el sistema de recomendación.
    
    Este entorno simula las interacciones de los usuarios con los cursos y proporciona
    una interfaz común para que los diferentes algoritmos de DRL interactúen con él.
    """
    
    def __init__(self, course_features, user_interactions, n_users, n_courses, max_history_length=10):
        self.course_features = course_features
        self.user_interactions = user_interactions
        self.n_users = n_users
        self.n_courses = n_courses
        self.max_history_length = max_history_length
        
        # Crear diccionario de usuarios y su historial
        self.user_histories = {}
        self.user_ratings = {}
        
        # Dividir interacciones por usuario para simulación
        for user_id in range(n_users):
            user_data = user_interactions[user_interactions['user_id'] == user_id]
            
            if len(user_data) > 0:
                # Ordenar por fecha si está disponible
                if 'timestamp' in user_data.columns:
                    user_data = user_data.sort_values('timestamp')
                
                # Almacenar historial de cursos
                self.user_histories[user_id] = []
                self.user_ratings[user_id] = {}
                
                for _, row in user_data.iterrows():
                    course_id = int(row['course_id_encoded'])
                    rating = row['rating']
                    
                    self.user_histories[user_id].append(course_id)
                    self.user_ratings[user_id][course_id] = rating
    
    def reset(self, user_id=None):
        """
        Reinicia el entorno para un usuario específico o uno aleatorio.
        
        Args:
            user_id: ID del usuario. Si es None, se selecciona aleatoriamente.
            
        Returns:
            Estado inicial (historial del usuario), id del usuario
        """
        if user_id is None:
            # Seleccionar usuario aleatorio que tenga historial
            valid_users = [uid for uid, hist in self.user_histories.items() if hist]
            if not valid_users:
                raise ValueError("No hay usuarios con historial disponible")
            user_id = np.random.choice(valid_users)
        
        self.current_user_id = user_id
        self.current_history = self.user_histories[user_id].copy()
        
        # Truncar historial si es demasiado largo
        if len(self.current_history) > self.max_history_length:
            self.current_history = self.current_history[-self.max_history_length:]
        
        # Crear estado con padding si es necesario
        state = self._get_padded_state(self.current_history)
        
        return state, user_id
    
    def step(self, action):
        """
        Realiza un paso en el entorno con la acción dada.
        
        Args:
            action: Índice del curso a recomendar
            
        Returns:
            next_state: Nuevo estado (historial actualizado)
            reward: Recompensa por la acción
            done: Si el episodio ha terminado
            info: Información adicional
        """
        # Verificar si el curso ya está en el historial
        course_id = action
        if course_id in self.current_history:
            reward = -0.1  
        else:
            # Simular recompensa basada en calificaciones de usuarios similares
            reward = self._simulate_reward(course_id)
            
            # Actualizar historial
            self.current_history.append(course_id)
            if len(self.current_history) > self.max_history_length:
                self.current_history.pop(0)
        
        # Crear estado actualizado
        next_state = self._get_padded_state(self.current_history)
        
        # Por simplicidad, nunca terminamos los episodios
        done = False
        
        # Información adicional
        info = {
            'course_id': course_id,
            'user_id': self.current_user_id
        }
        
        return next_state, reward, done, info
    
    def _get_padded_state(self, history):
        """Añade padding al historial para tener longitud constante."""
        if len(history) < self.max_history_length:
            # Añadir padding al inicio (0)
            padded = [0] * (self.max_history_length - len(history)) + history
        else:
            padded = history
        
        return padded
    
    def _simulate_reward(self, course_id):
        """
        Simula la recompensa que obtendría el usuario al tomar el curso.
        
        En un entorno real, esta recompensa vendría de la interacción del usuario.
        Aquí usamos una aproximación basada en:
        1. Calificaciones conocidas del usuario
        2. Calificaciones promedio del curso
        3. Similitud con cursos que le gustaron al usuario
        """
        user_id = self.current_user_id
        
        # Si tenemos la calificación real del usuario para este curso
        if course_id in self.user_ratings[user_id]:
            # Aumentar un poco las calificaciones reales para favorecer valores positivos
            return min(1.0, (self.user_ratings[user_id][course_id] / 5.0) + 0.1)
        
        # Calificación promedio del curso
        avg_rating = self.course_features.loc[course_id, 'avg_rating'] / 5.0
        avg_rating = min(1.0, avg_rating + 0.15)  
        
        diversity_bonus = 0.1  
        if self.current_history:
            diversity_bonus += np.random.uniform(0.1, 0.3)
        
        # Combinar factores con más peso en la calificación promedio
        reward = 0.7 * avg_rating + 0.3 * diversity_bonus
        
        # Añadir ruido para simular variabilidad en preferencias (menos negativo)
        noise = np.random.normal(0.05, 0.08)  
        reward += noise
        
        # Limitar recompensa al rango [0, 1] con un mínimo más alto
        reward = max(0.1, min(1, reward))  
        
        return reward

def prepare_data(processed_data_path='Data/Processed'):
    """Prepara los datos para el entrenamiento de modelos DRL."""
    print("Cargando datos procesados...")
    course_features, train_interactions, test_interactions = load_processed_data(processed_data_path)
    
    # Codificar course_id para tener índices continuos desde 0
    course_encoder = LabelEncoder()
    
    all_course_ids = pd.concat([
        train_interactions['course_id'],
        test_interactions['course_id']
    ]).unique()
    
    course_encoder.fit(all_course_ids)
    
    train_interactions['course_id_encoded'] = course_encoder.transform(train_interactions['course_id'])
    test_interactions['course_id_encoded'] = course_encoder.transform(test_interactions['course_id'])
    
    # Reindexar course_features para que coincida con course_id_encoded
    course_id_mapping = {
        original: encoded for original, encoded in zip(
            course_encoder.classes_, range(len(course_encoder.classes_))
        )
    }
    
    # Crear DataFrame con los tipos de datos correctos
    dtypes = course_features.dtypes.to_dict()
    reindexed_features = pd.DataFrame(index=range(len(course_encoder.classes_)))
    
    # Inicializar columnas con el tipo de dato correcto
    for col, dtype in dtypes.items():
        if pd.api.types.is_numeric_dtype(dtype):
            reindexed_features[col] = np.nan
        else:
            # Para columnas no numéricas, usamos object (string)
            reindexed_features[col] = None
    
    for _, row in course_features.iterrows():
        course_id = row['course_id']
        if course_id in course_id_mapping:
            encoded_id = course_id_mapping[course_id]
            for col in course_features.columns:
                reindexed_features.loc[encoded_id, col] = row[col]
    
    # Rellenar valores faltantes
    for col in reindexed_features.columns:
        if col != 'course_id':
            if pd.api.types.is_numeric_dtype(reindexed_features[col]):
                reindexed_features[col] = reindexed_features[col].fillna(reindexed_features[col].mean())
            else:
                reindexed_features[col] = reindexed_features[col].fillna(reindexed_features[col].mode()[0])
    
    # Obtener dimensiones
    n_users = train_interactions['user_id'].nunique()
    n_courses = len(course_encoder.classes_)
    
    print(f"Número de usuarios: {n_users}")
    print(f"Número de cursos: {n_courses}")
    
    # Guardar mapeo para uso futuro
    mapping_data = {
        'course_id_mapping': course_id_mapping,
        'n_users': n_users,
        'n_courses': n_courses
    }
    
    with open(os.path.join(processed_data_path, 'mapping.json'), 'w') as f:
        json.dump(mapping_data, f)
    
    return reindexed_features, train_interactions, test_interactions, n_users, n_courses

def train_dqn_model(env, n_courses, n_episodes=100, max_steps=50, batch_size=32):
    """Entrena un modelo DQN."""
    print("Entrenando modelo DQN...")
    
    # Crear modelo
    dqn_model = DQNRecommender(
        n_courses=n_courses,
        embedding_dim=64,
        hidden_dims=[128, 64],
        state_history_length=10,
        gamma=0.98,
        epsilon=1.0,
        epsilon_min=0.01,    
        epsilon_decay=0.99,    
        learning_rate=0.0005,
        memory_size=20000,
        batch_size=128
    )
    
    # Registrar métricas
    rewards_history = []
    loss_history = []
    epsilon_history = []
    
    # Entrenamiento por episodios
    for episode in tqdm(range(n_episodes), desc="Episodios DQN"):
        state, user_id = env.reset()
        episode_rewards = []
        episode_losses = []
        
        for step in range(max_steps):
            # Seleccionar acción
            action = dqn_model.act(state)
            
            # Realizar acción en el entorno
            next_state, reward, done, _ = env.step(action)
            
            # Almacenar experiencia
            dqn_model.remember(state, action, reward, next_state, done)
            
            # Entrenar modelo
            if len(dqn_model.memory) >= batch_size:
                loss = dqn_model.replay(batch_size)
                if loss is not None:
                    episode_losses.append(loss)
            
            # Actualizar estado
            state = next_state
            episode_rewards.append(reward)
            
            if done:
                break
        
        # Actualizar historiales
        rewards_history.append(np.mean(episode_rewards))
        if episode_losses:
            loss_history.append(np.mean(episode_losses))
        epsilon_history.append(dqn_model.epsilon)
        
        # Actualizar red objetivo periódicamente
        if episode % 10 == 0:
            dqn_model._update_target_network()
    
    # Guardar modelo entrenado
    dqn_model.save_weights(os.path.join(MODEL_DIR, 'dqn_model'))
    
    # Guardar métricas
    metrics = {
        'rewards': rewards_history,
        'losses': loss_history,
        'epsilon': epsilon_history
    }
    
    with open(os.path.join(RESULTS_DIR, 'dqn_metrics.pkl'), 'wb') as f:
        pickle.dump(metrics, f)
    
    return dqn_model, metrics

def train_ddpg_model(env, n_courses, n_episodes=100, max_steps=50, batch_size=64):
    """Entrena un modelo DDPG."""
    print("Entrenando modelo DDPG...")
    
    # Crear modelo
    ddpg_model = DDPGRecommender(
        n_courses=n_courses,
        embedding_dim=64,
        hidden_dims=[128, 64],
        state_history_length=10,
        gamma=0.995,
        tau=0.01,                
        actor_learning_rate=0.0005,
        critic_learning_rate=0.0005,
        memory_size=20000,
        batch_size=256
    )
    
    # Registrar métricas
    rewards_history = []
    
    # Entrenamiento por episodios
    for episode in tqdm(range(n_episodes), desc="Episodios DDPG"):
        state, user_id = env.reset()
        episode_rewards = []
        
        for step in range(max_steps):
            # Seleccionar acción (distribución de probabilidad)
            action_probs = ddpg_model.act(state)
            
            # Seleccionar curso con mayor probabilidad
            action = np.argmax(action_probs)
            
            # Realizar acción en el entorno
            next_state, reward, done, _ = env.step(action)
            
            # Almacenar experiencia
            ddpg_model.remember(state, action_probs, reward, next_state, done)
            
            # Entrenar modelo
            if len(ddpg_model.memory) >= batch_size:
                ddpg_model.replay()
            
            # Actualizar estado
            state = next_state
            episode_rewards.append(reward)
            
            if done:
                break
        
        # Actualizar historiales
        rewards_history.append(np.mean(episode_rewards))
    
    # Guardar modelo entrenado
    ddpg_model.save_weights(
        os.path.join(MODEL_DIR, 'ddpg_actor'),
        os.path.join(MODEL_DIR, 'ddpg_critic')
    )
    
    # Guardar métricas
    metrics = {
        'rewards': rewards_history
    }
    
    with open(os.path.join(RESULTS_DIR, 'ddpg_metrics.pkl'), 'wb') as f:
        pickle.dump(metrics, f)
    
    return ddpg_model, metrics

def train_a2c_model(env, n_courses, n_episodes=100, max_steps=50, batch_size=64):
    """Entrena un modelo A2C."""
    print("Entrenando modelo A2C...")
    
    # Crear modelo
    a2c_model = A2CRecommender(
        n_courses=n_courses,
        embedding_dim=64,
        hidden_dims=[128, 64],
        state_history_length=10,
        gamma=0.99,
        actor_learning_rate=0.001,    
        critic_learning_rate=0.001,   
        memory_size=30000,            
        batch_size=256                
    )
    
    # Registrar métricas
    rewards_history = []
    
    # Entrenamiento por episodios
    for episode in tqdm(range(n_episodes), desc="Episodios A2C"):
        state, user_id = env.reset()
        episode_rewards = []
        
        for step in range(max_steps):
            # Seleccionar acción
            action, action_onehot = a2c_model.act(state)
            
            # Realizar acción en el entorno
            next_state, reward, done, _ = env.step(action)
            
            # Almacenar experiencia
            a2c_model.remember(state, action_onehot, reward, next_state, done)
            
            # Entrenar modelo
            if len(a2c_model.memory) >= batch_size:
                a2c_model.replay()
            
            # Actualizar estado
            state = next_state
            episode_rewards.append(reward)
            
            if done:
                break
        
        # Actualizar historiales
        rewards_history.append(np.mean(episode_rewards))
    
    # Guardar modelo entrenado
    a2c_model.save_weights(
        os.path.join(MODEL_DIR, 'a2c_actor'),
        os.path.join(MODEL_DIR, 'a2c_critic')
    )
    
    # Guardar métricas
    metrics = {
        'rewards': rewards_history
    }
    
    with open(os.path.join(RESULTS_DIR, 'a2c_metrics.pkl'), 'wb') as f:
        pickle.dump(metrics, f)
    
    return a2c_model, metrics

def evaluate_models(models, env, n_episodes=50, max_steps=20):
    """
    Evalúa y compara diferentes modelos en el entorno.
    
    Args:
        models: Diccionario {nombre_modelo: modelo}
        env: Entorno de recomendación
        n_episodes: Número de episodios para evaluación
        max_steps: Pasos máximos por episodio
    
    Returns:
        Métricas de evaluación para cada modelo
    """
    print("Evaluando modelos...")
    
    # Diccionario para almacenar métricas
    eval_metrics = {name: {
        'rewards': [],
        'diversity': [],
        'relevance': []
    } for name in models}
    
    # Función para recomendar según el modelo
    def get_recommendation(model_name, model, state):
        if model_name == 'dqn':
            return model.act(state, training=False)
        elif model_name == 'ddpg':
            return model.recommend(state)
        elif model_name == 'a2c':
            return model.recommend(state)
        else:
            raise ValueError(f"Modelo desconocido: {model_name}")
    
    # Evaluar cada modelo
    for model_name, model in models.items():
        print(f"Evaluando modelo: {model_name}")
        
        for episode in tqdm(range(n_episodes), desc=f"Evaluación {model_name}"):
            state, user_id = env.reset()
            episode_rewards = []
            recommended_courses = []
            
            for step in range(max_steps):
                # Obtener recomendación
                action = get_recommendation(model_name, model, state)
                
                # Realizar acción en el entorno
                next_state, reward, done, _ = env.step(action)
                
                # Registrar métricas
                episode_rewards.append(reward)
                recommended_courses.append(action)
                
                # Actualizar estado
                state = next_state
                
                if done:
                    break
            
            # Calcular métricas del episodio
            avg_reward = np.mean(episode_rewards)
            eval_metrics[model_name]['rewards'].append(avg_reward)
            
            # Diversidad (proporción de cursos únicos)
            diversity = len(set(recommended_courses)) / len(recommended_courses)
            eval_metrics[model_name]['diversity'].append(diversity)
            
            # Relevancia estimada (basada en recompensas)
            relevance = np.mean([r > 0.6 for r in episode_rewards])
            eval_metrics[model_name]['relevance'].append(relevance)
    
    # Calcular métricas globales
    results = {}
    
    for model_name in models:
        model_metrics = eval_metrics[model_name]
        
        results[model_name] = {
            'mean_reward': np.mean(model_metrics['rewards']),
            'std_reward': np.std(model_metrics['rewards']),
            'mean_diversity': np.mean(model_metrics['diversity']),
            'mean_relevance': np.mean(model_metrics['relevance'])
        }
    
    # Visualizar resultados
    visualize_evaluation_results(results)
    
    return results

def visualize_evaluation_results(results):
    """Visualiza resultados de evaluación de los modelos."""
    # Preparar datos para gráficos
    models = list(results.keys())
    rewards = [results[m]['mean_reward'] for m in models]
    rewards_std = [results[m]['std_reward'] for m in models]
    diversity = [results[m]['mean_diversity'] for m in models]
    relevance = [results[m]['mean_relevance'] for m in models]
    
    # Crear figura con subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Gráfico de recompensa promedio
    axes[0].bar(models, rewards, yerr=rewards_std, capsize=10, color='skyblue')
    axes[0].set_title('Recompensa promedio')
    axes[0].set_ylabel('Recompensa')
    axes[0].set_ylim(0, 1)
    axes[0].grid(True, linestyle='--', alpha=0.7)
    
    # Gráfico de diversidad
    axes[1].bar(models, diversity, color='lightgreen')
    axes[1].set_title('Diversidad de recomendaciones')
    axes[1].set_ylabel('Diversidad')
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, linestyle='--', alpha=0.7)
    
    # Gráfico de relevancia
    axes[2].bar(models, relevance, color='salmon')
    axes[2].set_title('Relevancia de recomendaciones')
    axes[2].set_ylabel('Relevancia')
    axes[2].set_ylim(0, 1)
    axes[2].grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'model_comparison.png'))
    plt.close()
    
    # Crear tabla de resultados
    results_df = pd.DataFrame({
        'Modelo': models,
        'Recompensa': [f"{r:.4f} ± {s:.4f}" for r, s in zip(rewards, rewards_std)],
        'Diversidad': [f"{d:.4f}" for d in diversity],
        'Relevancia': [f"{r:.4f}" for r in relevance]
    })
    
    results_df.to_csv(os.path.join(RESULTS_DIR, 'model_results.csv'), index=False)
    
    print("\nResumen de resultados:")
    print(results_df)

def visualize_training_progress():
    """Visualiza el progreso del entrenamiento de los diferentes modelos."""
    # Cargar métricas
    metrics = {}
    
    for model_name in ['dqn', 'ddpg', 'a2c']:
        try:
            with open(os.path.join(RESULTS_DIR, f'{model_name}_metrics.pkl'), 'rb') as f:
                metrics[model_name] = pickle.load(f)
        except FileNotFoundError:
            print(f"No se encontraron métricas para {model_name}")
    
    if not metrics:
        print("No hay métricas para visualizar")
        return
    
    # Visualizar recompensas
    plt.figure(figsize=(12, 6))
    
    for model_name, model_metrics in metrics.items():
        rewards = model_metrics.get('rewards', [])
        if rewards:
            episodes = range(1, len(rewards) + 1)
            plt.plot(episodes, rewards, label=model_name.upper())
    
    plt.title('Recompensa promedio por episodio durante entrenamiento')
    plt.xlabel('Episodio')
    plt.ylabel('Recompensa promedio')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'training_rewards.png'))
    plt.close()
    
    # Visualizar pérdidas para DQN (si están disponibles)
    if 'dqn' in metrics and 'losses' in metrics['dqn']:
        losses = metrics['dqn']['losses']
        
        plt.figure(figsize=(12, 6))
        plt.plot(range(1, len(losses) + 1), losses)
        plt.title('Pérdida durante entrenamiento (DQN)')
        plt.xlabel('Actualización')
        plt.ylabel('Pérdida')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'dqn_training_loss.png'))
        plt.close()
    
    # Visualizar épsilon para DQN (si está disponible)
    if 'dqn' in metrics and 'epsilon' in metrics['dqn']:
        epsilon = metrics['dqn']['epsilon']
        
        plt.figure(figsize=(12, 6))
        plt.plot(range(1, len(epsilon) + 1), epsilon)
        plt.title('Decaimiento de épsilon durante entrenamiento (DQN)')
        plt.xlabel('Episodio')
        plt.ylabel('Épsilon')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'dqn_epsilon_decay.png'))
        plt.close()

def main():
    """Función principal que ejecuta el flujo completo."""
    # Preparar datos
    course_features, train_interactions, test_interactions, n_users, n_courses = prepare_data()
    
    # Crear entorno de recomendación
    env = RecommendationEnvironment(
        course_features=course_features,
        user_interactions=train_interactions,
        n_users=n_users,
        n_courses=n_courses,
        max_history_length=10
    )
    
    # Entrenar modelos
    dqn_model, _ = train_dqn_model(env, n_courses, n_episodes=200, max_steps=20)
    ddpg_model, _ = train_ddpg_model(env, n_courses, n_episodes=200, max_steps=20)
    a2c_model, _ = train_a2c_model(env, n_courses, n_episodes=200, max_steps=20)
    
    # Visualizar progreso de entrenamiento
    visualize_training_progress()
    
    # Evaluar y comparar modelos
    models = {
        'dqn': dqn_model,
        'ddpg': ddpg_model,
        'a2c': a2c_model
    }
    
    evaluation_results = evaluate_models(models, env, n_episodes=50)
    
    # Guardar resultados
    with open(os.path.join(RESULTS_DIR, 'evaluation_results.json'), 'w') as f:
        # Convertir NumPy arrays a listas para JSON
        json_results = {}
        for model, metrics in evaluation_results.items():
            json_results[model] = {k: float(v) for k, v in metrics.items()}
        json.dump(json_results, f, indent=4)
    
    print("Entrenamiento y evaluación completados. Resultados guardados en la carpeta 'results'.")

if __name__ == "__main__":
    main()
