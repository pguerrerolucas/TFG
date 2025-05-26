import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Concatenate, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import random
from collections import deque

class A2CRecommender:
    """
    Sistema de recomendación basado en Actor-Critic con Experience Replay (A2C).
    
    Arquitectura híbrida que combina estimación de valor (critic) y aprendizaje 
    de política (actor) para optimizar las recomendaciones.
    """
    
    def __init__(
        self, 
        n_courses,
        embedding_dim=32,
        hidden_dims=[64, 32],
        state_history_length=10,
        gamma=0.99,
        actor_learning_rate=0.0001,
        critic_learning_rate=0.001,
        memory_size=10000,
        batch_size=64
    ):
        self.n_courses = n_courses
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims
        self.state_history_length = state_history_length
        self.gamma = gamma
        self.actor_lr = actor_learning_rate
        self.critic_lr = critic_learning_rate
        self.memory_size = memory_size
        self.batch_size = batch_size
        
        # Buffer de experiencias pasadas
        self.memory = deque(maxlen=memory_size)
        
        # Construir modelos
        self.actor = self._build_actor()
        self.critic = self._build_critic()
    
    def _build_actor(self):
        """Construye la red del actor que determina la política de recomendación."""
        # Historial de cursos del usuario
        input_history = Input(shape=(self.state_history_length,), name='actor_history_input')
        
        # Embedding para representar los cursos
        embedding = Embedding(
            input_dim=self.n_courses + 1,
            output_dim=self.embedding_dim,
            mask_zero=True,
            name='actor_embedding'
        )(input_history)
        
        # LSTM para procesar la secuencia
        lstm_out = LSTM(units=self.hidden_dims[0], name='actor_lstm')(embedding)
        
        # Capas densas
        x = Dense(self.hidden_dims[0], activation='relu', name='actor_dense_1')(lstm_out)
        x = Dropout(0.3)(x)
        
        for i, dim in enumerate(self.hidden_dims[1:], 2):
            x = Dense(dim, activation='relu', name=f'actor_dense_{i}')(x)
            x = Dropout(0.3)(x)
        
        # Capa de salida: distribución de probabilidad sobre cursos (logits)
        logits = Dense(self.n_courses, activation=None, name='actor_logits')(x)
        
        def custom_loss(y_true, y_pred):
            """
            Loss personalizada para el actor.
            y_true: [acciones, ventajas]
            y_pred: logits para distribución de probabilidad
            """
            actions = y_true[:, :self.n_courses]
            advantages = y_true[:, self.n_courses:]
            
            # Convertir logits a probabilidades con softmax
            probs = tf.nn.softmax(y_pred)
            
            # Calcular log prob de acciones tomadas
            log_probs = tf.math.log(tf.reduce_sum(actions * probs, axis=1) + 1e-10)
            
            # Loss de Policy Gradient
            policy_loss = -tf.reduce_mean(log_probs * advantages)
            
            # Entropía para fomentar exploración
            entropy = -tf.reduce_mean(tf.reduce_sum(probs * tf.math.log(probs + 1e-10), axis=1))
            
            return policy_loss - 0.01 * entropy
        
        model = Model(inputs=input_history, outputs=logits)
        model.compile(optimizer=Adam(learning_rate=self.actor_lr), loss=custom_loss)
        
        return model
    
    def _build_critic(self):
        """Construye la red del crítico (función de valor)."""
        # Entrada para el estado (historial de cursos)
        state_input = Input(shape=(self.state_history_length,), name='critic_state_input')
        
        # Embedding para el estado
        state_embedding = Embedding(
            input_dim=self.n_courses + 1,
            output_dim=self.embedding_dim,
            mask_zero=True,
            name='critic_state_embedding'
        )(state_input)
        
        # LSTM para procesar secuencia
        lstm_out = LSTM(units=self.hidden_dims[0], name='critic_lstm')(state_embedding)
        
        # Capas densas
        x = Dense(self.hidden_dims[0], activation='relu', name='critic_dense_1')(lstm_out)
        x = Dropout(0.3)(x)
        
        for i, dim in enumerate(self.hidden_dims[1:], 2):
            x = Dense(dim, activation='relu', name=f'critic_dense_{i}')(x)
            x = Dropout(0.3)(x)
        
        # Salida: valor estimado del estado
        value = Dense(1, activation='linear', name='critic_output')(x)
        
        model = Model(inputs=state_input, outputs=value)
        model.compile(optimizer=Adam(learning_rate=self.critic_lr), loss='mse')
        
        return model
    
    def remember(self, state, action, reward, next_state, done):
        """Almacena una experiencia en la memoria."""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """
        Selecciona una acción basada en la política actual.
        
        Args:
            state: Historial de cursos del usuario
            
        Returns:
            Acción seleccionada (ID del curso) y one-hot encoding de la acción
        """
        # Predecir logits
        logits = self.actor.predict(np.array([state]), verbose=0)[0]
        
        # Aplicar softmax para obtener probabilidades
        probs = tf.nn.softmax(logits).numpy()
        
        # Excluir cursos ya vistos
        for course_id in state:
            if course_id > 0:  # Ignorar padding (0)
                probs[course_id-1] = 0
        
        # Renormalizar si es necesario
        probs_sum = np.sum(probs)
        if probs_sum > 0:
            probs = probs / probs_sum
        else:
            # Si todos los cursos ya han sido vistos, asignar probabilidad uniforme
            probs = np.ones(self.n_courses) / self.n_courses
        
        # Seleccionar acción según la distribución de probabilidad
        action = np.random.choice(self.n_courses, p=probs)
        
        # Crear one-hot encoding de la acción
        action_onehot = np.zeros(self.n_courses)
        action_onehot[action] = 1
        
        return action, action_onehot
    
    def recommend(self, state):
        """
        Recomienda un curso para el usuario (modo de inferencia).
        
        Args:
            state: Historial de cursos del usuario
            
        Returns:
            Índice del curso recomendado
        """
        # Predecir logits
        logits = self.actor.predict(np.array([state]), verbose=0)[0]
        
        # Aplicar máscara para cursos ya vistos
        for course_id in state:
            if course_id > 0:
                logits[course_id-1] = float('-inf')
        
        # Seleccionar curso con mayor valor
        return np.argmax(logits)
    
    def replay(self):
        """Entrena las redes con experiencias almacenadas."""
        if len(self.memory) < self.batch_size:
            return
        
        # Muestrear experiencias aleatorias
        minibatch = random.sample(self.memory, self.batch_size)
        
        states = []
        action_onehots = []
        values = []
        returns = []
        advantages = []
        
        for state, action_onehot, reward, next_state, done in minibatch:
            states.append(state)
            action_onehots.append(action_onehot)
            
            # Calcular retorno (reward discounted)
            if done:
                target_value = reward
            else:
                # Predecir valor del siguiente estado
                next_value = self.critic.predict(np.array([next_state]), verbose=0)[0]
                target_value = reward + self.gamma * next_value
            
            # Predecir valor del estado actual
            current_value = self.critic.predict(np.array([state]), verbose=0)[0]
            
            # Calcular ventaja (usado para entrenar el actor)
            advantage = target_value - current_value
            
            values.append(current_value)
            returns.append(target_value)
            advantages.append(advantage)
        
        states = np.array(states)
        action_onehots = np.array(action_onehots)
        returns = np.array(returns)
        advantages = np.array(advantages)
        
        # Entrenar crítico para predecir mejor los valores
        self.critic.fit(states, returns, epochs=1, verbose=0, batch_size=self.batch_size)
        
        # Preparar datos para entrenar el actor
        actor_targets = np.hstack([action_onehots, advantages])
        
        # Entrenar actor para mejorar la política
        self.actor.fit(states, actor_targets, epochs=1, verbose=0, batch_size=self.batch_size)
    
    def train_episode(self, user_histories, course_features, n_interactions=100):
        """
        Entrena el modelo para un episodio completo.
        
        Args:
            user_histories: Diccionario {user_id: [lista de course_ids]}
            course_features: DataFrame con características de los cursos
            n_interactions: Número de interacciones a simular
            
        Returns:
            reward_history: Lista de recompensas obtenidas
        """
        reward_history = []
        
        # Seleccionar usuarios aleatorios para el episodio
        user_ids = list(user_histories.keys())
        episode_users = np.random.choice(user_ids, n_interactions, replace=True)
        
        for user_id in episode_users:
            # Obtener el historial actual del usuario
            history = user_histories[user_id].copy()
            
            if len(history) == 0:
                continue
                
            # Asegurar que el historial tenga el tamaño correcto
            if len(history) < self.state_history_length:
                # Padding al inicio
                state = [0] * (self.state_history_length - len(history)) + history
            else:
                # Tomar los últimos elementos
                state = history[-self.state_history_length:]
            
            # Seleccionar acción según la política
            action, action_onehot = self.act(state)
            
            # Obtener recompensa simulada
            reward = self._simulate_reward(action, history, course_features)
            
            # Actualizar historial con el nuevo curso
            next_history = history + [action + 1]
            
            if len(next_history) < self.state_history_length:
                next_state = [0] * (self.state_history_length - len(next_history)) + next_history
            else:
                next_state = next_history[-self.state_history_length:]
            
            done = False
            
            self.remember(state, action_onehot, reward, next_state, done)
            
            # Entrenar con batch de experiencias
            if len(self.memory) >= self.batch_size:
                self.replay()
            
            reward_history.append(reward)
            
            # Actualizar el historial del usuario para la próxima interacción
            user_histories[user_id] = next_history
        
        return reward_history
    
    def _simulate_reward(self, action, history, course_features, avg_rating_col='avg_rating'):
        """
        Simula la recompensa que un usuario daría a un curso recomendado.
        
        En un entorno real, esta recompensa vendría de la interacción del usuario.
        Aquí usamos una aproximación basada en la calificación promedio y diversidad.
        """
        # Usar calificación promedio como base
        base_reward = course_features.iloc[action][avg_rating_col] / 5.0
        
        # Penalizar si el curso es muy similar a los que ya ha visto
        diversity_bonus = 0.0
        if len(history) > 0:
            diversity_bonus = np.random.uniform(0, 0.2)
        
        # La recompensa final es una combinación de calificación y diversidad
        reward = base_reward + diversity_bonus
        
        # Asegurar que la recompensa esté en [0, 1]
        reward = max(0, min(1, reward))
        
        return reward
    
    def load_weights(self, actor_path, critic_path):
        """Carga pesos de los modelos desde archivos."""
        self.actor.load_weights(actor_path)
        self.critic.load_weights(critic_path)
    
    def save_weights(self, actor_path, critic_path):
        """Guarda pesos de los modelos a archivos."""
        def format_path(path):
            if not path.endswith('.weights.h5'):
                if path.endswith('.h5'):
                    return path[:-3] + '.weights.h5'
                else:
                    return path + '.weights.h5'
            return path
            
        actor_path = format_path(actor_path)
        critic_path = format_path(critic_path)
        
        self.actor.save_weights(actor_path)
        self.critic.save_weights(critic_path)
