import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Concatenate, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import random
from collections import deque

class OUActionNoise:
    """
    Implementación del proceso de Ornstein-Uhlenbeck para exploración en espacios continuos.
    Proporciona ruido temporalmente correlacionado para mejorar la eficacia de exploración.
    """
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)

class DDPGRecommender:
    """
    Sistema de recomendación basado en Deep Deterministic Policy Gradient (DDPG).
    
    Arquitectura de aprendizaje por refuerzo que combina un modelo actor para seleccionar 
    acciones y un modelo crítico para evaluar las acciones tomadas.
    """
    
    def __init__(
        self, 
        n_courses,
        embedding_dim=32,
        hidden_dims=[64, 32],
        state_history_length=10,
        gamma=0.99,
        tau=0.005,
        actor_learning_rate=0.001,
        critic_learning_rate=0.002,
        memory_size=10000,
        batch_size=64
    ):
        self.n_courses = n_courses
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims
        self.state_history_length = state_history_length
        self.gamma = gamma  # Factor de descuento
        self.tau = tau  # Tasa de actualización suave
        self.actor_lr = actor_learning_rate
        self.critic_lr = critic_learning_rate
        self.memory_size = memory_size
        self.batch_size = batch_size
        
        # Memoria de experiencias
        self.memory = deque(maxlen=memory_size)
        
        # Crear redes actor y crítico
        self.actor, self.actor_target = self._build_actor()
        self.critic, self.critic_target = self._build_critic()
        
        # Inicializar redes objetivo con los mismos pesos
        self.actor_target.set_weights(self.actor.get_weights())
        self.critic_target.set_weights(self.critic.get_weights())
        
        # Ruido para exploración
        self.noise = OUActionNoise(
            mean=np.zeros(n_courses), 
            std_deviation=0.15 * np.ones(n_courses),
            theta=0.2 
        )
        
    def _build_actor(self):
        """Construye la red del actor (política)."""
        # Entrada: historial de cursos
        input_history = Input(shape=(self.state_history_length,), name='actor_history_input')
        
        # Embedding para representar los cursos
        embedding = Embedding(
            input_dim=self.n_courses + 1,  # +1 para padding
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
        
        # Capa de salida: distribución de probabilidad sobre cursos
        # Usamos softmax para obtener una distribución de probabilidad
        output = Dense(self.n_courses, activation='softmax', name='actor_output')(x)
        
        actor = Model(inputs=input_history, outputs=output)
        actor.compile(optimizer=Adam(learning_rate=self.actor_lr))
        
        # Crear una copia para la red objetivo
        actor_target = tf.keras.models.clone_model(actor)
        actor_target.set_weights(actor.get_weights())
        
        return actor, actor_target
    
    def _build_critic(self):
        """Construye la red del crítico (función de valor Q)."""
        # Entrada para el estado (historial de cursos)
        state_input = Input(shape=(self.state_history_length,), name='critic_state_input')
        
        # Embedding para el estado
        state_embedding = Embedding(
            input_dim=self.n_courses + 1,
            output_dim=self.embedding_dim,
            mask_zero=True,
            name='critic_state_embedding'
        )(state_input)
        
        # LSTM para el estado
        state_lstm = LSTM(units=self.hidden_dims[0], name='critic_state_lstm')(state_embedding)
        
        # Entrada para la acción (distribución de probabilidad sobre cursos)
        action_input = Input(shape=(self.n_courses,), name='critic_action_input')
        
        # Combinar estado y acción
        concat = Concatenate()([state_lstm, action_input])
        
        # Capas densas
        x = Dense(self.hidden_dims[0], activation='relu', name='critic_dense_1')(concat)
        x = Dropout(0.3)(x)
        
        for i, dim in enumerate(self.hidden_dims[1:], 2):
            x = Dense(dim, activation='relu', name=f'critic_dense_{i}')(x)
            x = Dropout(0.3)(x)
        
        # Salida: valor Q estimado
        q_value = Dense(1, activation='linear', name='critic_output')(x)
        
        critic = Model(inputs=[state_input, action_input], outputs=q_value)
        critic.compile(optimizer=Adam(learning_rate=self.critic_lr), loss='mse')
        
        # Crear una copia para la red objetivo
        critic_target = tf.keras.models.clone_model(critic)
        critic_target.set_weights(critic.get_weights())
        
        return critic, critic_target
    
    def remember(self, state, action, reward, next_state, done):
        """Almacena una experiencia en la memoria."""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, training=True):
        """
        Selecciona una acción (distribución de probabilidad sobre cursos) basado en el estado.
        
        Args:
            state: Historial de cursos del usuario (numpy array)
            training: Si es True, se añade ruido para exploración
            
        Returns:
            Vector de distribución de probabilidad sobre cursos
        """
        # Predecir distribución de probabilidad con el actor
        action = self.actor.predict(np.array([state]), verbose=0)[0]
        
        if training:
            # Añadir ruido para exploración
            noise = self.noise()
            action = action + noise
            
            # Asegurar que sigue siendo una distribución válida (no negativa y suma 1)
            action = np.maximum(action, 0)
            action_sum = np.sum(action)
            if action_sum > 0:
                action = action / action_sum
        
        # Excluir cursos ya vistos (colocar probabilidad 0)
        for course_id in state:
            if course_id > 0:  # Ignorar padding (0)
                action[course_id-1] = 0
        
        # Renormalizar si es necesario
        action_sum = np.sum(action)
        if action_sum > 0:
            action = action / action_sum
        
        return action
    
    def recommend(self, state):
        """
        Recomienda un curso para el usuario.
        
        Args:
            state: Historial de cursos del usuario
            
        Returns:
            Índice del curso recomendado
        """
        # Obtener distribución de probabilidad sin ruido
        probabilities = self.act(state, training=False)
        
        # Seleccionar el curso con mayor probabilidad
        return np.argmax(probabilities)
    
    def _update_target_models(self):
        """Actualización suave de los modelos objetivo."""
        # Actualizar pesos del actor objetivo
        actor_weights = self.actor.get_weights()
        actor_target_weights = self.actor_target.get_weights()
        for i in range(len(actor_weights)):
            actor_target_weights[i] = self.tau * actor_weights[i] + (1 - self.tau) * actor_target_weights[i]
        self.actor_target.set_weights(actor_target_weights)
        
        # Actualizar pesos del crítico objetivo
        critic_weights = self.critic.get_weights()
        critic_target_weights = self.critic_target.get_weights()
        for i in range(len(critic_weights)):
            critic_target_weights[i] = self.tau * critic_weights[i] + (1 - self.tau) * critic_target_weights[i]
        self.critic_target.set_weights(critic_target_weights)
    
    def replay(self):
        """Entrena las redes con experiencias de la memoria."""
        if len(self.memory) < self.batch_size:
            return
        
        # Muestrear experiencias aleatorias de la memoria
        minibatch = random.sample(self.memory, self.batch_size)
        
        states, actions, rewards, next_states, dones = zip(*minibatch)
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)
        
        # Calcular valores Q objetivo
        next_actions = self.actor_target.predict(next_states, verbose=0)
        q_next = self.critic_target.predict([next_states, next_actions], verbose=0).flatten()
        q_target = rewards + self.gamma * q_next * (1 - dones)
        
        # Entrenar el crítico
        self.critic.fit(
            [states, actions], 
            q_target, 
            epochs=1, 
            verbose=0,
            batch_size=self.batch_size
        )
        
        # Entrenar el actor usando el gradiente del crítico
        with tf.GradientTape() as tape:
            # Convertir estados a tensor si no lo es ya
            states_tensor = tf.convert_to_tensor(states, dtype=tf.float32)
            # Obtener acciones como tensor
            actions_pred = self.actor(states_tensor)
            # Pasar ambos como tensores al crítico
            critic_value = self.critic([states_tensor, actions_pred])
            # Queremos maximizar el valor Q, por lo que minimizamos su negativo
            actor_loss = -tf.math.reduce_mean(critic_value)
        
        actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(
            zip(actor_grad, self.actor.trainable_variables)
        )
        
        # Actualizar redes objetivo
        self._update_target_models()
        
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
            
            # Seleccionar acción (distribución de probabilidad)
            action = self.act(state)
            
            # Seleccionar curso basado en la distribución
            course_idx = np.random.choice(self.n_courses, p=action)
            
            # Obtener recompensa simulada
            reward = self._simulate_reward(course_idx, history, course_features)
            
            # Actualizar historial con el nuevo curso
            next_history = history + [course_idx + 1]  # +1 porque los índices en el embedding empiezan en 1
            
            if len(next_history) < self.state_history_length:
                next_state = [0] * (self.state_history_length - len(next_history)) + next_history
            else:
                next_state = next_history[-self.state_history_length:]
            
            # Estado terminal (para este ejemplo, siempre False)
            done = False
            
            # Guardar experiencia
            self.remember(state, action, reward, next_state, done)
            
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
        self.actor_target.set_weights(self.actor.get_weights())
        self.critic_target.set_weights(self.critic.get_weights())
    
    def save_weights(self, actor_path, critic_path):
        """Guarda pesos de los modelos a archivos."""
        # Asegurar que los paths terminen en .weights.h5 para compatibilidad con Keras
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
