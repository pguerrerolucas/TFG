import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Flatten, Concatenate, Dropout
from tensorflow.keras.optimizers import Adam
from collections import deque
import random

class DQNRecommender:
    """
    Sistema de recomendación basado en Deep Q-Network (DQN).
    
    Modelo que aprende a predecir el valor Q de recomendar cada curso posible
    dado el historial de interacciones previas del usuario.
    """
    
    def __init__(
        self, 
        n_courses,
        embedding_dim=32,
        hidden_dims=[64, 32],
        state_history_length=10,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.1,
        epsilon_decay=0.995,
        learning_rate=0.001,
        memory_size=10000,
        batch_size=32
    ):
        self.n_courses = n_courses
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims
        self.state_history_length = state_history_length
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.memory_size = memory_size
        self.batch_size = batch_size
        
        # Memoria de experiencias (experience replay)
        self.memory = deque(maxlen=memory_size)
        
        # Construir modelos
        self.q_network = self._build_model()
        self.target_network = self._build_model()
        self._update_target_network()
    
    def _build_model(self):
        """Construye la arquitectura de la red Q."""
        # Input: Historial de cursos del usuario
        input_history = Input(shape=(self.state_history_length,), name='history_input')
        
        # Capa de embedding para representar los cursos
        embedding = Embedding(
            input_dim=self.n_courses + 1,  # +1 para el padding
            output_dim=self.embedding_dim,
            mask_zero=True,
            name='course_embedding'
        )(input_history)
        
        # Procesar la secuencia con LSTM
        lstm_out = LSTM(units=self.hidden_dims[0], name='lstm_layer')(embedding)
        
        # Capas densas
        x = Dense(self.hidden_dims[0], activation='relu', name='dense_1')(lstm_out)
        x = Dropout(0.3)(x)
        
        for i, dim in enumerate(self.hidden_dims[1:], 2):
            x = Dense(dim, activation='relu', name=f'dense_{i}')(x)
            x = Dropout(0.3)(x)
        
        # Capa de salida: Q-valores para cada curso posible
        output = Dense(self.n_courses, activation='linear', name='q_values')(x)
        
        model = Model(inputs=input_history, outputs=output)
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        
        return model
    
    def _update_target_network(self):
        """Actualiza los pesos de la red objetivo con los de la red principal."""
        self.target_network.set_weights(self.q_network.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        """Almacena una experiencia en la memoria."""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, training=True):
        """
        Selecciona una acción (curso a recomendar) basado en el estado actual.
        
        Args:
            state: Historial de cursos del usuario (numpy array)
            training: Si es True, se aplica política epsilon-greedy
        
        Returns:
            Índice del curso recomendado
        """
        # Durante entrenamiento, aplicar política epsilon-greedy para exploración
        if training and np.random.rand() <= self.epsilon:
            return np.random.randint(self.n_courses)
        
        # Predecir Q-valores para todos los cursos posibles
        q_values = self.q_network.predict(np.array([state]), verbose=0)[0]
        
        # Excluir cursos ya vistos (en el historial)
        for course_id in state:
            if course_id > 0:  # Ignorar padding (0)
                q_values[course_id-1] = float('-inf')
        
        # Seleccionar curso con mayor Q-valor
        return np.argmax(q_values)
    
    def replay(self, batch_size=None):
        """Entrena la red con experiencias aleatorias de la memoria."""
        if batch_size is None:
            batch_size = self.batch_size
            
        if len(self.memory) < batch_size:
            return
        
        # Muestrear experiencias aleatorias
        minibatch = random.sample(self.memory, batch_size)
        
        states = []
        targets = []
        
        for state, action, reward, next_state, done in minibatch:
            target = self.q_network.predict(np.array([state]), verbose=0)[0]
            
            if done:
                target[action] = reward
            else:
                # Predicción con la red objetivo para estabilidad
                t = self.target_network.predict(np.array([next_state]), verbose=0)[0]
                target[action] = reward + self.gamma * np.amax(t)
            
            states.append(state)
            targets.append(target)
        
        # Entrenamiento batch
        history = self.q_network.fit(
            np.array(states), 
            np.array(targets), 
            epochs=1, 
            verbose=0
        )
        
        # Decaimiento de epsilon para menos exploración con el tiempo
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        return history.history['loss'][0]
    
    def load_weights(self, filepath):
        """Carga pesos del modelo desde un archivo."""
        self.q_network.load_weights(filepath)
        self._update_target_network()
    
    def save_weights(self, filepath):
        """Guarda pesos del modelo a un archivo."""
        # Asegurar que el filepath termine en .weights.h5 para compatibilidad con Keras
        if not filepath.endswith('.weights.h5'):
            if filepath.endswith('.h5'):
                filepath = filepath[:-3] + '.weights.h5'
            else:
                filepath = filepath + '.weights.h5'
        self.q_network.save_weights(filepath)
    
    def get_model_summary(self):
        """Retorna un resumen de la arquitectura del modelo."""
        return self.q_network.summary()
    
    def train_episode(self, user_histories, course_features, n_interactions=100):
        """
        Entrena el modelo para un episodio completo.
        
        Args:
            user_histories: Diccionario {user_id: [lista de course_ids]}
            course_features: DataFrame con características de los cursos
            n_interactions: Número de interacciones a simular
            
        Returns:
            loss_history: Lista de pérdidas durante el entrenamiento
            reward_history: Lista de recompensas obtenidas
        """
        loss_history = []
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
            
            # Seleccionar acción (recomendar curso)
            action = self.act(state)
            
            # Obtener recompensa (simular interacción)
            reward = self._simulate_reward(action, history, course_features)
            
            # Actualizar historial con el nuevo curso
            next_history = history + [action + 1] 
            
            if len(next_history) < self.state_history_length:
                next_state = [0] * (self.state_history_length - len(next_history)) + next_history
            else:
                next_state = next_history[-self.state_history_length:]
            
            done = False
            
            # Guardar experiencia
            self.remember(state, action, reward, next_state, done)
            
            # Entrenar con batch de experiencias
            if len(self.memory) >= self.batch_size:
                loss = self.replay()
                if loss is not None:
                    loss_history.append(loss)
            
            reward_history.append(reward)
            
            # Actualizar red objetivo periódicamente
            if len(self.memory) % 100 == 0:
                self._update_target_network()
                
            # Actualizar el historial del usuario para la próxima interacción
            user_histories[user_id] = next_history
        
        return loss_history, reward_history
    
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
