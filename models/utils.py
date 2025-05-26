import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_processed_data(base_path):
    """Función para cargar los datos preprocesados desde archivos CSV."""
    course_features = pd.read_csv(f"{base_path}/course_features.csv")
    train_interactions = pd.read_csv(f"{base_path}/train_interactions.csv")
    test_interactions = pd.read_csv(f"{base_path}/test_interactions.csv")
    
    return course_features, train_interactions, test_interactions

def create_user_course_matrices(train_interactions, n_users, n_courses):
    """Función para crear matrices de interacción usuario-curso."""
    rating_matrix = np.zeros((n_users, n_courses))
    for _, row in train_interactions.iterrows():
        rating_matrix[int(row['user_id']), int(row['course_id_encoded'])] = row['rating']
    
    interaction_matrix = (rating_matrix > 0).astype(float)
    
    return rating_matrix, interaction_matrix

def create_feature_matrix(course_features, feature_cols):
    """Función para normalizar las características numéricas de los cursos."""
    features = course_features[feature_cols].values
    scaler = MinMaxScaler()
    normalized_features = scaler.fit_transform(features)
    
    return normalized_features

def get_user_history(interactions_df, user_id, course_id_mapping, max_history=10):
    """Función para obtener la secuencia cronológica de cursos de un usuario."""
    user_courses = interactions_df[interactions_df['user_id'] == user_id]
    
    if 'timestamp' in user_courses.columns:
        user_courses = user_courses.sort_values('timestamp')
    
    course_history = [course_id_mapping.get(course_id, 0) 
                      for course_id in user_courses['course_id'].values]
    
    if len(course_history) > max_history:
        course_history = course_history[-max_history:]
    
    return course_history

def batch_history_generator(interactions_df, batch_size, n_users, n_courses, course_id_mapping, max_history=10):
    """Generador de lotes de datos para entrenamiento de modelos secuenciales."""
    user_ids = interactions_df['user_id'].unique()
    
    while True:
        batch_users = np.random.choice(user_ids, batch_size, replace=True)
        
        histories = []
        next_items = []
        rewards = []
        
        for user_id in batch_users:
            user_data = interactions_df[interactions_df['user_id'] == user_id]
            
            if len(user_data) <= 1:  # Se requieren al menos 2 interacciones
                continue
                
            if 'timestamp' in user_data.columns:
                user_data = user_data.sort_values('timestamp')
            
            split_idx = np.random.randint(1, len(user_data))
            
            # Obtener historial hasta el punto de división
            history_data = user_data.iloc[:split_idx]
            next_item_data = user_data.iloc[split_idx]
            
            # Crear secuencia de historial codificada
            history = [course_id_mapping.get(cid, 0) for cid in history_data['course_id'].values]
            if len(history) > max_history:
                history = history[-max_history:]
            
            # Próximo elemento y recompensa (calificación)
            next_item = course_id_mapping.get(next_item_data['course_id'], 0)
            reward = next_item_data['rating'] / 5.0  # Normalizar a [0,1]
            
            histories.append(history)
            next_items.append(next_item)
            rewards.append(reward)
        
        padded_histories = pad_sequences(histories, maxlen=max_history, padding='pre')
        
        yield padded_histories, np.array(next_items), np.array(rewards)
