import os
import numpy as np
import pandas as pd
import tensorflow as tf
import json
import pickle
from models.dqn_model import DQNRecommender
from models.ddpg_model import DDPGRecommender
from models.a2c_model import A2CRecommender

# Configuración
MODEL_DIR = 'models/trained'
PROCESSED_DATA_PATH = 'Data/Processed'
MAX_HISTORY_LENGTH = 10

def load_data():
    """Carga datos procesados necesarios para el sistema de recomendación."""
    print("Cargando datos...")
    
    # Cargar características de cursos
    course_features = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, 'course_features.csv'))
    
    # Cargar mapeo de IDs
    with open(os.path.join(PROCESSED_DATA_PATH, 'mapping.json'), 'r') as f:
        mapping_data = json.load(f)
    
    # Extraer información relevante
    course_id_mapping = mapping_data.get('course_id_mapping', {})
    reverse_mapping = {int(v): k for k, v in course_id_mapping.items()}
    n_courses = mapping_data.get('n_courses', len(course_features))
    
    return course_features, reverse_mapping, n_courses

def load_models(n_courses):
    """Carga los modelos entrenados."""
    print("\nCargando modelos...")
    models = {}
    
    # Cargar DQN
    try:
        dqn_model = DQNRecommender(
            n_courses=n_courses,
            embedding_dim=64,            
            hidden_dims=[128, 64],        
            state_history_length=MAX_HISTORY_LENGTH,
            gamma=0.95                    
        )
        dqn_model.load_weights(os.path.join(MODEL_DIR, 'dqn_model.weights.h5'))
        models['dqn'] = dqn_model
        print("Modelo DQN cargado correctamente")
    except Exception as e:
        print(f"Error al cargar modelo DQN: {e}")
    
    # Cargar DDPG
    try:
        ddpg_model = DDPGRecommender(
            n_courses=n_courses,
            embedding_dim=64,             
            hidden_dims=[128, 64],        
            state_history_length=MAX_HISTORY_LENGTH,
            gamma=0.995,                  
            tau=0.01                      
        )
        ddpg_model.load_weights(
            os.path.join(MODEL_DIR, 'ddpg_actor.weights.h5'),
            os.path.join(MODEL_DIR, 'ddpg_critic.weights.h5')
        )
        models['ddpg'] = ddpg_model
        print("Modelo DDPG cargado correctamente")
    except Exception as e:
        print(f"Error al cargar modelo DDPG: {e}")
    
    # Cargar A2C
    try:
        a2c_model = A2CRecommender(
            n_courses=n_courses,
            embedding_dim=64,               
            hidden_dims=[128, 64],        
            state_history_length=MAX_HISTORY_LENGTH,
            gamma=0.99                    
        )
        a2c_model.load_weights(
            os.path.join(MODEL_DIR, 'a2c_actor.weights.h5'),
            os.path.join(MODEL_DIR, 'a2c_critic.weights.h5')
        )
        models['a2c'] = a2c_model
        print("Modelo A2C cargado correctamente")
    except Exception as e:
        print(f"Error al cargar modelo A2C: {e}")
    
    return models

def get_course_details(course_id, course_features, reverse_mapping):
    """Obtiene detalles de un curso por su ID."""
    original_id = reverse_mapping.get(course_id)
    
    if original_id is None:
        return None
    
    course_data = course_features[course_features['course_id'] == original_id]
    
    if course_data.empty:
        return None
    
    return course_data.iloc[0]

def preprocess_course_history(course_ids, max_length=MAX_HISTORY_LENGTH):
    """Preprocesa el historial de cursos para los modelos."""
    # Convertir IDs a enteros
    history = [int(course_id) + 1 for course_id in course_ids]  
    
    # Limitar longitud
    if len(history) > max_length:
        history = history[-max_length:]
    
    # Añadir padding
    if len(history) < max_length:
        padded = [0] * (max_length - len(history)) + history
    else:
        padded = history
    
    return padded

def get_recommendations(model, state, n_recommendations=5, exclude_courses=None):
    """Obtiene recomendaciones de un modelo específico."""
    if exclude_courses is None:
        exclude_courses = []
    
    if model.__class__.__name__ == 'DQNRecommender':
        # Para DQN: Predecir Q-valores y ordenar
        q_values = model.q_network.predict(np.array([state]), verbose=0)[0]
        
        # Excluir cursos ya vistos
        for course_id in state:
            if course_id > 0: 
                q_values[course_id-1] = float('-inf')
        
        # Excluir cursos adicionales
        for course_id in exclude_courses:
            q_values[course_id] = float('-inf')
        
        # Obtener los mejores cursos
        recommendations = np.argsort(q_values)[::-1][:n_recommendations]
        
    elif model.__class__.__name__ == 'DDPGRecommender':
        # Para DDPG: Usar recommend para obtener el mejor curso
        action_probs = model.act(state, training=False)
        
        # Excluir cursos ya vistos
        for course_id in state:
            if course_id > 0:
                action_probs[course_id-1] = 0
        
        # Excluir cursos adicionales
        for course_id in exclude_courses:
            action_probs[course_id] = 0
        
        # Obtener los mejores cursos
        recommendations = np.argsort(action_probs)[::-1][:n_recommendations]
        
    elif model.__class__.__name__ == 'A2CRecommender':
        # Para A2C: Predecir logits y ordenar
        logits = model.actor.predict(np.array([state]), verbose=0)[0]
        
        # Excluir cursos ya vistos
        for course_id in state:
            if course_id > 0:
                logits[course_id-1] = float('-inf')
        
        # Excluir cursos adicionales
        for course_id in exclude_courses:
            logits[course_id] = float('-inf')
        
        # Obtener los mejores cursos
        recommendations = np.argsort(logits)[::-1][:n_recommendations]
    
    else:
        raise ValueError(f"Tipo de modelo no soportado: {model.__class__.__name__}")
    
    return recommendations

def display_recommendations(recommendations, course_features, reverse_mapping):
    """Muestra las recomendaciones de forma amigable."""
    for i, course_id in enumerate(recommendations):
        course_details = get_course_details(course_id, course_features, reverse_mapping)
        
        if course_details is not None:
            print(f"{i+1}. {course_details['name']} - {course_details['institution']}")
            print(f"   URL: {course_details['course_url']}")
            if 'avg_rating' in course_details:
                print(f"   Calificación promedio: {course_details['avg_rating']:.2f}/5.0")
            print()

def interactive_recommendation_session():
    """Sesión interactiva de recomendación."""
    # Cargar datos y modelos
    course_features, reverse_mapping, n_courses = load_data()
    models = load_models(n_courses)
    
    if not models:
        print("No se pudo cargar ningún modelo. Terminando sesión.")
        return
    
    # Elegir modelo para usar
    available_models = list(models.keys())
    print("\nModelos disponibles:")
    for i, model_name in enumerate(available_models):
        print(f"{i+1}. {model_name.upper()}")
    
    model_idx = int(input("\nSelecciona un modelo (número): ")) - 1
    if model_idx < 0 or model_idx >= len(available_models):
        print("Selección inválida. Usando DQN por defecto.")
        model_name = 'dqn' if 'dqn' in models else available_models[0]
    else:
        model_name = available_models[model_idx]
    
    model = models[model_name]
    print(f"\nUsando modelo: {model_name.upper()}")
    
    # Iniciar sesión
    course_history = []
    excluded_courses = []
    
    print("\n===== SISTEMA DE RECOMENDACIÓN DE CURSOS =====")
    print("Para comenzar, indica algunos cursos que ya hayas tomado o te interesen.")
    print("Puedes buscar en el catálogo o ingresar un ID de curso directamente.")
    
    while True:
        print("\nOpciones:")
        print("1. Buscar cursos en el catálogo")
        print("2. Agregar curso por ID")
        print("3. Ver historial actual")
        print("4. Obtener recomendaciones")
        print("5. Actualizar historial con curso recomendado")
        print("6. Salir")
        
        choice = input("\nSelecciona una opción: ")
        
        if choice == '1':
            # Buscar cursos
            search_term = input("Ingresa término de búsqueda: ")
            results = course_features[course_features['name'].str.contains(search_term, case=False)]
            
            if results.empty:
                print("No se encontraron resultados.")
            else:
                print("\nResultados de la búsqueda:")
                for i, (_, course) in enumerate(results.iterrows()):
                    print(f"{i+1}. {course['name']} - {course['institution']}")
                    print(f"   ID: {course_id_mapping.get(course['course_id'], 'N/A')}")
                    print()
        
        elif choice == '2':
            # Agregar curso por ID
            try:
                course_id = int(input("Ingresa el ID del curso: "))
                course_details = get_course_details(course_id, course_features, reverse_mapping)
                
                if course_details is not None:
                    print(f"\nAgregado: {course_details['name']} - {course_details['institution']}")
                    course_history.append(course_id)
                else:
                    print("Curso no encontrado.")
            except ValueError:
                print("ID inválido. Debe ser un número.")
        
        elif choice == '3':
            # Ver historial
            print("\nHistorial actual:")
            if not course_history:
                print("No hay cursos en el historial.")
            else:
                for i, course_id in enumerate(course_history):
                    course_details = get_course_details(course_id, course_features, reverse_mapping)
                    if course_details is not None:
                        print(f"{i+1}. {course_details['name']} - {course_details['institution']}")
        
        elif choice == '4':
            # Obtener recomendaciones
            if not course_history:
                print("Agrega al menos un curso al historial primero.")
                continue
            
            state = preprocess_course_history(course_history)
            recommendations = get_recommendations(
                model, 
                state, 
                n_recommendations=5, 
                exclude_courses=excluded_courses
            )
            
            print("\nCursos recomendados:")
            display_recommendations(recommendations, course_features, reverse_mapping)
            
            # Guardar recomendaciones para posible selección
            current_recommendations = recommendations
        
        elif choice == '5':
            # Actualizar historial con curso recomendado
            try:
                if 'current_recommendations' not in locals():
                    print("Primero debes obtener recomendaciones.")
                    continue
                
                rec_idx = int(input("Selecciona el número de la recomendación para agregar al historial: ")) - 1
                
                if rec_idx < 0 or rec_idx >= len(current_recommendations):
                    print("Selección inválida.")
                else:
                    course_id = current_recommendations[rec_idx]
                    course_details = get_course_details(course_id, course_features, reverse_mapping)
                    
                    if course_details is not None:
                        print(f"\nAgregado: {course_details['name']} - {course_details['institution']}")
                        course_history.append(course_id)
                    else:
                        print("Error: Curso no encontrado.")
            except ValueError:
                print("Entrada inválida. Debe ser un número.")
        
        elif choice == '6':
            # Salir
            print("Gracias por usar el sistema de recomendación. ¡Hasta pronto!")
            break
        
        else:
            print("Opción inválida. Intenta de nuevo.")

def main():
    """Función principal."""
    print("=== Sistema de Recomendación de Cursos con DRL ===\n")
    interactive_recommendation_session()

if __name__ == "__main__":
    main()
