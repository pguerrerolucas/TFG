import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Definir rutas
data_path = os.path.join(os.getcwd(), 'Data', 'Kaggle', 'Cousera data')
courses_file = os.path.join(data_path, 'Coursera_courses.csv')
reviews_file = os.path.join(data_path, 'Coursera_reviews.csv')
output_path = os.path.join(os.getcwd(), 'Data', 'Processed')

# Crear directorio de salida si no existe
if not os.path.exists(output_path):
    os.makedirs(output_path)
    print(f"Directorio creado: {output_path}")

print("Cargando datos de cursos...")
courses_df = pd.read_csv(courses_file)

print("Información de los cursos:")
print(f"- Número de cursos: {courses_df.shape[0]}")
print(f"- Columnas: {', '.join(courses_df.columns)}")
print(f"- Instituciones únicas: {courses_df['institution'].nunique()}")
print("\nPrimeras 5 filas de cursos:")
print(courses_df.head())

# Cargar datos de reseñas (leer en chunks debido al gran tamaño)
print("\nCargando datos de reseñas (puede tardar unos minutos)...")
chunk_size = 100000
chunks = []

for chunk in pd.read_csv(reviews_file, chunksize=chunk_size):
    chunks.append(chunk)
    print(f"Cargado chunk de {chunk.shape[0]} filas")

reviews_df = pd.concat(chunks)

print("\nInformación de las reseñas:")
print(f"- Número de reseñas: {reviews_df.shape[0]}")
print(f"- Columnas: {', '.join(reviews_df.columns)}")

# Limpieza de reseñas
print("\nLimpiando datos de reseñas...")

# Verificar valores nulos
print("\nValores nulos en cursos:")
print(courses_df.isnull().sum())

print("\nValores nulos en reseñas:")
print(reviews_df.isnull().sum())

# Convertir fechas de reseñas a datetime
def parse_date(date_str):
    try:
        return pd.to_datetime(date_str, errors='coerce')
    except:
        return None

print("\nConvirtiendo fechas...")
reviews_df['date_reviews'] = reviews_df['date_reviews'].apply(parse_date)

# Extraer características temporales
reviews_df['review_year'] = reviews_df['date_reviews'].dt.year
reviews_df['review_month'] = reviews_df['date_reviews'].dt.month

# Procesar texto de las reseñas
print("\nProcesando texto de reseñas...")
# Eliminar comillas innecesarias al inicio y final
reviews_df['reviews'] = reviews_df['reviews'].str.replace('^"|"$', '', regex=True)
# Limpiar campo de reviewers
reviews_df['reviewers'] = reviews_df['reviewers'].str.replace('By ', '')

# Crear características para el sistema de recomendación
print("\nCreando características para el sistema de recomendación...")

# 1. Popularidad del curso (número de reseñas)
course_popularity = reviews_df.groupby('course_id').size().reset_index(name='review_count')

# 2. Calificación promedio del curso
course_avg_rating = reviews_df.groupby('course_id')['rating'].mean().reset_index(name='avg_rating')

# 3. Sentimiento de las reseñas (simple aproximación basada en la calificación)
reviews_df['sentiment'] = reviews_df['rating'].apply(lambda x: 'positive' if x >= 4 else ('neutral' if x >= 3 else 'negative'))

# Combinar características con información de cursos
print("\nCombinando datos...")
course_features = pd.merge(courses_df, course_popularity, on='course_id', how='left')
course_features = pd.merge(course_features, course_avg_rating, on='course_id', how='left')

# Rellenar valores nulos (cursos sin reseñas)
course_features['review_count'] = course_features['review_count'].fillna(0)
course_features['avg_rating'] = course_features['avg_rating'].fillna(0)

# Codificar variables categóricas
print("\nCodificando variables categóricas...")
institution_encoder = LabelEncoder()
course_features['institution_encoded'] = institution_encoder.fit_transform(course_features['institution'])

# Crear matrices para el sistema de recomendación
print("\nCreando matrices usuario-item...")

# Simulamos IDs de usuario basados en reviewers
user_encoder = LabelEncoder()
reviews_df['user_id'] = user_encoder.fit_transform(reviews_df['reviewers'])

# Crear matriz de interacciones usuario-curso
interactions = reviews_df[['user_id', 'course_id', 'rating']]

# Estadísticas de interacciones
print(f"\nNúmero de usuarios únicos: {reviews_df['user_id'].nunique()}")
print(f"Número de interacciones usuario-curso: {interactions.shape[0]}")

# Dividir datos para entrenamiento y validación
print("\nDividiendo datos para entrenamiento y validación...")
train_interactions, test_interactions = train_test_split(
    interactions, test_size=0.2, random_state=42
)

# Guardar datos preprocesados
print("\nGuardando datos preprocesados...")
course_features.to_csv(os.path.join(output_path, 'course_features.csv'), index=False)
train_interactions.to_csv(os.path.join(output_path, 'train_interactions.csv'), index=False)
test_interactions.to_csv(os.path.join(output_path, 'test_interactions.csv'), index=False)

# Guardar mapeo de IDs para referencia
user_mapping = pd.DataFrame({
    'user_id': range(len(user_encoder.classes_)),
    'reviewer': user_encoder.classes_
})
user_mapping.to_csv(os.path.join(output_path, 'user_mapping.csv'), index=False)

# Análisis exploratorio básico
print("\nGenerando visualizaciones...")

# 1. Distribución de calificaciones
plt.figure(figsize=(10, 6))
sns.countplot(x='rating', data=reviews_df)
plt.title('Distribución de calificaciones')
plt.xlabel('Calificación')
plt.ylabel('Número de reseñas')
plt.savefig(os.path.join(output_path, 'rating_distribution.png'))

# 2. Top 20 instituciones por número de cursos
top_institutions = courses_df['institution'].value_counts().head(20)
plt.figure(figsize=(12, 8))
top_institutions.plot(kind='bar')
plt.title('Top 20 instituciones por número de cursos')
plt.xlabel('Institución')
plt.ylabel('Número de cursos')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(output_path, 'top_institutions.png'))

# 3. Popularidad vs. Calificación promedio
plt.figure(figsize=(10, 6))
plt.scatter(course_features['review_count'], course_features['avg_rating'], alpha=0.5)
plt.title('Popularidad vs. Calificación promedio de cursos')
plt.xlabel('Número de reseñas')
plt.ylabel('Calificación promedio')
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig(os.path.join(output_path, 'popularity_vs_rating.png'))

print("\nPreprocesamiento completado. Los datos están listos para el sistema de recomendación.")
print(f"Archivos guardados en: {output_path}")
