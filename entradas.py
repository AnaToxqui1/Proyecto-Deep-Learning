import pandas as pd
import numpy as np
from datetime import timedelta

df = pd.read_csv("/home/ana/Escritorio/Deep_Learning /Proyecto/resultados.csv")

df['Día'] = pd.to_datetime(df['Día'])

# Define la longitud de la secuencia (por ejemplo, 7 días)
longitud_secuencia = 7

# Crear un nuevo DataFrame para almacenar las secuencias temporales
df_secuencias = pd.DataFrame(columns=['Latitud', 'Longitud', 'Secuencia_Accidentes'])

# Recorrer cada ubicación única en el conjunto de datos
for key, group in df.groupby(['Latitud', 'Longitud']):
    latitud, longitud = key

    # Ordenar por fecha
    group = group.sort_values(by='Día')

    # Crear secuencias temporales
    secuencias_accidentes = []
    for i in range(len(group) - longitud_secuencia + 1):
        secuencia = group.iloc[i:i + longitud_secuencia]['Accidentes'].values
        secuencias_accidentes.append(secuencia)

    # Crear el DataFrame de secuencias
    df_temporal = pd.DataFrame({
        'Latitud': [latitud] * len(secuencias_accidentes),
        'Longitud': [longitud] * len(secuencias_accidentes),
        'Secuencia_Accidentes': secuencias_accidentes
    })

    # Concatenar al DataFrame principal
    df_secuencias = pd.concat([df_secuencias, df_temporal], ignore_index=True)

# Convertir las secuencias a tensores numpy
tensores_accidentes = np.array(df_secuencias['Secuencia_Accidentes'].tolist())
tensores_accidentes = tensores_accidentes.reshape(tensores_accidentes.shape[0], longitud_secuencia, 1)
tensores_coords = df_secuencias[['Latitud', 'Longitud']].to_numpy()

# Verificar las formas de los tensores
print("Tensor de Secuencias de Accidentes:", tensores_accidentes.shape)
print("Tensor de Coordenadas:", tensores_coords.shape)