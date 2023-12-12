import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, concatenate
from tensorflow.python.client import device_lib

c51= "/content/drive/My Drive/resultadosdata13.csv"

# Leer el DataFrame
df = pd.read_csv(c51)

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
tensores_coords = df_secuencias[['Latitud', 'Longitud']].to_numpy()

# Asegurar que los tensores de accidentes tengan la forma correcta
tensores_accidentes = tensores_accidentes.reshape(tensores_accidentes.shape[0], longitud_secuencia, 1)

# Verificar las formas de los tensores
print("Tensor de Secuencias de Accidentes:", tensores_accidentes.shape)
print("Tensor de Coordenadas:", tensores_coords.shape)

# Paso 1: División de los Datos
X_train, X_temp, y_train, y_temp = train_test_split(tensores_accidentes, tensores_coords, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Asegurar que las coordenadas estén divididas de la misma manera que las secuencias de accidentes
coords_train, coords_temp, _, _ = train_test_split(tensores_coords, tensores_coords, test_size=0.3, random_state=42)
coords_val, coords_test, _, _ = train_test_split(coords_temp, coords_temp, test_size=0.5, random_state=42)

print("Conjunto de Entrenamiento Coords:", coords_train.shape)
print("Conjunto de Validación Coords:", coords_val.shape)
print("Conjunto de Prueba Coords:", coords_test.shape)

# Normalizar los datos de entrada
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train.reshape(-1, 1)).reshape(X_train.shape)
X_val_scaled = scaler.transform(X_val.reshape(-1, 1)).reshape(X_val.shape)
X_test_scaled = scaler.transform(X_test.reshape(-1, 1)).reshape(X_test.shape)

# Verificar las GPUs disponibles
print(device_lib.list_local_devices())

# Configurar TensorFlow para usar GPU
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Definir tensores de entrada
input_sequence = Input(shape=(longitud_secuencia, 1), name='input_sequence')

# Capas LSTM para procesar secuencias temporales con regularización Dropout
lstm_output1 = LSTM(units=500, activation='tanh', dropout=0.2, recurrent_dropout=0.2, return_sequences=True)(input_sequence)
lstm_output2 = LSTM(units=500, activation='tanh', dropout=0.2, recurrent_dropout=0.2, return_sequences=True)(lstm_output1)
lstm_output3 = LSTM(units=500, activation='tanh', dropout=0.2, recurrent_dropout=0.2, return_sequences=True)(lstm_output2)
lstm_output4 = LSTM(units=500, activation='tanh', dropout=0.2, recurrent_dropout=0.2)(lstm_output3)

# Capa Dense después de la capa LSTM con regularización Dropout
fc_layer1 = Dense(200, activation='relu')(lstm_output4)
fc_layer1 = Dropout(0.2)(fc_layer1)

# Capa de entrada para las coordenadas
input_coords = Input(shape=(2,), name='input_coords')

# Concatenar salida de la capa Dropout y entrada de coordenadas
merged = concatenate([fc_layer1, input_coords])

# Capa densa después de la concatenación con regularización Dropout
fc_layer2 = Dense(200, activation='relu')(merged)
fc_layer2 = Dropout(0.2)(fc_layer2)

# Capa de salida
output_layer = Dense(1, activation='linear', name='output')(fc_layer2)

# Compilar el modelo
model = Model(inputs=[input_sequence, input_coords], outputs=output_layer)
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae', 'mse'])

# Resumen del modelo
model.summary()

# Entrenar el modelo
history = model.fit([X_train_scaled, coords_train], y_train, epochs=50, batch_size=32, validation_data=([X_val_scaled, coords_val], y_val))

# Evaluar en el conjunto de prueba
test_loss, test_mae, test_mse = model.evaluate([X_test_scaled, coords_test], y_test)
print(f"Pérdida en el conjunto de prueba: {test_loss}")
print(f"MAE en el conjunto de prueba: {test_mae}")
print(f"MSE en el conjunto de prueba: {test_mse}")