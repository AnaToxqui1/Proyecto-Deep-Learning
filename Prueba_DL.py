import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Leer datos
df = pd.read_csv("/home/ana/Escritorio/Deep_Learning /Proyecto/resultados.csv")

# Convertir la columna 'Día' a datetime
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

# Convertir los numpy arrays a tensores de PyTorch
X_accidents = torch.from_numpy(tensores_accidentes).float()
X_coords = torch.from_numpy(tensores_coords).float()
y = torch.from_numpy(tensores_coords).float()  # y_coords es igual a las coordenadas

# Verificar si hay GPU disponible
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("No se detectó una GPU. Se utilizará la CPU.")

# Mover tensores a la GPU
X_accidents = X_accidents.to(device)
X_coords = X_coords.to(device)
y = y.to(device)

print("Paso 1: División de los Datos")
X_train, X_temp, y_train, y_temp = train_test_split(X_accidents, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Asegurar que las coordenadas estén divididas de la misma manera que las secuencias de accidentes
coords_train, coords_temp, _, _ = train_test_split(tensores_coords, tensores_coords, test_size=0.3, random_state=42)
coords_val, coords_test, _, _ = train_test_split(coords_temp, coords_temp, test_size=0.5, random_state=42)

# Convertir las coordenadas a tensores de PyTorch y mover a la GPU
coords_train = torch.from_numpy(coords_train).float().to(device)
coords_val = torch.from_numpy(coords_val).float().to(device)
coords_test = torch.from_numpy(coords_test).float().to(device)

print("Conjunto de Entrenamiento Coords:", coords_train.shape)
print("Conjunto de Validación Coords:", coords_val.shape)
print("Conjunto de Prueba Coords:", coords_test.shape)

print(" Definir la arquitectura del modelo en PyTorch")
class MyModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MyModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc_coords = nn.Linear(hidden_size + 2, output_size)

    def forward(self, x_seq, x_coords):
        lstm_out, _ = self.lstm(x_seq)
        lstm_out = lstm_out[:, -1, :]  # Tomar la última salida de la secuencia
        merged = torch.cat([lstm_out, x_coords], dim=1)
        output = self.fc_coords(merged)
        return output

# Definir el modelo y mover a la GPU
model = MyModel(input_size=1, hidden_size=50, output_size=2).to(device)

# Definir función de pérdida y optimizador
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Funciones para calcular métricas
def calculate_metrics(predictions, targets):
    mae = mean_absolute_error(targets, predictions)
    mse = mean_squared_error(targets, predictions)
    rmse = np.sqrt(mse)
    return mae, mse, rmse

print("Paso 3: Entrenamiento del Modelo")
epochs = 1
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train, coords_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

print("Paso 4: Evaluación del Modelo en el conjunto de validación")
model.eval()
with torch.no_grad():
    val_outputs = model(X_val, coords_val)
    val_loss = criterion(val_outputs, y_val)
    val_predictions = val_outputs.cpu().numpy()
    val_targets = y_val.cpu().numpy()
    val_mae, val_mse, val_rmse = calculate_metrics(val_predictions, val_targets)
    print(f"Pérdida en el conjunto de validación: {val_loss.item():.4f}")
    print(f"MAE en el conjunto de validación: {val_mae:.4f}")
    print(f"MSE en el conjunto de validación: {val_mse:.4f}")
    print(f"RMSE en el conjunto de validación: {val_rmse:.4f}")

print(" Paso 5: Prueba del Modelo en el conjunto de prueba")
test_outputs = model(X_test, coords_test)
test_loss = criterion(test_outputs, y_test)
test_predictions = test_outputs.cpu().numpy()
test_targets = y_test.cpu