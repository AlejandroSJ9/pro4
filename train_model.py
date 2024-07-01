import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pickle

# Cargar los datos
df = pd.read_csv('dataset.csv')  # Asegúrate de que este es el archivo correcto

# Eliminar espacios en blanco en los nombres de las columnas
df.columns = df.columns.str.strip()

# Mostrar los nombres de las columnas para verificar
print(df.columns)

# Seleccionar las características y la variable objetivo
features = ['season', 'mnth', 'holiday', 'workingday', 'weathersit', 'temp', 'hum', 'windspeed']
target = 'cnt'

# Verificar que las columnas están en el DataFrame
for feature in features:
    if feature not in df.columns:
        print(f"Error: {feature} no está en las columnas del DataFrame")

X = df[features]
y = df[target]

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar el modelo de regresión lineal
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluar el modelo
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error (MSE): {mse}')
print(f'R^2 Score: {r2}')

# Guardar el modelo entrenado
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
