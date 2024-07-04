# Predicción de Alquiler de Bicicletas

Este proyecto utiliza un modelo de regresión lineal para predecir el número de alquileres de bicicletas en función de diversas condiciones como la temperatura, la humedad y las condiciones meteorológicas.

## Requisitos

- Python 3.7 o superior
- pip

## Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/AlejandroSJ9/pro4
cd pro4
```

### 2. Crear un entorno virtual
```bash
python -m venv venv
```

### 3. Activar el entorno virtual
```bash
python -m venv venv
```

### 4. Instalar las dependencias
```bash
pip install -r requirements.txt
```
## Ejecución del Proyecto
### 1. Entrenar el Modelo
```bash
python train_model.py
```
Esto generará el archivo model.pkl que contiene el modelo entrenado.

### 2. Ejecutar la Aplicación Web
Para iniciar la aplicación web, ejecuta el siguiente comando
```bash
python app.py
```
### 3. Acceder a la Aplicación
Abre un navegador web y accede a http://127.0.0.1:5000 para interactuar con la aplicación.

## Estructura del Proyecto
app.py: Archivo principal que contiene el código de la aplicación Flask.

train_model.py: Script para entrenar el modelo de regresión lineal.

dataset.csv: Conjunto de datos utilizado para entrenar el modelo.

model.pkl: Archivo que contiene el modelo de regresión lineal entrenado.

templates/: Directorio que contiene los archivos HTML para las plantillas de la aplicación.

static/: Directorio que contiene archivos estáticos como CSS y JavaScript.

