from flask import Flask, render_template, request
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
import seaborn as sns
import io
import base64

app = Flask(__name__)

# Cargar el modelo entrenado
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Función para cargar el dataset
def load_dataset():
    return pd.read_csv('dataset.csv').rename(columns=str.strip)

# Función para realizar la predicción
def perform_prediction(season, month, holiday, workingday, weathersit, temp, hum, windspeed):
    # Preparar los datos para hacer la predicción
    data = pd.DataFrame([[season, month, holiday, workingday, weathersit, temp, hum, windspeed]],
                        columns=['season', 'mnth', 'holiday', 'workingday', 'weathersit', 'temp', 'hum', 'windspeed'])

    # Realizar la predicción
    prediction = round(model.predict(data)[0])

    # Explicación del resultado
    explanation = f"Según nuestro modelo, se estima que aproximadamente {int(prediction)} bicicletas serán alquiladas en las condiciones especificadas. "

    return int(prediction), explanation

# Ruta principal
@app.route('/')
def index():
    return render_template('index.html')

# Ruta para la predicción
@app.route('/predict', methods=['POST'])
def predict():
    # Obtener los datos del formulario
    season = int(request.form['season'])
    month = int(request.form['month'])
    holiday = int(request.form['holiday'])
    workingday = int(request.form['workingday'])
    weathersit = int(request.form['weathersit'])
    temp = float(request.form['temp'])
    hum = float(request.form['hum'])
    windspeed = float(request.form['windspeed'])

    # Realizar la predicción
    prediction, explanation = perform_prediction(season, month, holiday, workingday, weathersit, temp, hum, windspeed)
    

    return render_template('result.html', prediction=prediction, explanation=explanation)

# Ruta para el análisis de datos
@app.route('/data-analysis')
def data_analysis():
    df = load_dataset()

    # Generar gráficos de análisis y convertir a base64
    temperature_plot_url = analyze_temperature_and_convert_to_base64(df)
    month_distribution_plot_url = analyze_month_distribution_and_convert_to_base64(df)
    season_distribution_plot_url = analyze_season_distribution_and_convert_to_base64(df)

    return render_template('data_analysis.html', temperature_plot_url=temperature_plot_url, 
                           month_distribution_plot_url=month_distribution_plot_url,
                           season_distribution_plot_url=season_distribution_plot_url)

# Función para realizar análisis de temperatura y convertir a base64
def analyze_temperature_and_convert_to_base64(df):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='temp', y='cnt', data=df)
    plt.title('Relación entre Temperatura y Rentas de Bicicletas')
    plt.xlabel('Temperatura (Celsius)')
    plt.ylabel('Número de Alquileres')
    plt.grid(True)
    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()  # Cerrar la figura para liberar memoria

    return plot_url

# Función para realizar análisis de distribución por mes y convertir a base64
def analyze_month_distribution_and_convert_to_base64(df):
    plt.figure(figsize=(10, 6))
    sns.barplot(x='mnth', y='cnt', data=df, estimator=sum, ci=None)
    plt.title('Distribución de Alquileres de Bicicletas por Mes')
    plt.xlabel('Mes')
    plt.ylabel('Número de Alquileres')
    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()  # Cerrar la figura para liberar memoria

    return plot_url

# Función para realizar análisis de distribución por temporada y convertir a base64
def analyze_season_distribution_and_convert_to_base64(df):
    plt.figure(figsize=(10, 6))
    sns.barplot(x='season', y='cnt', data=df, estimator=sum, ci=None)
    plt.title('Distribución de Alquileres de Bicicletas por Temporada')
    plt.xlabel('Temporada')
    plt.ylabel('Número de Alquileres')
    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()  # Cerrar la figura para liberar memoria

    return plot_url

if __name__ == '__main__':
    app.run(debug=True)