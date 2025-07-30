from flask import Flask, render_template, request
import joblib
import os
import pickle
from pickle import load

app = Flask(__name__, template_folder='templates', static_folder='static')

# Carga del modelo
modelo_path = os.path.join(os.path.dirname(__file__), 'modelo_diabetes_arbol.pkl')
with open(modelo_path, 'rb') as f:
    modelo = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediccion = None
    if request.method == 'POST':
        try:
            # Suponemos que el input se llama 'input1' y es numérico
            valor = float(request.form['input1'])
            prediccion = modelo.predict([[valor]])[0]
        except Exception as e:
            prediccion = f"Error: {str(e)}"
    return render_template('index.html', prediccion=prediccion)

if __name__ == '__main__':
    app.run(debug=True)

