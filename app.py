from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# Cargar modelo entrenado
modelo = tf.keras.models.load_model("modelo.h5")

@app.route("/", methods=["GET", "POST"])
def index():

    resultado = None

    if request.method == "POST":

        cateto1 = float(request.form["cateto1"])
        cateto2 = float(request.form["cateto2"])

        datos = np.array([[cateto1, cateto2]])/100

        prediccion = modelo.predict(datos)

        resultado = round(float(prediccion[0][0] * 100), 2)

    return render_template("index.html", resultado=resultado)

if __name__ == "__main__":
    app.run()