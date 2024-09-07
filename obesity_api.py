import joblib
import pandas as pd
from flask import Flask, jsonify
from flask_pydantic import validate
from pydantic import BaseModel

# Create a Flask app
app = Flask(__name__)


# Define a Pydantic model for the request body
class RequestBody(BaseModel):
    genero_masculino: int
    idade: int
    historico_familiar_sobrepeso: int
    consumo_alta_caloria_com_frequencia: int
    consumo_vegetais_com_frequencia: int
    refeicoes_dia: int
    consumo_alimentos_entre_refeicoes: int
    fumante: int
    consumo_agua: int
    monitora_calorias_ingeridas: int
    nivel_atividade_fisica: int
    nivel_uso_tela: int
    consumo_alcool: int
    transporte_automovel: int
    transporte_bicicleta: int
    transporte_motocicleta: int
    transporte_publico: int
    transporte_caminhada: int


# Load the model
model = joblib.load("models/obesity_model.pkl")


# Define a route for the prediction
@app.route("/predict", methods=["POST"])
@validate(body=RequestBody)
def predict(body: RequestBody):
    # Create a DataFrame from the request body
    predict_df = pd.DataFrame(dict(body), index=[1])

    # Preprocess the data
    bins = [10, 20, 30, 40, 50, 60, 70]
    ordinal_bins = [0, 1, 2, 3, 4, 5]
    predict_df["faixa_etaria"] = pd.cut(
        x=predict_df["idade"], bins=bins, labels=ordinal_bins, include_lowest=True
    )

    # Select the features
    predict_df = predict_df[
        [
            "historico_familiar_sobrepeso",
            "consumo_alta_caloria_com_frequencia",
            "consumo_alimentos_entre_refeicoes",
            "monitora_calorias_ingeridas",
            "nivel_atividade_fisica",
            "nivel_uso_tela",
            "transporte_caminhada",
            "faixa_etaria",
        ]
    ]

    # Make the prediction
    y_pred = model.predict(predict_df)[0].astype(int)

    # Return the prediction
    return jsonify({"obeso": y_pred.tolist()})


# Run the app
if __name__ == "__main__":
    app.run()
