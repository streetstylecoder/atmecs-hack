from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline
import torch

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Initialize the ChemBERTa pipeline
device = 0 if torch.cuda.is_available() else -1  # Use GPU if available, else CPU
fill_mask = pipeline(
    "fill-mask",
    model='mrm8488/chEMBL_smiles_v1',
    tokenizer='mrm8488/chEMBL_smiles_v1',
    device=device
)

@app.route('/generate-molecules', methods=['POST'])
def generate_molecules():
    try:
        # Get the input SMILES string from the request
        data = request.json
        smile = data.get("smile")

        if not smile:
            return jsonify({"error": "SMILES string is required"}), 400

        # Perform the masked token prediction
        smile=smile+"<mask>"
        predictions = fill_mask(smile)

        # Return the predictions as JSON
        return jsonify(predictions), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=8000)
