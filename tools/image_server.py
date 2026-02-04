from flask import Flask, send_from_directory
from flask_cors import CORS

# ✅ folder where images exist
RAW_DIR = r"C:\Users\shrad\OneDrive\Desktop\infosoft project\invoice_ai_reader\data\raw"

app = Flask(__name__)
CORS(app)  # allow label studio to load images

@app.route("/images/<path:filename>")
def images(filename):
    return send_from_directory(RAW_DIR, filename)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5005, debug=False)
