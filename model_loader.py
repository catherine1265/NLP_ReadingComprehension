import os
import gdown

MODELS = {
    "model.pkl"              : "1BT5HNwFJuaRsfuIOnpnDmxGBP2oMsYwo",
    "squad_preprocessed.csv" : "1BLbROC4QhxAMR33v-nb51GvlemFWZAo-",
}

def ensure_models():
    for filename, file_id in MODELS.items():
        if not os.path.exists(filename):
            gdown.download(f"https://drive.google.com/uc?id={file_id}", filename, quiet=False)
