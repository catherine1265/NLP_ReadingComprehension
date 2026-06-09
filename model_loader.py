import os
import gdown

MODELS = {
    "model.pkl"              : "1lGWcyQBK-3375ZSxIaxcYGzpwaZI_Vhd",
    "squad_preprocessed.csv" : "1BLbROC4QhxAMR33v-nb51GvlemFWZAo-",
}

def ensure_models():
    for filename, file_id in MODELS.items():
        if not os.path.exists(filename):
            try:
                print(f"Downloading {filename}...")
                gdown.download(f"https://drive.google.com/uc?id={file_id}", filename, quiet=False)
                print(f"{filename} downloaded successfully")
            except Exception as e:
                print(f"Error downloading {filename}: {e}")
