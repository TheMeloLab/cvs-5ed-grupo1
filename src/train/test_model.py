# download_model.py
import os

from roboflow import Roboflow
from dotenv import load_dotenv


load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))
project = rf.workspace("ml-sdznj").project("yolov8-number-plate-detection")
version = project.version(1)

# Descarrega o modelo treinado
model = version.model  

# Ou descarrega o dataset completo para fine-tuning depois
dataset = version.download("yolov8")