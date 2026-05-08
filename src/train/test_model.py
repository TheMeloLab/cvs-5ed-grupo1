# download_model.py
from roboflow import Roboflow

rf = Roboflow(api_key="CDX8D75RPaotjihDlajE")
project = rf.workspace("ml-sdznj").project("yolov8-number-plate-detection")
version = project.version(1)

# Descarrega o modelo treinado
model = version.model  

# Ou descarrega o dataset completo para fine-tuning depois
dataset = version.download("yolov8")