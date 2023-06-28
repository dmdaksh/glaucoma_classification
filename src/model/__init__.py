from src.model.supervised import ResNet50, GlaucomaModel
from src.model.ssl.mae import MAE, MAELinearProbing

models = {
    "resnet50": ResNet50,
    'glaucoma_model': GlaucomaModel,
    "mae": MAE,
    "mae_linearprobing": MAELinearProbing,
}