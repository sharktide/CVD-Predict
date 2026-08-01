import tensorflow
from transformers import AutoModel
model = AutoModel.from_pretrained("sharktide/ohca-predictor-v1")
print(model.__all__)