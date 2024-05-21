from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
)
import torch
import json
from pathlib import Path

base_dir = Path("models/detection/fasterrcnn_resnet50_fpn")
base_dir.mkdir(parents=True, exist_ok=True)

model_file = base_dir / "model.onnx"
categories_file = base_dir / "categories.json"

weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn(weights=weights)
model.eval()

categories = weights.meta["categories"]
transforms = weights.transforms()

with open(categories_file, "w") as f:
    json.dump(categories, f)

onnx_input = torch.rand(1, 3, 224, 224)
onnx_input = transforms(onnx_input)

torch.onnx.export(
    model,
    onnx_input,
    str(model_file),
    verbose=False,
    input_names=["input"],
    output_names=["boxes", "labels", "scores"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    export_params=True,
)

import onnxruntime as onnxrt

sesh = onnxrt.InferenceSession(str(model_file))
inputs = {sesh.get_inputs()[0].name: onnx_input.numpy()}
outputs = [x.name for x in sesh.get_outputs()]
print(outputs)
output = sesh.run(outputs, inputs)
print(len(output))
print(len(output[0]))
print(output)

print(model(onnx_input))
