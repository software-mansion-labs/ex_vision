from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_fpn,
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
)
import torch
import json
from pathlib import Path

base_dir = Path("models/detection/ssdlite320_mobilenetv3")
base_dir.mkdir(parents=True, exist_ok=True)

model_file = base_dir / "model.onnx"
categories_file = base_dir / "categories.json"

weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
model = fasterrcnn_mobilenet_v3_large_fpn(weights=weights)
model.eval()

categories = weights.meta["categories"]
transforms = weights.transforms()

with open(categories_file, "w") as f:
    json.dump(categories, f)

onnx_input = torch.rand(1, 3, 224, 224)

torch.onnx.export(
    model,
    onnx_input,
    str(model_file),
    verbose=False,
    input_names=["input"],
    output_names=["output", "scores", "labels"],
    dynamic_axes={
        "input": {0: "batch_size"},
        "output": {0: "batch_size"},
        "scores": {0: "batch_size"},
        "labels": {0: "batch_size"},
    },
    export_params=True,
)

import onnxruntime as onnxrt

onnx_input = torch.rand(1, 3, 224, 224)

sesh = onnxrt.InferenceSession(str(model_file))
inputs = {sesh.get_inputs()[0].name: onnx_input.numpy()}
outputs = [x.name for x in sesh.get_outputs()]
print(outputs)
output = sesh.run(outputs, inputs)
print(len(output))
print(len(output[0]))
