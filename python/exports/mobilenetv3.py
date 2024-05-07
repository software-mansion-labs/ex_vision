from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torchvision.io import read_file
from torchvision.transforms.functional import to_tensor, resize
import torch
import json
from pathlib import Path
from PIL import Image

base_dir = Path("models/classification/mobilenet_v3_small")
base_dir.mkdir(parents=True, exist_ok=True)

model_file = base_dir / "model.onnx"
categories_file = base_dir / "categories.json"

weights = MobileNet_V3_Small_Weights.DEFAULT
model = mobilenet_v3_small(weights=weights)
model.eval()

categories = [x.lower().replace(" ", "_") for x in weights.meta["categories"]]
transforms = weights.transforms()

print(transforms)

with open(categories_file, "w") as f:
    json.dump(categories, f)

onnx_input = to_tensor(Image.open("test/assets/cat.jpg")).unsqueeze(0)
onnx_input = resize(onnx_input, [224, 224])
onnx_input = transforms(onnx_input)

torch.onnx.export(
    model,
    onnx_input,
    str(model_file),
    verbose=False,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    export_params=True,
)

expected_output: torch.Tensor = model(onnx_input)
expected_output = expected_output.softmax(dim=1)
print(expected_output.shape)
print(expected_output)

result = dict(zip(categories, expected_output[0].tolist()))

file = Path("test/assets/results/classification/mobilenetv3.json")
file.parent.mkdir(exist_ok=True, parents=True)

with file.open("w") as f:
    json.dump(result, f)
