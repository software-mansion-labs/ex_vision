from torchvision.models import squeezenet1_1, SqueezeNet1_1_Weights
from torchvision.transforms.functional import to_tensor, resize
import torch
import json
from pathlib import Path
from PIL import Image

base_dir = Path("models/classification/squeezenet1_1")
base_dir.mkdir(parents=True, exist_ok=True)

model_file = base_dir / "model.onnx"
categories_file = base_dir / "categories.json"

weights = SqueezeNet1_1_Weights.DEFAULT
model = squeezenet1_1(weights=weights)
model.eval()

categories = [x.lower().replace(" ", "_") for x in weights.meta["categories"]]
transforms = weights.transforms()

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

result = dict(zip(categories, expected_output[0].tolist()))

file = Path("test/assets/results/classification/squeezenet1_1.json")
file.parent.mkdir(exist_ok=True, parents=True)

with file.open("w") as f:
    json.dump(result, f)
