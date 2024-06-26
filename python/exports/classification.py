import argparse
from torchvision.transforms.functional import to_tensor, resize
import torch
import json
from pathlib import Path
from PIL import Image


def export(model_builder, Model_Weights, input_shape):
    base_dir = Path(f"models/classification/{model_builder.__name__}")
    base_dir.mkdir(parents=True, exist_ok=True)

    model_file = base_dir / "model.onnx"
    categories_file = base_dir / "categories.json"

    weights = Model_Weights.DEFAULT
    model = model_builder(weights=weights)
    model.eval()

    categories = [x.lower().replace(" ", "_")
                  for x in weights.meta["categories"]]
    transforms = weights.transforms()

    with open(categories_file, "w") as f:
        json.dump(categories, f)

    onnx_input = to_tensor(Image.open("test/assets/cat.jpg")).unsqueeze(0)
    onnx_input = resize(onnx_input, input_shape)
    onnx_input = transforms(onnx_input)

    torch.onnx.export(
        model,
        onnx_input,
        str(model_file),
        verbose=False,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        },
        export_params=True,
    )

    expected_output: torch.Tensor = model(onnx_input)
    expected_output = expected_output.softmax(dim=1)

    result = dict(zip(categories, expected_output[0].tolist()))

    file = Path(
        f"test/assets/results/classification/{model_builder.__name__}.json"
    )
    file.parent.mkdir(exist_ok=True, parents=True)

    with file.open("w") as f:
        json.dump(result, f)


parser = argparse.ArgumentParser()
parser.add_argument("model")
args = parser.parse_args()

match(args.model):
    case "mobilenet_v3_small":
        from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
        export(mobilenet_v3_small, MobileNet_V3_Small_Weights, [224, 224])
    case "efficientnet_v2_s":
        from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
        export(efficientnet_v2_s, EfficientNet_V2_S_Weights, [384, 384])
    case "efficientnet_v2_m":
        from torchvision.models import efficientnet_v2_m, EfficientNet_V2_M_Weights
        export(efficientnet_v2_m, EfficientNet_V2_M_Weights, [480, 480])
    case "efficientnet_v2_l":
        from torchvision.models import efficientnet_v2_l, EfficientNet_V2_L_Weights
        export(efficientnet_v2_l, EfficientNet_V2_L_Weights, [480, 480])
    case "squeezenet1_1":
        from torchvision.models import squeezenet1_1, SqueezeNet1_1_Weights
        export(squeezenet1_1, SqueezeNet1_1_Weights, [224, 224])
    case _:
        print("Model not found")
