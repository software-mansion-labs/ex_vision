defmodule ExVision.Classification.EfficientNet_V2_S do
  @moduledoc """
  An object classifier based on EfficientNet_V2_S.
  Exported from `torchvision`.
  Weights from Imagenet 1k.
  """
  use ExVision.Model.Definition.Ortex,
    model: "efficientnet_v2_s_classifier.onnx",
    categories: "priv/categories/imagenet_v2_categories.json"

  use ExVision.Classification.GenericClassifier

  @impl true
  def preprocessing(image, _metadata) do
    image
    |> ExVision.Utils.resize({384, 384})
    |> NxImage.normalize(
      Nx.tensor([0.485, 0.456, 0.406]),
      Nx.tensor([0.229, 0.224, 0.225]),
      channels: :first
    )
  end
end
