defmodule ExVision.Classification.EfficientNet_V2_L do
  @moduledoc """
  An object classifier based on EfficientNet_V2_L.
  Exported from `torchvision`.
  Weights from Imagenet 1k.
  """
  use ExVision.Model.Definition.Ortex,
    model: "efficientnet_v2_l_classifier.onnx",
    categories: "priv/categories/imagenet_v2_categories.json"

  use ExVision.Classification.GenericClassifier

  @impl true
  def preprocessing(image, _metadata) do
    image
    |> ExVision.Utils.resize({480, 480})
    |> NxImage.normalize(
      Nx.f32([0.5, 0.5, 0.5]),
      Nx.f32([0.5, 0.5, 0.5]),
      channels: :first
    )
  end
end
