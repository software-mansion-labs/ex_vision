defmodule ExVision.Classification.SqueezeNet1_1 do
  @moduledoc """
  An object classifier based on SqueezeNet1_1.
  Exported from `torchvision`.
  Weights from Imagenet 1k.
  """
  use ExVision.Model.Definition.Ortex,
    model: "squeezenet1_1_classifier.onnx",
    categories: "priv/categories/imagenet_v2_categories.json"

  use ExVision.Classification.GenericClassifier

  @impl true
  def preprocessing(image, _metadata) do
    image
    |> ExVision.Utils.resize({224, 224})
    |> NxImage.normalize(
      Nx.f32([0.485, 0.456, 0.406]),
      Nx.f32([0.229, 0.224, 0.225]),
      channels: :first
    )
  end
end
