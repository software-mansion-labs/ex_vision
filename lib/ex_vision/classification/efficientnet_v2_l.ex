defmodule ExVision.Classification.EfficientNet_V2_L do
  @moduledoc """
  An object classifier based on EfficientNet_V2_L.
  Exported from `torchvision`.
  Weights from Imagenet 1k.
  """
  use ExVision.Model.Definition.Ortex,
    model: "efficientnet_v2_l_classifier.onnx",
    categories: "priv/categories/imagenet_v2_categories.json"

  require Bunch.Typespec
  alias ExVision.Utils

  @typedoc """
  A type describing the output of EfficientNet_V2_L classifier as a mapping of category to probability.
  """
  @type output_t() :: %{category_t() => number()}

  @impl true
  def preprocessing(image, _metadata) do
    image
    |> ExVision.Utils.resize({480, 480})
    |> NxImage.normalize(
      Nx.tensor([0.5, 0.5, 0.5]),
      Nx.tensor([0.5, 0.5, 0.5]),
      channels: :first
    )
  end

  @impl true
  def postprocessing(%{"output" => scores}, _metadata) do
    scores
    |> Nx.backend_transfer()
    |> Nx.flatten()
    |> Utils.softmax()
    |> Nx.to_flat_list()
    |> then(&Enum.zip(categories(), &1))
    |> Map.new()
  end
end
