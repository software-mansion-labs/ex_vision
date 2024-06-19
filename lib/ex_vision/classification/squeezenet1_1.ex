defmodule ExVision.Classification.SqueezeNet1_1 do
  @moduledoc """
  An object classifier based on SqueezeNet1_1.
  Exported from `torchvision`.
  Weights from Imagenet 1k.
  """
  use ExVision.Model.Definition.Ortex,
    model: "squeezenet1_1_classifier.onnx",
    categories: "priv/categories/imagenet_v2_categories.json"

  require Bunch.Typespec
  alias ExVision.Utils

  @typedoc """
  A type describing the output of SqueezeNet1_1 classifier as a mapping of category to probability.
  """
  @type output_t() :: %{category_t() => number()}

  @impl true
  def preprocessing(image, _metadata) do
    image
    |> ExVision.Utils.resize({224, 224})
    |> NxImage.normalize(
      Nx.tensor([0.485, 0.456, 0.406]),
      Nx.tensor([0.229, 0.224, 0.225]),
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
