defmodule ExVision.Classification.MobileNetV3 do
  @moduledoc """
  An object detector based on MobileNetV1 Large.
  Exported from `torchvision`.
  Weights from Imagenet 1k.
  """
  use ExVision.Model.Behavior, base_dir: "models/classification/mobilenet_v3_small"

  require Bunch.Typespec
  alias ExVision.Utils

  @spec run(t(), ExVision.Model.input_t()) :: %{category_t() => number()}
  def run(%__MODULE__{model: model}, input) do
    {_size, image} = Utils.load_image(input, size: {224, 224})

    model
    |> Ortex.run(image)
    |> elem(0)
    |> Nx.backend_transfer()
    |> Nx.flatten()
    |> Utils.softmax()
    |> Nx.to_flat_list()
    |> then(&Enum.zip(categories(), &1))
    |> Map.new()
  end
end

defimpl ExVision.Model, for: ExVision.Classification.MobileNetV3 do
  alias ExVision.Classification.MobileNetV3, as: Model

  @spec as_serving(Model.t()) :: Nx.Serving.t()
  def as_serving(%Model{model: model}) do
    Nx.Serving.new(Ortex.Serving, model)
  end

  defdelegate run(model, input), to: Model
end
