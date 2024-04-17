defmodule ExVision.Segmentation.DeepLabV3_MobileNetV3 do
  @moduledoc """
  A semantic segmentation model for MobileNetV3 Backbone. Exported from torchvision.
  """

  use ExVision.Model.Behavior, base_dir: "models/segmentation/deeplab_v3"

  require Bunch.Typespec

  alias ExVision.Utils

  @spec run(t(), ExVision.Model.input_t()) :: ExVision.Model.output_t()
  def run(%__MODULE__{model: model}, input) do
    model
    |> Ortex.run(preprocessing(input))
    |> elem(0)
    |> postprocessing()
  end

  @spec preprocessing(ExVision.Model.input_t()) :: Nx.Tensor.t()
  defdelegate preprocessing(image), to: Utils, as: :load_image

  @spec postprocessing(Nx.Tensor.t()) :: ExVision.Model.output_t()
  def postprocessing(tensor) do
    tensor
    |> Nx.backend_transfer()
    # Remove batch
    |> Nx.squeeze()
    # Apply softmax for each pixel
    |> Axon.Activations.softmax(axis: [0])
    # Split categories
    |> Nx.to_batched(1)
    |> Enum.map(&Nx.squeeze/1)
    |> then(&Enum.zip(categories(), &1))
    |> Map.new()
  end
end

defimpl ExVision.Model, for: ExVision.Segmentation.DeepLabV3_MobileNetV3 do
  alias ExVision.Segmentation.DeepLabV3_MobileNetV3, as: Model

  @spec as_serving(Model.t()) :: Nx.Serving.t()
  def as_serving(%Model{model: model}) do
    Nx.Serving.new(Ortex.Serving, model)
  end

  @spec run(Model.t(), ExVision.Model.input_t()) :: ExVision.Model.output_t()
  defdelegate run(model, input), to: Model
end
