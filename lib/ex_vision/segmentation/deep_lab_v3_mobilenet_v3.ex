defmodule ExVision.Segmentation.DeepLabV3_MobileNetV3 do
  @moduledoc """
  A semantic segmentation model for MobileNetV3 Backbone. Exported from torchvision.
  """

  use ExVision.Model.Behavior, base_dir: "models/segmentation/deeplab_v3"
  require Bunch.Typespec
  alias ExVision.Utils

  @type output_t() :: %{category_t() => Nx.Tensor.t()}

  @spec run(t(), ExVision.Model.input_t()) :: ExVision.Model.output_t()
  def run(%__MODULE__{model: model}, input) do
    # aux is used only for training, it's not important in this case
    {out, _aux} = Ortex.run(model, preprocessing(input))
    postprocessing(out)
  end

  @spec preprocessing(ExVision.Model.input_t()) :: Nx.Tensor.t()
  def preprocessing(image), do: Utils.load_image(image, size: {224, 224}) |> elem(1)

  @spec postprocessing(Nx.Tensor.t()) :: ExVision.Model.output_t()
  def postprocessing(tensor) do
    cls_per_pixel =
      tensor
      |> Nx.backend_transfer()
      # Remove batch
      |> Nx.squeeze()
      # Apply softmax for each pixel
      |> Axon.Activations.softmax(axis: [0])
      |> Nx.argmax(axis: 0)

    categories()
    |> Enum.with_index()
    |> Map.new(fn {category, i} -> {category, Nx.equal(cls_per_pixel, i)} end)
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
