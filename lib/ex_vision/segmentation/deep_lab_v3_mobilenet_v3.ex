defmodule ExVision.Segmentation.DeepLabV3_MobileNetV3 do
  @moduledoc """
  A semantic segmentation model for MobileNetV3 Backbone. Exported from torchvision.
  """

  use ExVision.Model.Behavior, base_dir: "models/segmentation/deeplab_v3"
  require Bunch.Typespec

  @type output_t() :: %{category_t() => Nx.Tensor.t()}

  @impl true
  @spec postprocessing(tuple(), ExVision.Model.Behavior.Metadata.t()) :: output_t()
  def postprocessing({out, _aux}, _metadat) do
    cls_per_pixel =
      out
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
