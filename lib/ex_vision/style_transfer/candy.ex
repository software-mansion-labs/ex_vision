defmodule ExVision.StyleTransfer.Candy do
    @moduledoc """
    An instance segmentation model with a ResNet-50-FPN backbone. Exported from torchvision.
    """
    use ExVision.Model.Definition.Ortex,
      # model: "udnie.onnx",
      model: "udnie.onnx",
      categories: "priv/categories/coco_categories.json"

    import ExVision.Utils

    require Logger

    alias ExVision.Types.BBoxWithMask

    @type output_t() :: [BBoxWithMask.t()]

    @impl true
    def load(options \\ []) do
      if Keyword.has_key?(options, :batch_size) do
        Logger.warning(
          "`:max_batch_size` was given, but this model can only process batch of size 1. Overriding"
        )
      end

      options
      |> Keyword.put(:batch_size, 1)
      |> default_model_load()
    end

    @impl true
    def preprocessing(img, _metdata) do
      # {ExVision.Utils.resize(img, {640, 480}), Nx.tensor([1.0,1.0,1.0,1.0], type: :f32)}
      Logger.info(ExVision.Utils.resize(img, {640, 480}))

      ExVision.Utils.resize(img, {640, 480}) |> Nx.divide(255.0)
    end

    @impl true
    def postprocessing(
          stylized_frame,
          metadata
        ) do
      categories = categories()

      {h, w} = metadata.original_size
      scale_x = w / 640
      scale_y = h / 480

      stylized_frame
    end
    defp clamp(tensor) do
      tensor
      |> Nx.max(0.0)
      |> Nx.min(255.0)
    end
  end
