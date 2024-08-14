defmodule Configuration do
  @low_resolution {400,300}
  @high_resolution {640,480}
  def configuration do
    %{
      ExVision.StyleTransfer.Candy  => [model: "candy.onnx", resolution: @high_resolution], #fix model
      ExVision.StyleTransfer.CandyFast  => [model: "candy_fast.onnx", resolution: @low_resolution],
      ExVision.StyleTransfer.Princess  => [model: "princess.onnx", resolution: @high_resolution],
      ExVision.StyleTransfer.PrincessFast  => [model: "princess_fast.onnx", resolution: @low_resolution],
      ExVision.StyleTransfer.Udnie  => [model: "udnie.onnx", resolution: @high_resolution],
      ExVision.StyleTransfer.UdnieFast  => [model: "udnie_fast.onnx", resolution: @low_resolution],
      ExVision.StyleTransfer.Mosaic  => [model: "mosaic.onnx", resolution: @high_resolution],
      ExVision.StyleTransfer.MosaicFast  => [model: "mosaic_fast.onnx", resolution: @low_resolution],
    }
  end
end

for {module, opts} <- Configuration.configuration() do
  defmodule module do
    require Logger
    @type output_t() :: [Nx.Tensor.t()]

    use ExVision.Model.Definition.Ortex, model: unquote(opts[:model]), categories: "priv/categories/coco_categories.json"

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
      ExVision.Utils.resize(img, unquote(opts[:resolution])) |> Nx.divide(255.0)
    end

    @impl true
    def postprocessing(
          stylized_frame,
          metadata
        ) do

      stylized_frame = stylized_frame["55"]
      NxImage.resize(stylized_frame, metadata.original_size, channels: :first)
    end
  end
end
