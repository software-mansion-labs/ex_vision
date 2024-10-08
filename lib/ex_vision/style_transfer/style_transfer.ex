defmodule Configuration do
  @moduledoc false

  @low_resolution {400, 300}
  @high_resolution {640, 480}

  @spec configuration() :: %{}
  def configuration do
    %{
      ExVision.StyleTransfer.Candy => [model: "candy.onnx", resolution: @high_resolution],
      ExVision.StyleTransfer.CandyFast => [model: "candy_fast.onnx", resolution: @low_resolution],
      ExVision.StyleTransfer.Princess => [model: "princess.onnx", resolution: @high_resolution],
      ExVision.StyleTransfer.PrincessFast => [
        model: "princess_fast.onnx",
        resolution: @low_resolution
      ],
      ExVision.StyleTransfer.Udnie => [model: "udnie.onnx", resolution: @high_resolution],
      ExVision.StyleTransfer.UdnieFast => [model: "udnie_fast.onnx", resolution: @low_resolution],
      ExVision.StyleTransfer.Mosaic => [model: "mosaic.onnx", resolution: @high_resolution],
      ExVision.StyleTransfer.MosaicFast => [
        model: "mosaic_fast.onnx",
        resolution: @low_resolution
      ]
    }
  end
end

for {module, opts} <- Configuration.configuration() do
  defmodule module do
    @moduledoc """
    #{module} is a custom style transfer model optimised for devices with low computational capabilities and CPU inference.
    """
    use ExVision.Model.Definition.Ortex, model: unquote(opts[:model])

    require Logger

    @typedoc """
    A type consisting of output tesnor (stylized image tensor) from style transfer models of shape {#{Enum.join(Tuple.to_list(opts[:resolution]) ++ [3], ", ")}}.
    """
    @type output_t() :: Nx.Tensor.t()

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
      img |> ExVision.Utils.resize(unquote(opts[:resolution])) |> Nx.divide(255.0)
    end

    @impl true
    def postprocessing(
          stylized_frame,
          metadata
        ) do
      {h, w} = unquote(opts[:resolution])

      stylized_frame["55"]
      |> Nx.reshape({3, h, w}, names: [:channel, :height, :width])
      |> NxImage.resize(metadata.original_size, channels: :first, method: :bilinear)
      |> Nx.clip(0.0, 255.0)
      |> Nx.as_type(:u8)
      |> Nx.transpose(axes: [1, 2, 0])
    end
  end
end
