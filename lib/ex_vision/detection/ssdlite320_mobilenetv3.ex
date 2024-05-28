defmodule ExVision.Detection.Ssdlite320_MobileNetv3 do
  @moduledoc """
  SSDLite320 object detector with MobileNetV3 Large architecture, exported from torchvision.
  """
  use ExVision.Model.Definition.Ortex,
    model: "ssdlite320_mobilenetv3_detector.onnx",
    categories: "priv/categories/coco_categories.json"

  use ExVision.Detection.GenericDetector

  require Logger

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
end
