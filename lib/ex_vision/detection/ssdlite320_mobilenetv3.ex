defmodule ExVision.Detection.Ssdlite320_MobileNetv3 do
  @moduledoc """
  SSDLite320 object detector with MobileNetV3 Large architecture, exported from torchvision.
  """
  use ExVision.Model.Behavior, base_dir: "models/detection/ssdlite320_mobilenetv3"

  alias __MODULE__.BBox

  @typedoc """
  A type describing output of `run/2` as a list of a bounding boxes.

  Each bounding box describes the location of the object indicated by the `label`.
  It also provides the `score` field marking the probability of the prediction.
  Bounding boxes with very low scores should most likely be ignored.
  """
  @type output_t() :: [BBox.t(category_t())]

  @impl true
  def postprocessing({bboxes, scores, labels}, metadata) do
    {h, w} = metadata.original_size
    scale_x = w / 224
    scale_y = h / 224

    bboxes =
      bboxes
      |> Nx.multiply(Nx.tensor([scale_x, scale_y, scale_x, scale_y]))
      |> Nx.round()
      |> Nx.as_type(:s64)
      |> Nx.to_list()

    scores = scores |> Nx.to_list()
    labels = labels |> Nx.to_list()

    [bboxes, scores, labels]
    |> Enum.zip()
    |> Enum.filter(fn {_bbox, score, _label} -> score > 0.1 end)
    |> Enum.map(fn {[x1, y1, x2, y2], score, label} ->
      %BBox{
        x1: x1,
        x2: x2,
        y1: y1,
        y2: y2,
        score: score,
        label: Enum.at(categories(), label)
      }
    end)
  end
end
