defmodule ExVision.Detection.Ssdlite320_MobileNetv3 do
  @moduledoc """
  SSDLite320 object detector with MobileNetV3 Large architecture, exported from torchvision.
  """
  use ExVision.Model.Behavior, base_dir: "models/detection/ssdlite320_mobilenetv3"

  alias ExVision.Utils

  alias __MODULE__.BBox

  @spec run(t(), ExVision.Model.input_t()) :: [BBox.t(category_t())]
  def run(%__MODULE__{model: model}, input) do
    {_size, image} = Utils.load_image(input, size: {224, 224}) |> elem(1)

    {bboxes, scores, labels} =
      Ortex.run(model, image)

    bboxes = bboxes |> Nx.to_list()
    scores = scores |> Nx.to_list()
    labels = labels |> Nx.to_list()

    [bboxes, scores, labels]
    |> Enum.zip()
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

defimpl ExVision.Model, for: ExVision.Detection.Ssdlite320_MobileNetv3 do
  alias ExVision.Detection.Ssdlite320_MobileNetv3, as: Model

  @spec as_serving(Model.t()) :: Nx.Serving.t()
  def as_serving(%Model{model: model}) do
    Nx.Serving.new(Ortex.Serving, model)
  end

  defdelegate run(model, input), to: Model
end
