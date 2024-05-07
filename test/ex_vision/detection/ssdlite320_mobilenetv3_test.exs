defmodule ExVision.Detection.Ssdlite320_MobileNetv3Test do
  use ExVision.Model.Case, module: ExVision.Detection.Ssdlite320_MobileNetv3

  alias Model.BBox

  @impl true
  def test_inference_result(result) do
    assert [%BBox{x1: 132, y1: 12, label: :cat, score: score}] = result
    assert score > 0.95
  end
end
