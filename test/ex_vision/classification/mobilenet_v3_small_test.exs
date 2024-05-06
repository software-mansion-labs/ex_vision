defmodule ExVision.Classification.MobileNetV3Test do
  @moduledoc false
  use ExVision.Model.Case, module: ExVision.Classification.MobileNetV3

  @impl true
  def test_inference_result(result) do
    top_result = Enum.max_by(result, &elem(&1, 1))
    assert {:tabby, _pred} = top_result
  end
end
