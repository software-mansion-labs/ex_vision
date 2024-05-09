defmodule ExVision.Classification.MobileNetV3Test do
  @moduledoc false
  use ExVision.Model.Case, module: ExVision.Classification.MobileNetV3Small
  use ExVision.TestUtils

  @expected_result "test/assets/results/classification/mobilenetv3.json"
                   |> File.read!()
                   |> Jason.decode!()
                   |> Map.new(fn {k, v} -> {ExVision.Utils.normalize_category_name(k), v} end)

  @impl true
  def test_inference_result(result) do
    assert_float_dicts_equal(@expected_result, result)

    top_result = Enum.max_by(result, &elem(&1, 1))
    assert {:tabby, _pred} = top_result
  end
end
