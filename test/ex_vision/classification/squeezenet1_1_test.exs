defmodule ExVision.Classification.SqueezeNet1_1_Test do
  @moduledoc false
  use ExVision.Model.Case, module: ExVision.Classification.SqueezeNet1_1
  use ExVision.TestUtils

  @expected_result "test/assets/results/classification/squeezenet1_1.json"
                   |> File.read!()
                   |> Jason.decode!()
                   |> Map.new(fn {k, v} -> {ExVision.Utils.normalize_category_name(k), v} end)

  @impl true
  def test_inference_result(result) do
    assert_float_dicts_equal(@expected_result, result, 0.21)

    top_result = Enum.max_by(result, &elem(&1, 1))
    assert {:egyptian_cat, _pred} = top_result
  end
end
