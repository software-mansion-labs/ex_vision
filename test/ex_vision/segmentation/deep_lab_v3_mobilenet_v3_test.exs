defmodule ExVision.Segmentation.DeepLabV3_MobileNetV3Test do
  use ExVision.Model.Case, module: ExVision.Segmentation.DeepLabV3_MobileNetV3

  @impl true
  def test_inference_result(result) do
    assert %{cat: cat, __background__: background} = result,
           "The result doesn't contain required classes"

    assert float_eq(nx_mean(cat) + nx_mean(background), 1.0),
           "The segmentation seems to be incorrect"

    assert float_eq(nx_mean(cat), 0.36), "The cat seems have been misdetected"
  end

  defp nx_mean(t), do: t |> Nx.mean() |> Nx.to_number()

  defp float_eq(value, target, delta \\ 0.01) do
    abs(value - target) < delta
  end
end
