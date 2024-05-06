defmodule ExVision.Segmentation.DeepLabV3_MobileNetV3Test do
  use ExVision.Model.Case, module: ExVision.Segmentation.DeepLabV3_MobileNetV3

  @impl true
  def test_inference_result(%{cat: cat, __background__: background}) do
    assert float_eq(nx_mean(cat) + nx_mean(background), 1.0)
    assert float_eq(nx_mean(cat), 0.36)
  end

  defp nx_mean(t), do: t |> Nx.mean() |> Nx.to_number()

  defp float_eq(value, target, delta \\ 0.01) do
    abs(value - target) < delta
  end
end
