defmodule Configuration do
  def configuration do
    %{
      ExVision.StyleTransfer.CandyTest  => [module: ExVision.StyleTransfer.Candy, gt_file: "cat_candy.out"],
      ExVision.StyleTransfer.CandyFastTest  => [module: ExVision.StyleTransfer.CandyFast, gt_file: "cat_candy_fast.out"],
      ExVision.StyleTransfer.PrincessTest  => [module: ExVision.StyleTransfer.Princess, gt_file: "cat_princess.out"],
      ExVision.StyleTransfer.PrincessFastTest  => [module: ExVision.StyleTransfer.PrincessFast, gt_file: "cat_princess_fast.out"],
      ExVision.StyleTransfer.UdnieTest  => [module: ExVision.StyleTransfer.Udnie, gt_file: "cat_udnie.out"],
      ExVision.StyleTransfer.UdnieFastTest  => [module: ExVision.StyleTransfer.UdnieFast, gt_file: "cat_udnie_fast.out"],
      ExVision.StyleTransfer.MosaicTest  => [module: ExVision.StyleTransfer.Mosaic, gt_file: "cat_mosaic.out"],
      ExVision.StyleTransfer.MosaicFastTest  => [module: ExVision.StyleTransfer.MosaicFast, gt_file: "cat_mosaic_fast.out"],
    }
  end
end

for {module, opts} <- Configuration.configuration() do

  defmodule module do
    use ExVision.Model.Case, module: unquote(opts[:module])
    use ExVision.TestUtils

    @impl true
    def test_inference_result(result) do
      expected_result = "test/assets/results/style_transfer/#{unquote(opts[:gt_file])}"
                    |> File.read!()
                    |> Nx.deserialize()
      assert_tensors_equal(result, expected_result)
    end
  end
end
