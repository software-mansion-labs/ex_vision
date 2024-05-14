defmodule ExVision.Model.Case do
  @moduledoc false
  @img_path "test/assets/cat.jpg"

  @callback test_inference_result(result :: any()) :: any()

  defmacro __using__(opts) do
    opts = Keyword.validate!(opts, [:module])

    quote do
      use ExUnit.Case, async: true
      use ExVision.TestUtils.MockCacheServer
      @behaviour ExVision.Model.Case

      setup_all do
        {:ok, model} = unquote(opts[:module]).load(cache_path: "models")
        [model: model]
      end

      test "load/0", %{model: model} do
        assert model
      end

      test "inference", %{model: model} do
        model
        |> unquote(opts[:module]).run(unquote(@img_path))
        |> test_inference_result()
      end

      test "inference for batch", %{model: model} do
        model
        |> unquote(opts[:module]).run([unquote(@img_path), unquote(@img_path)])
        |> Enum.each(&test_inference_result/1)
      end
    end
  end
end
