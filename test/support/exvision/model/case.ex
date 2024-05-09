defmodule ExVision.Model.Case do
  @moduledoc false
  @img_path "test/assets/cat.jpg"

  defmodule Behaviour do
    @moduledoc false
    @callback test_inference_result(result :: any()) :: any()
  end

  defmacro __using__(opts) do
    opts = Keyword.validate!(opts, [:module])

    quote do
      use ExUnit.Case, async: true
      @behaviour ExVision.Model.Case.Behaviour
      alias unquote(opts[:module]), as: Model

      setup_all do
        {:ok, model} = Model.load()
        serving = ExVision.Model.as_serving(model)

        [
          serving: serving,
          model: model
        ]
      end

      describe "standalone usage" do
        test "load/0", %{model: model} do
          assert model
        end

        test "inference", %{model: model} do
          model
          |> Model.run(unquote(@img_path))
          |> test_inference_result()
        end
      end

      describe "usage as Nx.Serving" do
        test "loads", %{serving: s} do
          assert s
        end

        test "inference", %{serving: s} do
          s
          |> Nx.Serving.run(unquote(@img_path))
          |> test_inference_result()
        end
      end
    end
  end
end
