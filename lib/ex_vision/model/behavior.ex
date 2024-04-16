defmodule ExVision.Model.Behavior do
  @callback load() :: ExVision.Model.t()
  @callback run(ExVision.Model.t(), ExVision.Model.input_t()) :: any()

  defmacro __using__(_opts) do
    quote do
      @behaviour ExVision.Model.Behavior
    end
  end
end
