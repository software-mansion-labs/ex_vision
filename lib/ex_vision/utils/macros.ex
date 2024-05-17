defmodule ExVision.Utils.Macros do
  @moduledoc false
  defmacro defunimplemented(function, options \\ []) do
    options =
      Keyword.validate!(options,
        with_impl: false,
        message: "This function is not implemented"
      )

    quote do
      if unquote(options[:with_impl]) do
        @impl true
      end

      # credo:disable-for-next-line
      def unquote(function) do
        raise RuntimeError, message: unquote(options[:message])
      end
    end
  end

  defmacro __using__(_opts) do
    quote do
      import ExVision.Utils.Macros, only: :macros
    end
  end
end
