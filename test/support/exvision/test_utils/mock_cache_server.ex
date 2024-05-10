defmodule ExVision.TestUtils.MockCacheServer do
  @moduledoc false

  # It will add a setup step that will mock all calls to Req, eliminating the need to host the files during testing

  defmacro __using__(_opts) do
    quote do
      use Mimic

      setup_all do
        stub(Req, :get, fn
          %URI{path: path}, _options ->
            file = Path.join("models", path)

            if File.exists?(file),
              do: {:ok, %Req.Response{status: 200, body: File.read!(path)}},
              else: {:ok, %Req.Response{status: 404}}
        end)

        :ok
      end
    end
  end
end
