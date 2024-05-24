defmodule ExVision.TestUtils.MockCacheServer do
  @moduledoc false
  # It will add a setup step that will mock all calls to Req, eliminating the need to host the files during testing

  require Logger

  defp get_server() do
    quote do
      use Mimic

      setup_all do
        if File.exists?("models") do
          stub(Req, :get, fn
            %URI{path: path}, _options ->
              file = Path.join("models", path)

              if File.exists?(file),
                do: {:ok, %Req.Response{status: 200, body: File.read!(path)}},
                else: {:ok, %Req.Response{status: 404}}
          end)
        end

        :ok
      end
    end
  end

  defmacro __using__(_opts) do
    get_server()
  end
end
