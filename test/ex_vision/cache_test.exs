defmodule ExVision.CacheTest do
  use ExUnit.Case, async: false
  use Mimic

  alias ExVision.Classification.MobileNetV3Small, as: Model
  alias ExVision.Cache

  @moduletag :tmp_dir

  # TODO: rethink cache tests
  # I think we should use `:mock` library to mock the heck out of that

  setup %{tmp_dir: tmp_dir} do
    app_env_override(:server_url, URI.new!("http://mock_server:8000"))
    app_env_override(:cache_dir, tmp_dir)
  end

  setup ctx do
    files = Map.get(ctx, :files, %{
      "/test" => rand_string(256)
    })

    stub(Req, :get, fn
      (%URI{host: "mock_server", port: 8000, path: path}, options) ->
        options = Keyword.validate!(options, [:raw, :into])
        case Map.fetch(files, path) do
          {:ok, content} ->
            body = Enum.into([content], options[:into])
            {:ok, %Req.Response{status: 200, body: body}}
          :error -> {:ok, %Req.Response{status: 404}}
        end

      (_uri, _options) -> {:error, %Mint.TransportError{reason: :connection_failed}}
    end)

    [files: files]
  end

  test "Can download the file", ctx do
    [{path, expected_contents}] = Enum.to_list(ctx.files)
    expected_path = Path.join(ctx.tmp_dir, path)
    assert {:ok, %{model: ^expected_path}} = Cache.get(path)
    verify_download(expected_path, expected_contents)
  end

  test "will fail if server is unreachable" do
    app_env_override(:server_url, URI.new!("http://localhost:9999"))
    assert {:error, :connection_failed} = Cache.get("/test")
  end

  test "will fail if we request file that doesn't exist" do
    assert {:error, :doesnt_exist} = Cache.get("/idk")
  end

  defp app_env_override(key, new_value) do
    original = Application.fetch_env(:ex_vision, key)
    Application.put_env(:ex_vision, key, new_value)

    on_exit(fn ->
      case original do
        {:ok, value} -> Application.put_env(:ex_vision, key, value)
        :error -> Application.delete_env(:ex_vision, key)
      end
    end)
  end

  defp verify_download(path, expected_contents) do
    assert File.exists?(path)
    assert not File.dir?(path)
    assert File.read!(path) == expected_contents
  end

  defp rand_string(length), do: :crypto.strong_rand_bytes(length)
end
