defmodule ExVision.CacheTest do
  use ExUnit.Case, async: false
  use Mimic

  alias ExVision.Cache

  @moduletag :tmp_dir

  setup ctx do
    files =
      Map.get(ctx, :files, %{
        "/test" => rand_string(256)
      })

    set_mimic_global()

    stub(Req, :get, fn
      %URI{host: "mock_server", port: 8000, path: path}, options ->
        options = Keyword.validate!(options, [:raw, :into])

        case Map.fetch(files, path) do
          {:ok, content} ->
            body = Enum.into([content], options[:into])
            {:ok, %Req.Response{status: 200, body: body}}

          :error ->
            # Req seems to be saving the file anyway
            body = Enum.into([""], options[:into])
            {:ok, %Req.Response{status: 404, body: body}}
        end

      _uri, _options ->
        {:error, %Mint.TransportError{reason: :connection_failed}}
    end)

    [files: files]
  end

  setup %{tmp_dir: tmp_dir} do
    {:ok, _cache} =
      Cache.start_link(
        name: MyCache,
        server_url: URI.new!("http://mock_server:8000"),
        cache_path: tmp_dir
      )

    :ok
  end

  test "Can download the file", ctx do
    [{path, expected_contents}] = Enum.to_list(ctx.files)
    expected_path = Path.join(ctx.tmp_dir, path)
    assert {:ok, ^expected_path} = Cache.lazy_get(MyCache, path)
    verify_download(expected_path, expected_contents)
  end

  test "will fail if server is unreachable" do
    url = "http://localhost:9999"
    {:ok, c} = Cache.start_link(server_url: url, name: nil)

    assert {:error, :connection_failed} = Cache.lazy_get(c, "/test")
    assert {:error, :connection_failed} = Cache.lazy_get(c, "/test")
  end

  test "will fail if we request file that doesn't exist" do
    assert {:error, :doesnt_exist} = Cache.lazy_get(MyCache, "/idk")
    assert {:error, :doesnt_exist} = Cache.lazy_get(MyCache, "/idk")
  end

  defp verify_download(path, expected_contents) do
    assert File.exists?(path)
    assert not File.dir?(path)
    assert File.read!(path) == expected_contents
  end

  defp rand_string(length), do: :crypto.strong_rand_bytes(length)
end
