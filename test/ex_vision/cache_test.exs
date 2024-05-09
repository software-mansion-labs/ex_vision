defmodule ExVision.CacheTest do
  use ExUnit.Case, async: false

  alias ExVision.Classification.MobileNetV3Small, as: Model
  alias ExVision.Cache

  @moduletag :tmp_dir
  @source_file_path "models/classification/mobilenetv3small/model.onnx"

  setup %{tmp_dir: tmp_dir} do
    app_env_override(:server_url, URI.new!("http://localhost:8000"))
    app_env_override(:cache_dir, tmp_dir)
  end

  test "Can download the file" do
    assert {:ok, %{model: path}} = Cache.get_model_path(Model)
    verify_download(path)
  end

  test "Can will redownload deleted file" do
    assert {:ok, %{model: path}} = Cache.get_model_path(Model)
    File.rm!(path)
    assert {:ok, %{model: ^path}} = Cache.get_model_path(Model)

    verify_download(path)
  end

  test "will fail if server is unreachable" do
    app_env_override(:server_url, URI.new!("http://localhost:9999"))
    assert {:error, :download_failed} = Cache.get_model_path(Model)
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

  @expected_size @source_file_path |> File.read!() |> byte_size()
  defp verify_download(path) do
    assert File.exists?(path)
    assert not File.dir?(path)
    contents = File.read!(path)

    assert byte_size(contents) == @expected_size
    assert contents == File.read!(@source_file_path)
  end
end
