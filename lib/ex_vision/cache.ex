defmodule ExVision.Cache do
  @moduledoc false

  # Module responsible for handling model caching

  require Logger

  # TODO: add configurable cache directory and server url
  defp cache_dir, do: Application.fetch_env!(:ex_vision, :cache_dir)
  @server_url Application.fetch_env!(:ex_vision, :server_url)

  @type cache_entry_t() :: %{model: Path.t()}

  @spec get_model_description(ExVision.Model.t()) ::
          {:ok, cache_entry_t()} | {:error, reason :: atom()}
  def get_model_description(model) do
    path = path_for_model(model)
    cache_path = Path.join([@cache_dir, path, "model.onnx"])

    ok? = File.exists?(cache_path)

    if ok? do
      Logger.info("Found existing cache entry for #{inspect(model)} at `#{cache_path}`. Loading")
      {:ok, %{model: cache_path}}
    else
      Logger.info("Downloading model for #{inspect(model)}")

      download_cache_dir(path, cache_path)
    end
  end

  defp path_for_model(module) do
    ["ExVision" | rest] = Module.split(module)

    rest
    |> Enum.map(&String.downcase/1)
    |> Path.join()
  end

  defp download_cache_dir(path, cache) do
    with :ok <- cache |> Path.dirname() |> File.mkdir_p(),
         :ok <- download_file(path, cache) do
      {:ok, %{model: cache}}
    end
  end

  @spec download_file(String.t(), Path.t()) :: :ok | {:error, reason :: any()}
  def download_file(url, target_file_path) do
    url = URI.append_path(@server_url, ensure_backslash(url))

    with :ok <- target_file_path |> Path.dirname() |> File.mkdir_p!(),
         target_file = File.stream!(target_file_path),
         {:ok, _resp} <- get(url, into: target_file) do
      :ok
    end
  end

  @spec get(URI.t(), keyword()) :: {:ok, Req.Response.t()} | {:error, reason :: atom()}
  defp get(url, options) do
    url
    |> Req.get!(options)
    |> case do
      %{status: 200} = resp ->
        {:ok, resp}

      %{status: status} ->
        Logger.warning("Request has failed with status #{status}")
        {:error, :failed_to_fetch}
    end
  end

  defp ensure_backslash("/" <> _rest = path), do: path
  defp ensure_backslash(path), do: "/" <> path
end
