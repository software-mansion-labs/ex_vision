defmodule ExVision.Cache do
  @moduledoc false

  # Module responsible for handling model file caching

  require Logger

  # TODO: add configurable cache directory and server url
  defp cache_dir(), do: Application.fetch_env!(:ex_vision, :cache_dir)
  defp server_url(), do: Application.fetch_env!(:ex_vision, :server_url)

  @type cache_entry_t() :: %{model: Path.t()}

  @spec get_model_path(module()) ::
          {:ok, cache_entry_t()} | {:error, reason :: atom()}
  def get_model_path(model) do
    with {:ok, path} <- path_for_model(model) do
      model_path = Path.join(path, "model.onnx")
      cache_path = Path.join(cache_dir(), model_path)
      ok? = File.exists?(cache_path)

      if ok? do
        Logger.info(
          "Found existing cache entry for #{inspect(model)} at `#{cache_path}`. Loading."
        )

        {:ok, %{model: cache_path}}
      else
        Logger.info("Downloading model for #{inspect(model)}")
        download_cache_dir(model_path, cache_path)
      end
    end
  end

  defp path_for_model(module) do
    case Module.split(module) do
      ["ExVision" | rest] ->
        rest
        |> Enum.map(&String.downcase/1)
        |> Path.join()
        |> then(&{:ok, &1})

      _otherwise ->
        {:error, :module_not_supported}
    end
  end

  @spec download_cache_dir(Path.t(), Path.t()) ::
          {:ok, cache_entry_t()} | {:error, reason :: any()}
  defp download_cache_dir(path, cache) do
    with :ok <- cache |> Path.dirname() |> File.mkdir_p(),
         :ok <- download_file(path, cache) do
      {:ok, %{model: cache}}
    end
  end

  @spec download_file(String.t(), Path.t()) :: :ok | {:error, reason :: any()}
  def download_file(url, target_file_path) do
    Logger.debug("Downloading file from `#{url}` and saving to `#{target_file_path}`")
    url = URI.append_path(server_url(), ensure_backslash(url))

    with :ok <- target_file_path |> Path.dirname() |> File.mkdir_p!(),
         target_file = File.stream!(target_file_path),
         {:ok, _resp} <- get(url, raw: true, into: target_file) do
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
