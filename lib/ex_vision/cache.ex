defmodule ExVision.Cache do
  @moduledoc false

  # Module responsible for handling model file caching

  require Logger

  @default_cache_dir Application.compile_env(:ex_vision, :cache_dir, "/tmp/ex_vision/cache")
  defp get_cache_dir() do
    Application.get_env(:ex_vision, :cache_dir, @default_cache_dir)
  end

  @default_server_url Application.compile_env(:ex_vision, :server_url, "http://localhost:8000")
  defp get_server_url() do
    Application.get_env(:ex_vision, :server_url, @default_server_url)
  end

  @type lazy_get_option_t() ::
          {:cache_path, Path.t()} | {:server_url, String.t() | URI.t()} | {:force, true}

  @doc """
  Lazily evaluate the path from the cache directory.
  It will only download the file if it's missing or the `force: true` option is given.
  """
  @spec lazy_get(Path.t(), options :: [lazy_get_option_t()]) ::
          {:ok, Path.t()} | {:error, reason :: atom()}
  def lazy_get(path, options \\ []) do
    options =
      Keyword.validate!(options,
        cache_path: get_cache_dir(),
        server_url: get_server_url(),
        force: false
      )

    cache_path = Path.join(options[:cache_path], path)
    ok? = File.exists?(cache_path)

    if ok? and not options[:force] do
      Logger.debug("Found existing cache entry for #{path}. Loading.")
      {:ok, cache_path}
    else
      with {:ok, server_url} <- URI.new(options[:server_url]) do
        download_url = URI.append_path(server_url, ensure_backslash(path))
        download_file(download_url, cache_path)
      end
    end
  end

  @spec download_file(URI.t(), Path.t()) ::
          {:ok, Path.t()} | {:error, reason :: any()}
  defp download_file(url, cache) do
    with :ok <- cache |> Path.dirname() |> File.mkdir_p(),
         :ok <- do_download_file(url, cache) do
      if File.exists?(cache),
        do: {:ok, cache},
        else: {:error, :download_failed}
    end
  end

  @spec do_download_file(URI.t(), Path.t()) :: :ok | {:error, reason :: any()}
  defp do_download_file(%URI{} = url, target_file_path) do
    Logger.debug("Downloading file from `#{url}` and saving to `#{target_file_path}`")

    with :ok <- target_file_path |> Path.dirname() |> File.mkdir_p!(),
         target_file = File.stream!(target_file_path),
         {:ok, _resp} <- make_get_request(url, raw: true, into: target_file) do
      :ok
    end
  end

  defp make_get_request(url, options) do
    url
    |> Req.get(options)
    |> case do
      {:ok, %Req.Response{status: 200}} = resp ->
        resp

      {:ok, %Req.Response{status: 404}} ->
        {:error, :doesnt_exist}

      {:ok, %Req.Response{status: status}} ->
        Logger.warning("Request has failed with status #{status}")
        {:error, :server_error}

      {:error, %Mint.TransportError{reason: reason}} ->
        {:error, reason}

      {:error, _error} ->
        {:error, :connection_failed}
    end
  end

  defp ensure_backslash("/" <> _rest = path), do: path
  defp ensure_backslash(path), do: "/" <> path
end
