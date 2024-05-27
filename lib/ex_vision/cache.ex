defmodule ExVision.Cache do
  @moduledoc false
  # Module responsible for handling model file caching

  use GenServer
  require Logger

  @type lazy_get_option_t() :: {:force, boolean()}

  @doc """
  Lazily evaluate the path from the cache directory.
  It will only download the file if it's missing or the `force: true` option is given.
  """
  @spec lazy_get(term() | pid(), Path.t(), options :: [lazy_get_option_t()]) ::
          {:ok, Path.t()} | {:error, reason :: atom()}
  def lazy_get(server, path, options \\ []) do
    with {:ok, options} <- Keyword.validate(options, force: false),
         do: GenServer.call(server, {:download, path, options}, :infinity)
  end

  @spec start_link(keyword()) :: GenServer.on_start()
  def start_link(opts) do
    {init_args, opts} = Keyword.split(opts, [:server_url, :cache_path])
    GenServer.start_link(__MODULE__, init_args, opts)
  end

  @impl true
  def init(opts) do
    opts = Keyword.validate!(opts, cache_path: get_cache_path(), server_url: get_server_url())

    with {:ok, server_url} <- URI.new(opts[:server_url]),
         :ok <- File.mkdir_p(opts[:cache_path]) do
      {:ok,
       %{
         downloads: %{},
         server_url: server_url,
         cache_path: opts[:cache_path],
         refs: %{}
       }}
    end
  end

  @impl true
  def handle_call({:download, cache_path, options}, from, state) do
    file_path = Path.join(state.cache_path, cache_path)

    updated_downloads =
      Map.update(state.downloads, cache_path, MapSet.new([from]), &MapSet.put(&1, from))

    cond do
      Map.has_key?(state.downloads, cache_path) ->
        {:noreply, %{state | downloads: updated_downloads}}

      File.exists?(file_path) or options[:force] ->
        {:reply, {:ok, file_path}, state}

      true ->
        ref = do_create_download_job(cache_path, state)

        {:noreply,
         %{state | downloads: updated_downloads, refs: Map.put(state.refs, ref, cache_path)}}
    end
  end

  @impl true
  def handle_info({ref, result}, state) do
    Logger.info("Task #{inspect(ref)} finished with #{inspect(result)}")
    state = emit(result, ref, state)
    {:noreply, state}
  end

  @impl true
  def handle_info({:DOWN, ref, :process, _pid, reason}, state) do
    state =
      if reason != :normal do
        Logger.error("Task #{inspect(ref)} has crashed due to #{inspect(reason)}")
        emit({:error, reason}, ref, state)
      else
        state
      end

    {:noreply, state}
  end

  @impl true
  def handle_info(msg, state) do
    Logger.warning("Received an unknown message #{inspect(msg)}. Ignoring")
    {:noreply, state}
  end

  defp emit(message, ref, state) do
    path = state.refs[ref]

    state.downloads
    |> Map.get(path, [])
    |> Enum.each(fn from ->
      GenServer.reply(from, message)
    end)

    %{state | refs: Map.delete(state.refs, ref), downloads: Map.delete(state.downloads, path)}
  end

  defp do_create_download_job(path, %{server_url: server_url, cache_path: cache_path}) do
    target_file_path = Path.join(cache_path, path)
    download_url = URI.append_path(server_url, ensure_backslash(path))

    %Task{ref: ref} =
      Task.async(fn ->
        download_file(download_url, target_file_path)
      end)

    ref
  end

  @default_cache_path Application.compile_env(:ex_vision, :cache_path, "/tmp/ex_vision/cache")
  defp get_cache_path() do
    Application.get_env(:ex_vision, :cache_path, @default_cache_path)
  end

  @default_server_url Application.compile_env(
                        :ex_vision,
                        :server_url,
                        URI.new!("https://ai.swmansion.com/exvision/files")
                      )
  defp get_server_url() do
    Application.get_env(:ex_vision, :server_url, @default_server_url)
  end

  @spec download_file(URI.t(), Path.t()) ::
          {:ok, Path.t()} | {:error, reason :: any()}
  defp download_file(url, cache_path) do
    with :ok <- cache_path |> Path.dirname() |> File.mkdir_p(),
         tmp_file_path = cache_path <> ".unconfirmed",
         tmp_file = File.stream!(tmp_file_path),
         :ok <- do_download_file(url, tmp_file),
         :ok <- validate_download(tmp_file_path),
         :ok <- File.rename(tmp_file_path, cache_path) do
      {:ok, cache_path}
    end
  end

  defp ensure_backslash("/" <> _rest = i), do: i
  defp ensure_backslash(i), do: "/" <> i

  defp validate_download(path) do
    if File.exists?(path),
      do: :ok,
      else: {:error, :download_failed}
  end

  @spec do_download_file(URI.t(), File.Stream.t()) :: :ok | {:error, reason :: any()}
  defp do_download_file(%URI{} = url, %File.Stream{path: target_file_path} = target_file) do
    Logger.debug("Downloading file from `#{url}` and saving to `#{target_file_path}`")

    case make_get_request(url, raw: true, into: target_file) do
      {:ok, _resp} ->
        :ok

      {:error, reason} = error ->
        Logger.error("Failed to download the file due to #{inspect(reason)}")
        File.rm(target_file_path)
        error
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
end
