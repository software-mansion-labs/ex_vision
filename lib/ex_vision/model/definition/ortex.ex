defmodule ExVision.Model.Definition.Ortex do
  @moduledoc """
  A generic implementation of the `ExVision.Model.Definition` for Ortex based models.
  """

  # TODO: improve the documentation here

  require Logger

  alias ExVision.Types.ImageMetadata

  @doc """
  A callback used to apply preprocessing for your model.

  The requirements for that will differ depending on the model used.
  """
  @callback preprocessing(Nx.Tensor.t(), ImageMetadata.t()) :: Nx.Tensor.t()

  @doc """
  A callback used to apply postprocessing to the output of the ONNX model.

  In this callback, you should transform the output to match your desired format.
  """
  @callback postprocessing(tuple(), ImageMetadata.t()) :: ExVision.Model.output_t()

  @typedoc """
  A type describing ONNX provider that can be used with ExVision.

  For some providers, it may be necessary to use the local version of `libonnxruntime` and provide some configuration option.
  For details, please consult [Ortex documentaiton](https://hexdocs.pm/ortex/Ortex.html#load/3)
  """
  @type provider_t() :: :cpu | :coreml | :cpu

  @typedoc """
  A type describing all options possible to use with the default implementation of the `load/0` function.

  - `:cache_path` - specifies a caching directory for this model.
  - `:providers` - a list of desired providers, sorted by preference. Onnx will attempt to use the first available provider. If none of the provided is available, onnx will fallback to `:cpu`. Default: `[:cpu]`
  - `:batch_size` - specifies a default batch size for this instance. Default: `1`
  """
  @type load_option_t() ::
          {:cache_path, Path.t()}
          | {:providers, [provider_t()]}
          | {:batch_size, pos_integer()}

  defmacrop get_client_preprocessing(module) do
    quote do
      fn input ->
        images = ExVision.Utils.load_image(input)

        metadata =
          Enum.map(
            images,
            &%ExVision.Types.ImageMetadata{
              original_size: ExVision.Utils.image_size(&1)
            }
          )

        batch =
          images
          |> Enum.zip(metadata)
          |> Enum.map(fn {image, metadata} -> unquote(module).preprocessing(image, metadata) end)
          |> Nx.Batch.stack()

        {batch, metadata}
      end
    end
  end

  defmacrop get_client_postprocessing(module, output_names) do
    quote do
      fn {result, _server_metadata}, metadata ->
        result
        |> split_onnx_result(unquote(output_names))
        |> Enum.zip(metadata)
        |> Enum.map(fn {result, metadata} -> unquote(module).postprocessing(result, metadata) end)
      end
    end
  end

  @doc """
  Loads the ONNX model and attaches the `Nx.Serving` to callbacks defined in the module
  """
  @spec load_ortex_model(module(), Path.t(), [load_option_t()]) ::
          {:ok, ExVision.Model.t()} | {:error, atom()}
  def load_ortex_model(module, model_path, options) do
    with {:ok, options} <-
           Keyword.validate(options, [
             :cache_path,
             batch_size: 1,
             providers: [:cpu]
           ]),
         cache_options = Keyword.take(options, [:cache_path, :file_path]),
         {:ok, path} <- ExVision.Cache.lazy_get(model_path, cache_options),
         {:ok, model} <- do_load_model(path, options[:providers]) do
      output_names = ExVision.Utils.onnx_output_names(model)

      model
      |> then(&Nx.Serving.new(Ortex.Serving, &1))
      |> Nx.Serving.batch_size(options[:batch_size])
      |> Nx.Serving.client_preprocessing(get_client_preprocessing(module))
      |> Nx.Serving.client_postprocessing(get_client_postprocessing(module, output_names))
      |> then(&{:ok, struct!(module, serving: &1)})
    end
  end

  defp do_load_model(path, providers) do
    try do
      {:ok, Ortex.load(path, providers)}
    rescue
      e in RuntimeError ->
        require Logger
        Logger.error("Failed to load model from `#{inspect(path)}` due to #{inspect(e)}")
        {:error, :onnx_load_failure}
    end
  end

  defp split_onnx_result(tuple, outputs) do
    tuple
    |> Tuple.to_list()
    |> Enum.map(fn x ->
      # Do a backend transfer and also return a list of batches here
      x |> Nx.backend_transfer() |> Nx.to_batched(1)
    end)
    |> Enum.zip()
    |> Enum.map(fn parts ->
      parts |> Tuple.to_list() |> then(&Enum.zip(outputs, &1)) |> Enum.into(%{})
    end)
  end

  @type using_option_t() :: {:base_dir, Path.t()} | {:name, String.t()}
  @spec __using__([using_option_t()]) :: Macro.t()
  defmacro __using__(opts) do
    Application.ensure_all_started(:req)

    opts = Keyword.validate!(opts, [:base_dir, :name])
    base_dir = opts[:base_dir]

    model_path = Path.join(base_dir, "model.onnx")

    quote do
      use ExVision.Model.Definition, unquote(Keyword.take(opts, [:base_dir, :name]))
      @behaviour ExVision.Model.Definition.Ortex

      @model_name unquote(opts[:name])
      @model_path unquote(model_path)

      @doc """
      Creates the model instance
      """
      @impl true
      @spec load([ExVision.Model.Definition.Ortex.load_option_t()]) ::
              {:ok, t()} | {:error, reason :: atom()}
      def load(options \\ []) do
        default_model_load(options)
      end

      defp default_model_load(options) do
        ExVision.Model.Definition.Ortex.load_ortex_model(__MODULE__, @model_path, options)
      end

      @impl true
      def postprocessing(result, _metdata), do: result

      @impl true
      def preprocessing(image, _metadata), do: image

      defoverridable load: 0, load: 1, preprocessing: 2, postprocessing: 2
    end
  end
end
