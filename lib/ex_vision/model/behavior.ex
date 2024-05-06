defmodule ExVision.Model.Behavior do
  @moduledoc false

  require Bunch.Typespec
  alias ExVision.Utils

  defmodule Metadata do
    @moduledoc false

    @enforce_keys [:original_size]
    defstruct @enforce_keys ++ [options: []]

    @type t(options_t) :: %__MODULE__{
            original_size: ExVision.Types.image_size_t(),
            options: options_t
          }

    @type t() :: t([])
  end

  @callback load() :: ExVision.Model.t()
  @callback run(ExVision.Model.t(), ExVision.Model.input_t()) :: any()
  @callback preprocessing(ExVision.Model.input_t(), Metadata.t([any()])) :: Nx.Tensor.t()
  @callback postprocessing(tuple(), Metadata.t([any()])) :: ExVision.Model.output_t()

  @type using_option_t() :: {:base_dir, Path.t()} | {:name, String.t()}

  defp module_to_name(module),
    do:
      module
      |> Module.split()
      |> List.last()
      |> String.split("_")
      |> Enum.map_join(" ", fn <<first::binary-size(1), rest::binary>> ->
        String.upcase(first) <> rest
      end)

  defmacro __before_compile__(env) do
    has_type? = Module.defines_type?(env.module, {:t, 0})

    quote do
      unless unquote(has_type?) do
        @typedoc """
        A type describing the ExVision model instance for #{@model_name}
        """
        @opaque t() :: %__MODULE__{model: %Ortex.Model{}}
      end
    end
  end

  @spec __using__([using_option_t()]) :: Macro.t()
  defmacro __using__(opts) do
    module = __CALLER__.module

    opts = Keyword.validate!(opts, [:base_dir, name: module_to_name(module)])
    base_dir = opts[:base_dir]

    model = base_dir |> Path.join("model.onnx") |> Path.expand()
    unless File.exists?(model), do: throw("Model doesn't exist")

    categories_file = base_dir |> Path.join("categories.json") |> Path.expand()
    categories = if File.exists?(categories_file), do: categories_file |> Utils.load_categories()

    categories_spec =
      unless is_nil(categories),
        do: categories |> Enum.uniq() |> Bunch.Typespec.enum_to_alternative()

    quote do
      @before_compile ExVision.Model.Behavior
      @behaviour ExVision.Model.Behavior

      @model_name unquote(opts[:name])
      @model_path unquote(model)

      defstruct [:model]

      @doc """
      Creates the model instance
      """
      @impl true
      @spec load() :: t()
      def load() do
        %__MODULE__{model: Ortex.load(@model_path, [:coreml, :cuda, :cpu])}
      end

      @spec as_serving(t()) :: Nx.Serving.t()
      def as_serving(model \\ load()) do
        ExVision.Model.as_serving(model)
      end

      @spec child_spec(keyword()) :: tuple()
      def child_spec(opts) do
        opts
        |> Keyword.put(:serving, as_serving())
        |> Nx.Serving.child_spec()
      end

      @doc """
      Evaluates the model
      """
      @impl true
      @spec run(t(), ExVision.Model.input_t()) :: output_t()
      def run(%{model: model} = _model, input) do
        {original_size, image} = ExVision.Utils.load_image(input, size: {224, 224})
        metadata = %Metadata{original_size: original_size}

        image
        |> preprocessing(metadata)
        |> then(&Ortex.run(model, &1))
        |> ExVision.Utils.onnx_result_backend_transfer()
        |> postprocessing(metadata)
      end

      @impl true
      def preprocessing(image, _metadata), do: image

      defoverridable(load: 0, run: 2, preprocessing: 2)

      unless is_nil(unquote(categories)) do
        require Bunch.Typespec

        @typedoc """
        Type describing all categories recognised by #{unquote(opts[:name])}
        """
        @type category_t() :: unquote(categories_spec)

        @doc """
        Returns a list of all categories recognised by #{unquote(opts[:name])}
        """
        @spec categories() :: [category_t()]
        def categories(), do: unquote(categories)
      end

      defimpl ExVision.Model do
        @spec run(unquote(module).t(), ExVision.Model.input_t()) :: unquote(module).output_t()
        defdelegate run(model, input), to: unquote(module)

        @spec as_serving(unquote(module).t()) :: Nx.Serving.t()
        def as_serving(%unquote(module){model: model}) do
          # build the serving
          Ortex.Serving
          |> Nx.Serving.new(model)
          # Add preprocessing - this will handle our inputs and load it for the model
          |> Nx.Serving.client_preprocessing(fn input ->
            # TODO: get rid of repeated code, handle different input types
            {original_size, img} = ExVision.Utils.load_image(input, size: {224, 224})
            metadata = %Metadata{original_size: original_size}
            img = Nx.squeeze(img)
            {Nx.Batch.stack([img]), metadata}
          end)
          # post process the results
          |> Nx.Serving.client_postprocessing(fn {result, _server_metadata}, metadata ->
            result
            |> ExVision.Utils.onnx_result_backend_transfer()
            |> unquote(module).postprocessing(metadata)
          end)
        end
      end
    end
  end
end
