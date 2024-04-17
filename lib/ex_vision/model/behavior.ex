defmodule ExVision.Model.Behavior do
  @moduledoc false

  require Bunch.Typespec
  alias ExVision.Utils

  @callback load() :: ExVision.Model.t()
  @callback run(ExVision.Model.t(), ExVision.Model.input_t()) :: any()

  @type using_option_t() ::
          {:base_dir, Path.t()} | {:name, String.t()}

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

  defmacro __using__(opts) do
    opts = Keyword.validate!(opts, [:base_dir, name: module_to_name(__CALLER__.module)])
    base_dir = opts[:base_dir]

    model = base_dir |> Path.join("model.onnx") |> Path.expand()
    unless File.exists?(model), do: throw("Model doesn't exist")

    categories_file = base_dir |> Path.join("categories.json") |> Path.expand()
    categories = if File.exists?(categories_file), do: categories_file |> Utils.load_categories()

    categories_spec =
      unless is_nil(categories), do: Bunch.Typespec.enum_to_alternative(categories)

    quote do
      @before_compile ExVision.Model.Behavior
      @behaviour ExVision.Model.Behavior

      @model_name unquote(opts[:name])
      @model_path unquote(model)

      defstruct [:model]

      @doc """
      Creates the model instance
      """
      @spec load() :: t()
      def load() do
        %__MODULE__{model: Ortex.load(@model_path)}
      end

      defoverridable(load: 0)

      unless is_nil(unquote(categories)) do
        require Bunch.Typespec

        @categories unquote(categories)

        @typedoc """
        Type describing all available categories for this model
        """
        @type category_t() :: unquote(categories_spec)

        @doc """
        Returns a list of all available categories for this model
        """
        @spec categories() :: [category_t()]
        def categories(), do: @categories

        defoverridable(categories: 0)
      end
    end
  end
end
