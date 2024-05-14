defmodule ExVision.Model.Definition do
  @moduledoc """
  A module describing the behaviour that MUST be implemented by all ExVision models.
  """

  require Bunch.Typespec
  alias ExVision.{Cache, Utils}

  @callback load(keyword()) :: {:ok, ExVision.Model.t()} | {:error, reason :: atom()}
  @callback run(ExVision.Model.t(), ExVision.Model.input_t()) :: any()
  @callback batched_run(ExVision.Model.t(), ExVision.Model.input_t()) :: any()

  defp module_to_name(module),
    do:
      module
      |> Module.split()
      |> List.last()
      |> String.split("_")
      |> Enum.map_join(" ", fn <<first::binary-size(1), rest::binary>> ->
        String.upcase(first) <> rest
      end)

  defmacro __using__(options) do
    options =
      Keyword.validate!(options, [
        :base_dir,
        name: module_to_name(__CALLER__.module)
      ])

    model_path = Path.join(options[:base_dir], "model.onnx")

    categories =
      options[:base_dir]
      |> Path.join("categories.json")
      |> Cache.lazy_get(cache_path: "models")
      |> case do
        {:ok, categories_file} ->
          Utils.load_categories(categories_file)

        {:error, _reason} ->
          nil
      end

    categories_spec =
      unless is_nil(categories),
        do: categories |> Enum.uniq() |> Bunch.Typespec.enum_to_alternative()

    quote do
      @behaviour ExVision.Model.Definition

      @model_path unquote(model_path)

      @derive [ExVision.Model]
      @enforce_keys [:serving]
      defstruct [:serving]

      @typedoc """
      An instance of the #{__MODULE__}
      """
      @opaque t() :: %__MODULE__{serving: Nx.Serving.t()}

      @impl true
      defdelegate run(model, input), to: ExVision.Model

      @impl true
      defdelegate batched_run(model, input), to: ExVision.Model

      @spec child_spec(t()) :: tuple()
      defdelegate child_spec(model), to: ExVision.Model

      @spec child_spec() :: tuple()
      def child_spec() do
        child_spec(load())
      end

      defoverridable run: 2, batched_run: 2, child_spec: 0, child_spec: 1

      unless is_nil(unquote(categories)) do
        require Bunch.Typespec

        @typedoc """
        Type describing all categories recognised by #{unquote(options[:name])}
        """
        @type category_t() :: unquote(categories_spec)

        @doc """
        Returns a list of all categories recognised by #{unquote(options[:name])}
        """
        @spec categories() :: [category_t()]
        def categories(), do: unquote(categories)
      end
    end
  end
end
