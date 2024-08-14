defmodule ExVision.Model.Definition do
  @moduledoc """
  A module describing the behaviour that MUST be implemented by all ExVision models.
  """

  require Bunch.Typespec

  @callback load(keyword()) :: {:ok, ExVision.Model.t()} | {:error, reason :: atom()}
  @callback run(ExVision.Model.t(), ExVision.Model.input_t()) :: any()
  @callback batched_run(atom(), ExVision.Model.input_t()) :: any()
  @callback child_spec(keyword()) :: Supervisor.child_spec()
  @callback start_link(keyword()) :: GenServer.on_start()

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
    Application.ensure_all_started(:req)

    options =
      Keyword.validate!(options, [
        categories: nil,
        name: module_to_name(__CALLER__.module)
      ])

    quote do
      # conditional defintion based on whether `categories` option is present has to be moved inside __using__ macro
      # here is explenation https://cocoa-research.works/2022/10/conditional-compliation-with-if-and-use-in-elixir/
      use ExVision.Model.Definition.Parts.WithCategories, unquote(options)

      @behaviour ExVision.Model.Definition

      @derive [ExVision.Model]
      @enforce_keys [:serving]
      defstruct [:serving]

      @typedoc """
      An instance of the #{__MODULE__}
      """
      @opaque t() :: %__MODULE__{serving: Nx.Serving.t()}

      @impl true
      def start_link(options \\ []) do
        {start_link_options, load_options} =
          Keyword.split(options, [
            :partitions,
            :batch_timeout,
            :distribution_weight,
            :shutdown,
            :hibernate_after,
            :spawn_opt,
            :name
          ])

        with {:ok, model} <- load(load_options),
             do: ExVision.Model.start_link(model, start_link_options)
      end

      @doc """
      Same as `load/1`, but raises and error on failure.
      """
      @spec load!(keyword()) :: t()
      def load!(opts \\ []) do
        case load(opts) do
          {:ok, model} ->
            model

          {:error, reason} ->
            require Logger

            Logger.error(
              "Failed to load model #{unquote(options[:name])} due to #{inspect(reason)}"
            )

            raise "Failed to load model"
        end
      end

      @impl true
      @doc """
      Immediatelly applies the model to the given input, in the scope of the current process.
      """
      @spec run(t(), ExVision.Model.input_t()) :: output_t() | [output_t()]
      defdelegate run(model, input), to: ExVision.Model

      @doc """
      Submits the input for inference to the process running the Nx.Serving for this model.
      """
      @impl true
      @spec batched_run(atom(), ExVision.Model.input_t()) :: output_t() | [output_t()]
      def batched_run(name \\ __MODULE__, input), do: ExVision.Model.batched_run(name, input)

      @impl true
      @spec child_spec(keyword()) :: Supervisor.child_spec()
      def child_spec(options \\ []) do
        {child_spec_opts, start_link_options} = Keyword.split(options, [:id])
        child_spec_opts = Keyword.validate!(child_spec_opts, id: __MODULE__)

        %{
          id: child_spec_opts[:id],
          start: {__MODULE__, :start_link, [start_link_options]}
        }
      end

      defoverridable run: 2,
                     batched_run: 2,
                     child_spec: 1,
                     child_spec: 0,
                     start_link: 0,
                     start_link: 1
    end
  end
end
