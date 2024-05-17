defprotocol ExVision.Model do
  @moduledoc """
  A protocol describing a generic ExVision model.
  """

  @typedoc """
  A type describing a single element that can be processed by ExVision's models
  """
  @type model_input_t() :: Path.t() | Nx.Tensor.t() | Vix.Vips.Image.t()

  @typedoc """
  A typespec definiting ExVision's model input, either as single `t:model_input_t/0` or a list.
  """
  @type input_t() :: model_input_t() | [model_input_t()]

  @typedoc """
  A generic type indicating a model output. For details on each model, refer to it's own `output_t()` definition.
  """
  @type output_t() :: any()

  @doc """
  Starts and links the module in process workflow
  """
  @spec start_link(t(), keyword()) :: GenServer.on_start()
  def start_link(model, options \\ [])

  @doc """
  A function used to submit input for inference (inline variant).
  """
  @spec run(t(), input_t()) :: output_t() | [output_t()]
  def run(model, input)

  @doc """
  Function used to submit the input for inference in a process setting when the model is served as a process.
  """
  @spec batched_run(t(), input_t()) :: output_t()
  def batched_run(model, input)

  @spec as_serving(t()) :: Nx.Serving.t()
  def as_serving(model)
end

defimpl ExVision.Model, for: Any do
  require Logger

  def run(model, input) when is_list(input) do
    model |> as_serving() |> Nx.Serving.run(input)
  end

  def run(model, input) do
    model
    |> run([input])
    |> hd()
  end

  def start_link(model, options \\ []) do
    options
    |> validate_start_link_options!(name: process_name(model))
    |> Keyword.put(:serving, as_serving(model))
    |> Nx.Serving.start_link()
  end

  def batched_run(model, input) do
    Logger.warning("""
    Calling batched_run/2 at the ExVision.Model struct can lead to undefined behaviour.
    Referencing the already running process by name is preffered.
    """)

    model
    |> process_name()
    |> ExVision.Utils.batched_run(input)
  end

  def as_serving(%{serving: serving}), do: serving

  defp process_name(%module{}), do: module

  defp validate_start_link_options!(options, extras) do
    spec =
      [
        :partitions,
        :batch_timeout,
        :distribution_weight,
        :shutdown,
        :hibernate_after,
        :spawn_opt,
        :name
      ] -- Keyword.keys(extras)

    Keyword.validate!(
      options,
      spec ++ extras
    )
  end
end

defimpl ExVision.Model, for: Atom do
  use ExVision.Utils.Macros
  defunimplemented(run(_model, _input), with_impl: true)
  defunimplemented(start_link(_model, _opts), with_impl: true)
  defunimplemented(as_serving(_model), with_impl: true)

  @impl true
  def batched_run(module, input) do
    ExVision.Utils.batched_run(module, input)
  end
end
