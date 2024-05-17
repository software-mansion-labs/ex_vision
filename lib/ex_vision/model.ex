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
  A function used to submit input for inference (inline variant).
  """
  @spec run(t(), input_t()) :: output_t() | [output_t()]
  def run(model, input)

  @doc """
  Function used to obtain a child spec for a generic ExVision model.
  """
  @spec child_spec(t(), keyword()) :: Supervisor.child_spec()
  def child_spec(model, options \\ [])

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

  def child_spec(model, options \\ []) do
    options =
      Keyword.validate!(options, [
        :partitions,
        :batch_timeout,
        :distribution_weight,
        :shutdown,
        :hibernate_after,
        :spawn_opt,
        :batch_keys,
        name: process_name(model)
      ])

    options |> Keyword.put(:serving, as_serving(model)) |> Nx.Serving.child_spec()
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
end

defimpl ExVision.Model, for: Atom do
  use ExVision.Utils.Macros
  defunimplemented(run(_model, _input), with_impl: true)
  defunimplemented(as_serving(_model), with_impl: true)
  defunimplemented(child_spec(_model, _options), with_impl: true)

  @impl true
  def batched_run(module, input) do
    ExVision.Utils.batched_run(module, input)
  end
end
