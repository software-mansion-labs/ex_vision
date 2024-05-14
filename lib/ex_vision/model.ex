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
  @spec run(t(), input_t()) :: output_t()
  def run(model, input)

  @doc """
  Function used to obtain a child spec for a generic ExVision model.
  """
  @spec child_spec(t()) :: tuple()
  def child_spec(model)

  @doc """
  Function used to submit the input for inference in a process setting when the model is served as a process.
  """
  @spec batched_run(t(), input_t()) :: output_t()
  def batched_run(model, input)
end

defimpl ExVision.Model, for: Any do
  def run(%{serving: serving}, input) do
    Nx.Serving.run(serving, input)
  end

  def child_spec(%module{serving: serving}) do
    Nx.Serving.child_spec(serving: serving, name: process_name(module))
  end

  def batched_run(%module{}, input) do
    Nx.Serving.batched_run(process_name(module), input)
  end

  defp process_name(module), do: {:serving, module}
end
