defprotocol ExVision.Model do
  @moduledoc """
  A protocol describing a generic ExVision model.
  """

  @type t() :: struct()
  @type input_t() :: Path.t() | Nx.Tensor.t() | Evision.Mat.t()
  @type output_t() :: any()

  @spec run(t(), input_t()) :: output_t()
  def run(model, input)

  @spec as_serving(t()) :: Nx.Serving.t()
  def as_serving(model)
end
