defmodule ExVision.Segmentation.DeepLabV3_MobileNetV3 do
  @moduledoc """
  A semantic segmentation model for MobileNetV3 Backbone. Exported from torchvision.
  """

  alias ExVision.Utils
  require Bunch.Typespec
  use ExVision.Model.Behavior

  @dir "models/segmentation/deeplab_v3"
  @model_path @dir |> Path.join("model.onnx") |> Path.expand()
  @categories @dir
              |> Path.join("categories.json")
              |> File.read!()
              |> Jason.decode!()
              |> Enum.map(fn c ->
                c |> String.downcase() |> String.replace(~r(\ |\'|\-), "_") |> String.to_atom()
              end)

  @type category_t() :: unquote(Bunch.Typespec.enum_to_alternative(@categories))

  defstruct [:model]

  @type t() :: %__MODULE__{
          model: %Ortex.Model{}
        }

  @spec load() :: t()
  def load() do
    %__MODULE__{model: Ortex.load(@model_path)}
  end

  @spec run(t(), ExVision.Model.input_t()) :: ExVision.Model.output_t()
  def run(%__MODULE__{model: model}, input) do
    model
    |> Ortex.run(Utils.load_image(input))
    |> elem(0)
    |> Nx.backend_transfer()
    # Remove batch
    |> Nx.squeeze()
    # Apply softmax for each pixel
    |> Axon.Activations.softmax(axis: [0])
    # Split categories
    |> Nx.to_batched(1)
    |> Enum.map(&Nx.squeeze/1)
    |> then(&Enum.zip(categories(), &1))
    |> Map.new()
  end

  @spec categories() :: [category_t()]
  def categories(), do: @categories
end

defimpl ExVision.Model, for: ExVision.Segmentation.DeepLabV3_MobileNetV3 do
  alias ExVision.Segmentation.DeepLabV3_MobileNetV3, as: Model

  @spec as_serving(Model.t()) :: Nx.Serving.t()
  def as_serving(%Model{model: model}) do
    Nx.Serving.new(Ortex.Serving, model)
  end

  @spec run(Model.t(), ExVision.Model.input_t()) :: ExVision.Model.output_t()
  defdelegate run(model, input), to: Model
end
