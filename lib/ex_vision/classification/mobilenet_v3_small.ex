defmodule ExVision.Classification.MobilenetV3 do
  @moduledoc """
  An object detector based on MobileNetV1 Large.
  Exported from `torchvision`.
  Weights from Imagenet 1k.
  """
  alias ExVision.Utils
  use ExVision.Model.Behavior
  require Bunch.Typespec

  @dir "models/classification/mobilenet_v3_small"
  @model_path @dir |> Path.join("model.onnx") |> Path.expand()
  @categories @dir
              |> Path.join("categories.json")
              |> File.read!()
              |> Jason.decode!()
              |> Enum.map(fn c ->
                c |> String.downcase() |> String.replace(~r(\ |\'|\-), "_") |> String.to_atom()
              end)

  defstruct [:model]

  @type t() :: %__MODULE__{
          model: %Ortex.Model{}
        }

  @type category_t() :: unquote(Bunch.Typespec.enum_to_alternative(@categories))

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
    |> Nx.flatten()
    |> Utils.softmax()
    |> Nx.to_flat_list()
    |> then(&Enum.zip(categories(), &1))
    |> Map.new()
  end

  @spec categories() :: [category_t()]
  def categories(), do: @categories
end

defimpl ExVision.Model, for: ExVision.Classification.MobilenetV3 do
  alias ExVision.Classification.MobilenetV3, as: Model

  @spec as_serving(Model.t()) :: Nx.Serving.t()
  def as_serving(%Model{model: model}) do
    Nx.Serving.new(Ortex.Serving, model)
  end

  @spec run(Model.t(), ExVision.Model.input_t()) :: ExVision.Model.output_t()
  defdelegate run(model, input), to: Model
end
