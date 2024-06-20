defmodule ExVision.Classification.GenericClassifier do
  alias ExVision.Utils

  def postprocessing(%{"output" => scores}, _metadata, categories) do
    scores
    |> Nx.backend_transfer()
    |> Nx.flatten()
    |> Utils.softmax()
    |> Nx.to_flat_list()
    |> then(&Enum.zip(categories, &1))
    |> Map.new()
  end

  defmacro __using__(_opts) do
    quote do
      @typedoc """
      A type describing the output of a classification model as a mapping of category to probability.
      """
      @type output_t() :: %{category_t() => number()}

      @impl true
      @spec postprocessing(map(), ExVision.Types.ImageMetadata.t()) :: output_t()
      def postprocessing(output, metadata) do
        ExVision.Classification.GenericClassifier.postprocessing(output, metadata, categories())
      end

      defoverridable postprocessing: 2
    end
  end
end
