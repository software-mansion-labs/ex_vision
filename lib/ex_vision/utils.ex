defmodule ExVision.Utils do
  @moduledoc false

  import Nx.Defn

  @spec load_image(Path.t()) :: Nx.Tensor.t()
  def load_image(path) do
    path
    |> StbImage.read_file!()
    |> StbImage.resize(224, 224)
    |> StbImage.to_nx()
    |> Nx.new_axis(0)
    # Convert to float
    |> Nx.divide(255)
    # fix channels position. Reshape from {batch, width, height, channels} to {batch, channels, width, height}
    |> Nx.transpose(axes: [0, 3, 1, 2])
  end

  @spec load_categories(Path.t()) :: [atom()]
  def load_categories(path) do
    path
    |> File.read!()
    |> Jason.decode!()
    |> Enum.map(fn c ->
      c |> String.downcase() |> String.replace(~r(\ |\'|\-), "_") |> String.to_atom()
    end)
  end

  defn softmax(x) do
    Nx.divide(Nx.exp(x), Nx.sum(Nx.exp(x)))
  end
end
