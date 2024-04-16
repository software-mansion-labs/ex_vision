defmodule ExVision.Utils do
  import Nx.Defn

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

  defn softmax(x) do
    Nx.divide(Nx.exp(x), Nx.sum(Nx.exp(x)))
  end
end
