defmodule ExVision.Utils do
  @moduledoc false

  import Nx.Defn
  require Nx
  require Image
  alias ExVision.Types

  @type channel_spec_t() :: :first | :last
  @type pixel_size_t() :: 8 | 16 | 32 | 64
  @type pixel_type_t() :: {:u | :f, pixel_size_t()}
  @type load_image_option_t() :: {:size, {number(), number()}} | {:pixel_type, pixel_type_t()}

  @spec load_image(ExVision.Model.input_t(), [load_image_option_t()]) ::
          {Types.image_size_t(), Nx.Tensor.t()}
  def load_image(image, options \\ []) do
    options = Keyword.validate!(options, [:size, pixel_type: {:f, 32}])
    target_size = Keyword.get(options, :size)

    {original_size, image} = image |> read_image(target_size)

    image =
      image
      |> convert_pixel_type(options[:pixel_type])
      |> convert_channel_spec(:first)
      |> Nx.new_axis(0)

    {original_size, image}
  end

  @spec convert_channel_spec(Nx.Tensor.t(), channel_spec_t()) :: Nx.Tensor.t()
  def convert_channel_spec(tensor, target) do
    if guess_channel_spec(tensor) != target do
      Nx.transpose(tensor, axes: [2, 0, 1])
    else
      tensor
    end
  end

  @spec guess_channel_spec(Nx.Tensor.t()) :: channel_spec_t()
  defp guess_channel_spec(tensor) do
    case Nx.shape(tensor) do
      {_batch, 3, _w, _h} -> :first
      {3, _w, _h} -> :first
      {_batch, _w, _h, 3} -> :last
      {_w, _h, 3} -> :last
      shape -> raise "Failed to infer channel spec for shape #{inspect(shape)}"
    end
  end

  @spec convert_pixel_type(Nx.Tensor.t(), pixel_type_t()) :: Nx.Tensor.t()
  def convert_pixel_type(tensor, {:f, _size} = target) do
    case Nx.type(tensor) do
      {:f, _} -> Nx.as_type(tensor, target)
      {:u, _} -> tensor |> Nx.divide(255) |> convert_pixel_type(target)
    end
  end

  def convert_pixel_type(tensor, {:u, _size} = target) do
    case Nx.type(tensor) do
      ^target -> tensor
      {:u, _size} -> Nx.as_type(tensor, target)
      {:f, _size} -> tensor |> Nx.multiply(255) |> convert_pixel_type(target)
    end
  end

  def convert_pixel_type(tensor, nil), do: tensor

  @spec read_image(ExVision.Model.input_t(), Types.image_size_t()) :: Nx.Tensor.t()
  defp read_image(%Vix.Vips.Image{} = image, t_size) do
    image |> Image.to_nx!() |> read_image(t_size)
  end

  defp read_image(x, t_size) when Nx.is_tensor(x) do
    {image_size(x), NxImage.resize(x, t_size, channels: guess_channel_spec(x))}
  end

  defp read_image(x, t_size) when is_binary(x) do
    x |> Image.open!() |> read_image(t_size)
  end

  @spec image_size(Vix.Vips.Image.t() | Nx.Tensor.t()) :: Types.image_size_t()
  defp image_size(%Vix.Vips.Image{} = image), do: {Image.height(image), Image.width(image)}

  defp image_size(t) when Nx.is_tensor(t) do
    case t |> Nx.squeeze() |> Nx.shape() do
      {3, w, h} -> {w, h}
      {w, h, 3} -> {w, h}
    end
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

  @spec onnx_result_backend_transfer(tuple()) :: tuple()
  def onnx_result_backend_transfer(tuple),
    do: tuple |> Tuple.to_list() |> Enum.map(&Nx.backend_transfer/1) |> List.to_tuple()

  defn softmax(x) do
    Nx.divide(Nx.exp(x), Nx.sum(Nx.exp(x)))
  end
end
