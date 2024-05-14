defmodule ExVision.Utils do
  @moduledoc false

  import Nx.Defn
  require Nx
  require Image
  alias ExVision.Types

  @type channel_spec_t() :: :first | :last
  @type pixel_size_t() :: 8 | 16 | 32 | 64
  @type pixel_type_t() :: {:u | :f, pixel_size_t()}
  @type load_image_option_t() ::
          {:pixel_type, pixel_type_t()}
          | {:channel_spec, channel_spec_t()}

  @spec load_image(ExVision.Model.input_t(), [load_image_option_t()]) :: [Nx.Tensor.t()]
  def load_image(image, options \\ []) do
    options = Keyword.validate!(options, pixel_type: {:f, 32}, channel_spec: :first)

    image
    |> read_image()
    |> List.flatten()
    |> Stream.map(&convert_pixel_type(&1, options[:pixel_type]))
    |> Stream.map(&convert_channel_spec(&1, options[:channel_spec]))
    |> Enum.to_list()
  end

  @spec convert_channel_spec(Nx.Tensor.t(), channel_spec_t()) :: Nx.Tensor.t()
  def convert_channel_spec(tensor, target) do
    current_spec = guess_channel_spec(tensor)

    cond do
      current_spec == target -> tensor
      target == :first -> Nx.transpose(tensor, axes: [2, 0, 1])
      target == :last -> Nx.transpose(tensor, axes: [1, 2, 0])
    end
  end

  @spec guess_channel_spec(Nx.Tensor.t()) :: channel_spec_t()
  defp guess_channel_spec(tensor) do
    case Nx.shape(tensor) do
      {3, _w, _h} -> :first
      {_batch, 3, _w, _h} -> :first
      {_w, _h, 3} -> :last
      {_batch, _w, _h, 3} -> :last
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

  @spec read_image(ExVision.Model.input_t()) :: [Nx.Tensor.t()]
  defp read_image(%Nx.Batch{} = batch), do: read_image(batch.stack)

  defp read_image(list) when is_list(list) do
    list |> Enum.map(&read_image/1)
  end

  defp read_image(%Vix.Vips.Image{} = image) do
    image |> Image.to_nx!() |> read_image()
  end

  defp read_image(x) when Nx.is_tensor(x) do
    ensure_grad_3(x)
  end

  defp read_image(x) when is_binary(x) do
    x |> Image.open!() |> read_image()
  end

  defp ensure_grad_3(tensor) do
    tensor
    |> Nx.shape()
    |> tuple_size()
    |> case do
      3 -> [tensor]
      4 -> tensor |> Nx.to_batched(1) |> Stream.map(&Nx.squeeze(&1, axes: [0])) |> Enum.to_list()
      other -> raise "Received unexpected tensor of grad #{other}"
    end
  end

  @type resize_spec_t() :: number() | Types.image_size_t()
  @spec resize(Nx.Tensor.t(), resize_spec_t()) :: Nx.Tensor.t()
  def resize(tensor, size) when is_number(size) do
    NxImage.resize_short(tensor, size, channels: guess_channel_spec(tensor))
  end

  def resize(tensor, size) when is_tuple(size) do
    NxImage.resize(tensor, size, channels: guess_channel_spec(tensor))
  end

  @spec image_size(Vix.Vips.Image.t() | Nx.Tensor.t()) :: Types.image_size_t()
  def image_size(%Vix.Vips.Image{} = image), do: {Image.height(image), Image.width(image)}

  def image_size(t) when Nx.is_tensor(t) do
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
    |> Enum.map(&normalize_category_name/1)
  end

  @spec normalize_category_name(String.t()) :: atom()
  def normalize_category_name(name),
    do: name |> String.downcase() |> String.replace(~r(\ |\'|\-), "_") |> String.to_atom()

  @spec onnx_result_backend_transfer(tuple()) :: tuple()
  def onnx_result_backend_transfer(tuple),
    do: tuple |> Tuple.to_list() |> Enum.map(&Nx.backend_transfer/1) |> List.to_tuple()

  @spec onnx_input_shape(struct()) :: tuple()
  def onnx_input_shape(%Ortex.Model{reference: r}) do
    ["input", "Float32", shape] =
      Ortex.Native.show_session(r)
      |> Enum.find(fn [name, _type, _shape] -> name == "input" end)
      |> hd()

    List.to_tuple(shape)
  end

  defn softmax(x) do
    Nx.divide(Nx.exp(x), Nx.sum(Nx.exp(x)))
  end
end
