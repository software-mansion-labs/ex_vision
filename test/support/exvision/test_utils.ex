defmodule ExVision.TestUtils do
  @moduledoc false

  import ExUnit.Assertions, only: :macros

  @default_delta 0.05

  @doc """
  Compares to floats by ensuring that the distance between them is smaller than specified delta
  """
  @spec float_eq(float(), float(), float()) :: boolean()
  def float_eq(a, b, delta \\ @default_delta) do
    abs(a - b) < delta
  end

  @typedoc """
  Type describing a dictionary which values are floats
  """
  @type float_dict_t() :: %{any() => float()}

  @spec float_dict_eq(float_dict_t(), float_dict_t(), number()) :: boolean()
  def float_dict_eq(a, b, delta \\ @default_delta) do
    keys = MapSet.new(Map.keys(a) ++ Map.keys(b))

    Enum.reduce(keys, true, fn key, acc ->
      a = a[key]
      b = b[key]

      acc and not is_nil(a) and not is_nil(b) and float_eq(a, b, delta)
    end)
  end

  defmacro assert_floats_equal(a, b, delta \\ @default_delta) do
    quote do
      assert ExVision.TestUtils.float_eq(unquote(a), unquote(b), unquote(delta))
    end
  end

  defmacro assert_float_dicts_equal(a, b, delta \\ @default_delta) do
    quote do
      assert ExVision.TestUtils.float_dict_eq(unquote(a), unquote(b), unquote(delta))
    end
  end

  defmacro assert_tensors_equal(a, b, delta \\ @default_delta, relative_delta \\ 0.0) do
    quote do
      value_condition =
        unquote(a)
        |> Nx.all_close(unquote(b), atol: unquote(delta), rtol: unquote(relative_delta))
        |> Nx.reduce_min()
        |> Nx.to_number() == 1

      equal_on_count =
        unquote(a)
        |> Nx.equal(unquote(b))
        |> Nx.as_type(:u64)
        |> Nx.reduce(0, fn x, y -> Nx.add(x, y) end)
        |> Nx.to_number()

      number_count = unquote(a) |> Nx.shape() |> Tuple.product()
      proportional_condition = equal_on_count / number_count > 0.99

      assert value_condition or proportional_condition
    end
  end

  defmacro __using__(_opts) do
    quote do
      import ExVision.TestUtils, only: :macros
    end
  end
end
