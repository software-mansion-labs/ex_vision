defmodule ExVision.Model.Definition.Parts.WithCategories do
  @moduledoc false
  require Logger
  alias ExVision.{Cache, Utils}

  defp get_categories(file) do
    file
    |> Cache.lazy_get()
    |> case do
      {:ok, file} ->
        Utils.load_categories(file)

      error ->
        Logger.error("Failed to load categories from #{file} due to #{inspect(error)}")
        raise "Failed to load categories from #{file}"
    end
  end

  defmacro __using__(options) do
    options = Keyword.validate!(options, [:name, :categories])
    categories = options |> Keyword.fetch!(:categories) |> get_categories()
    spec = categories |> Enum.uniq() |> Bunch.Typespec.enum_to_alternative()

    quote do
      require Bunch.Typespec

      @typedoc """
      Type describing all categories recognised by #{unquote(options[:name])}
      """
      @type category_t() :: unquote(spec)

      @doc """
      Returns a list of all categories recognised by #{unquote(options[:name])}
      """
      @spec categories() :: [category_t()]
      def categories(), do: unquote(categories)
    end
  end
end
