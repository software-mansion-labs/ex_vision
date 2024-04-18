defmodule ExVision.Types do
  @moduledoc """
  A collection of commonly used types in ExVision
  """

  @typedoc """
  Type describing image size as a two element tuple `{width, height}`
  """
  @type image_size_t() :: {width :: number(), height :: number()}
end
