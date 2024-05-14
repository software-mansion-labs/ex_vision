defmodule ExVision.Types.ImageMetadata do
  @moduledoc """
  Type describing image metadata that is being passed to `ExVision.Model.Implementation` callbacks.
  """

  @enforce_keys [:original_size]
  defstruct @enforce_keys

  @typedoc """
  Type describing image metadata that is being passed to `ExVision.Model.Implementation` callbacks.

  - `original_size` - gives the original size of originally loaded image
  """
  @type t() :: %__MODULE__{
          original_size: ExVision.Types.image_size_t()
        }
end
