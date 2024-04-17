defmodule ExVision.Detection.Ssdlite320_MobileNetv3.BBox do
  @moduledoc """
  A struct describing the bounding box returned by the detection model.
  """

  @enforce_keys [:x1, :y1, :x2, :y2, :label, :score]
  defstruct @enforce_keys

  @type t(label_t) :: %__MODULE__{
          x1: number(),
          y1: number(),
          y2: number(),
          x2: number(),
          label: label_t,
          score: number()
        }
end
