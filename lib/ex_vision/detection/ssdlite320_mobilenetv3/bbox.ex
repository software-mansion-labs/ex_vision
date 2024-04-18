defmodule ExVision.Detection.Ssdlite320_MobileNetv3.BBox do
  @moduledoc """
  A struct describing the bounding box returned by the detection model.
  """

  @enforce_keys [:x1, :y1, :x2, :y2, :label, :score]
  defstruct @enforce_keys

  @typedoc """
  A type describing the Bounding Box object.

  Bounding box is a rectangle encompassing the region.
  When used in object detectors, this box will describe the location of the object in the image.

  - `x1` - x componenet of the upper left corner
  - `y1` - y componenet of the upper left corner
  - `x2` - x componenet of the lower right
  - `y2` - y componenet of the lower right
  - `score` - confidence of the predition
  - `label` - label assigned to this bounding box.
  """
  @type t(label_t) :: %__MODULE__{
          x1: number(),
          y1: number(),
          y2: number(),
          x2: number(),
          label: label_t,
          score: number()
        }
end
