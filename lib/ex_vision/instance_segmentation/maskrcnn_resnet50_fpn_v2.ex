defmodule ExVision.InstanceSegmentation.MaskRCNN_ResNet50_FPN_V2 do
  @moduledoc """
  An instance segmentation model with a ResNet-50-FPN backbone. Exported from torchvision.
  """
  use ExVision.Model.Definition.Ortex,
    model: "maskrcnn_resnet50_fpn_v2_instance_segmentation.onnx",
    categories: "priv/categories/coco_categories.json"

  import ExVision.Utils

  require Logger

  alias ExVision.Types.BBoxWithMask

  @type output_t() :: [BBoxWithMask.t()]

  @impl true
  def load(options \\ []) do
    if Keyword.has_key?(options, :batch_size) do
      Logger.warning(
        "`:max_batch_size` was given, but this model can only process batch of size 1. Overriding"
      )
    end

    options
    |> Keyword.put(:batch_size, 1)
    |> default_model_load()
  end

  @impl true
  def preprocessing(img, _metdata) do
    ExVision.Utils.resize(img, {224, 224})
  end

  @impl true
  def postprocessing(
        %{
          "boxes_unsqueezed" => bboxes,
          "labels_unsqueezed" => labels,
          "masks_unsqueezed" => masks,
          "scores_unsqueezed" => scores
        },
        metadata
      ) do
    categories = categories()

    {h, w} = metadata.original_size
    scale_x = w / 224
    scale_y = h / 224

    bboxes = scale_and_listify_bbox(bboxes, Nx.f32([scale_x, scale_y, scale_x, scale_y]))

    scores = squeeze_and_listify(scores)
    labels = squeeze_and_listify(labels)

    masks =
      masks
      |> Nx.backend_transfer()
      |> Nx.squeeze(axes: [0, 2])
      |> NxImage.resize(metadata.original_size, channels: :first)
      |> Nx.round()
      |> Nx.as_type(:s64)
      |> Nx.to_list()

    [bboxes, labels, scores, masks]
    |> Enum.zip()
    |> Enum.filter(fn {_bbox, _label, score, _mask} -> score > 0.1 end)
    |> Enum.map(fn {[x1, y1, x2, y2], label, score, mask} ->
      %BBoxWithMask{
        x1: x1,
        y1: y1,
        x2: x2,
        y2: y2,
        label: Enum.at(categories, label),
        score: score,
        mask: Nx.tensor(mask)
      }
    end)
  end
end
