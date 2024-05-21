defmodule ExVision.UtilsTest do
  use ExUnit.Case, async: true
  alias ExVision.Utils

  @img_path Path.join(__DIR__, "../assets/cat.jpg")
  @categories_path Path.join(__DIR__, "../assets/categories.json")

  describe "load_image/1 loads from" do
    test "path" do
      assert [img] = Utils.load_image(@img_path)
      assert Nx.shape(img) == {3, 360, 543}
      assert Nx.type(img) == {:f, 32}
    end

    test ":image library image" do
      img = Image.open!(@img_path)
      assert [img_from_image] = Utils.load_image(img)
      assert [img_from_path] = Utils.load_image(@img_path)
      assert Nx.equal(img_from_image, img_from_path)
    end

    test "Nx.Tensor" do
      tensor = @img_path |> Image.open!() |> Image.to_nx!(shape: :hwc)
      assert [img_from_tensor] = Utils.load_image(tensor)
      assert [img_from_path] = Utils.load_image(@img_path)
      assert Nx.equal(img_from_tensor, img_from_path)
    end

    test "batched Nx.Tensor" do
      tensor =
        @img_path
        |> Image.open!()
        |> Image.to_nx!(shape: :hwc)
        |> Stream.duplicate(2)
        |> Enum.to_list()
        |> Nx.stack()

      tensor
      |> Utils.load_image()
      |> Enum.each(&assert Nx.shape(&1) == {3, 360, 543})
    end

    test "list of differently sized tensors" do
      input = [
        Nx.iota({3, 10, 20}, type: :f32),
        Nx.iota({3, 20, 10}, type: :f32)
      ]

      input
      |> Utils.load_image()
      |> Enum.zip(input)
      |> Enum.each(fn {a, b} -> assert Nx.equal(a, b) end)
    end

    test "list of tensors" do
      tensor =
        @img_path
        |> Image.open!()
        |> Image.to_nx!(shape: :hwc)
        |> Stream.duplicate(2)
        |> Enum.to_list()

      tensor
      |> Utils.load_image()
      |> Enum.each(&assert Nx.shape(&1) == {3, 360, 543})
    end

    test "list of paths" do
      assert [a, b] = img = Utils.load_image([@img_path, @img_path])

      Enum.each(img, fn img ->
        assert Nx.shape(img) == {3, 360, 543}
        assert Nx.type(img) == {:f, 32}
      end)

      assert Nx.equal(a, b)
    end

    test "Nx.Batch" do
      tensor =
        @img_path
        |> Image.open!()
        |> Image.to_nx!(shape: :hwc)

      serving = Nx.Serving.new(fn opts -> Nx.Defn.jit(fn a -> a end, opts) end)
      batch = Nx.Batch.stack([tensor, tensor])
      batch = Nx.Serving.run(serving, batch)
      output = Utils.load_image(batch)

      Enum.each(output, fn img ->
        assert Nx.shape(img) == {3, 360, 543}
      end)
    end
  end

  describe "load_image/2 handles option to" do
    test "channel spec change from to :last" do
      tensor_first = Nx.iota({3, 128, 256}, type: :f32)
      assert [tensor_last] = Utils.load_image(tensor_first, channel_spec: :last)
      assert Nx.shape(tensor_last) == {128, 256, 3}

      assert [tensor_new_last] = Utils.load_image(tensor_last, channel_spec: :last)
      assert Nx.equal(tensor_last, tensor_new_last)
    end

    test "channel spec change to :first" do
      assert [img] = Utils.load_image(@img_path, channel_spec: :last)
      assert Nx.shape(img) == {360, 543, 3}
    end

    test "pixel format change" do
      for t <- [{:u, 8}, {:f, 16}] do
        assert [img] = Utils.load_image(@img_path, pixel_type: t)
        assert Nx.type(img) == t, "assertion failed for #{inspect(t)}"
      end
    end
  end

  test "load_categories/1" do
    expected_categories =
      [
        "__background__",
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic_light",
        "fire_hydrant",
        "n/a",
        "stop_sign",
        "parking_meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "n/a",
        "backpack",
        "umbrella",
        "n/a",
        "n/a",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports_ball",
        "kite",
        "baseball_bat",
        "baseball_glove",
        "skateboard",
        "surfboard",
        "tennis_racket",
        "bottle",
        "n/a",
        "wine_glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot_dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "couch",
        "potted_plant",
        "bed",
        "n/a",
        "dining_table",
        "n/a",
        "n/a",
        "toilet",
        "n/a",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell_phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "n/a",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy_bear",
        "hair_drier",
        "toothbrush"
      ]
      |> Enum.map(&String.to_atom/1)

    assert Utils.load_categories(@categories_path) == expected_categories
  end

  describe "convert_channel_spec/2" do
    test "converts :last to :first" do
      input = Nx.iota({1, 2, 3})
      assert input |> Utils.convert_channel_spec(:first) |> Nx.shape() == {3, 1, 2}
    end

    test "converts :first to :last" do
      input = Nx.iota({3, 1, 2})
      assert input |> Utils.convert_channel_spec(:last) |> Nx.shape() == {1, 2, 3}
    end
  end
end
