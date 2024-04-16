alias ExVision.Segmentation.DeepLabV3_MobileNetV3, as: Model

m = Model.load()
cat_path = "examples/files/cat.jpg"

Model.run(m, cat_path) |> dbg()
