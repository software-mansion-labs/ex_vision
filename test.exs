alias ExVision.Segmentation.DeepLabV3_MobileNetV3, as: Model

cat_path = "examples/files/cat.jpg"

model = Model.load()

model |> ExVision.Model.run(cat_path) |> dbg()
