# ExVision

[![Hex.pm](https://img.shields.io/hexpm/v/ex_vision.svg)](https://hex.pm/packages/ex_vision)
[![API Docs](https://img.shields.io/badge/api-docs-yellow.svg?style=flat)](https://hexdocs.pm/ex_vision)

ExVision is the collection of AI models related to vision delivered with ready to use package and easy to understand API.
ExVision will take care of all necessary input transformations internally and return the result in the sensible format.

ExVision models are powered by [Ortex](https://www.github.com/elixir-nx/ortex).

## Usage

In order to use the model, you need to first load it

```elixir
alias ExVision.Classification.MobileNetV3

model = MobileNetV3.load() #=> %MobileNetV3{}
```

After that, the model is available for inference.
ExVision will take care of all necessary input transformations and covert output to a format that makes sense.

```elixir
MobileNetV3.run(model, "example/files/cat.jpg") #=> %{cat: 0.98, dog: 0.01, car: 0.00, ...}
```

ExVision is also capable of accepting tensors on input:

```elixir
cat = "example/files/cat.jpg" |> StbImage.read_file!() |> StbImage.to_nx()
MobileNetV3.run(model, cat) #=> %{cat: 0.98, dog: 0.01, car: 0.00, ...}
```

## Installation

The package can be installed by adding `ex_vision` to your list of dependencies in `mix.exs`:

```elixir
def deps do
  [
    {:ex_vision, "~> 0.1.0"}
  ]
end
```

In order to compile, ExVision **requires Rust and Cargo** to be installed on your system.

## Current Timeline

We have identified a set of models that we would like to support.
If the model that you would like to use is missing, feel free to open the issue, express interest in an existing one or contribute the model directly.

- [ ] Classification
  - [x] MobileNetV3 Small
  - [ ] EfficientNetV2
  - [ ] SqueezeNet
- [ ] Object detection
  - [x] SSDLite320 - MobileNetV3 Large backbone
- [x] Semantic segmentation
  - [x] DeepLabV3 - MobileNetV3
- [ ] Instance segmentation
  - [x] Mask R-CNN
- [ ] Keypoint Detection
  - [ ] Keypoint R-CNN

## Copyright and License

Copyright 2024, [Software Mansion](https://swmansion.com/?utm_source=git&utm_medium=readme&utm_campaign=ex_vision)

[![Software Mansion](https://logo.swmansion.com/logo?color=white&variant=desktop&width=200&tag=membrane-github)](https://swmansion.com/?utm_source=git&utm_medium=readme&utm_campaign=ex_vision)

Licensed under the [Apache License, Version 2.0](LICENSE)
