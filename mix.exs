defmodule ExVision.Mixfile do
  use Mix.Project

  @version "0.1.0"
  @github_url "https://github.com/membraneframework/membrane_template_plugin"

  def project do
    [
      app: :ex_vision,
      version: @version,
      elixir: "~> 1.16",
      elixirc_paths: elixirc_paths(Mix.env()),
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      dialyzer: dialyzer(),

      # hex
      description: "A collection of ONNX models with wrappers based on Ortex",
      package: package(),

      # docs
      name: "Ex Vision",
      source_url: @github_url,
      docs: docs(),
      homepage_url: "https://membrane.stream"
    ]
  end

  def application do
    [
      extra_applications: []
    ]
  end

  defp elixirc_paths(:test), do: ["lib", "test/support"]
  defp elixirc_paths(_env), do: ["lib"]

  defp deps do
    [
      {:nx, ">= 0.0.0"},
      {:ortex, ">= 0.0.0"},
      {:nx_image, "~> 0.1.2"},
      {:bunch, "~> 1.6", runtime: false},
      {:axon, "~> 0.6.1"},
      {:evision, "~> 0.1.38"},
      {:exla, ">= 0.0.0"},
      {:ex_doc, ">= 0.0.0", only: :dev, runtime: false},
      {:dialyxir, ">= 0.0.0", only: :dev, runtime: false},
      {:credo, ">= 0.0.0", only: [:dev, :test], runtime: false}
    ]
  end

  defp dialyzer() do
    opts = [
      flags: [:error_handling]
    ]

    if System.get_env("CI") == "true" do
      # Store PLTs in cacheable directory for CI
      [plt_local_path: "priv/plts", plt_core_path: "priv/plts"] ++ opts
    else
      opts
    end
  end

  defp package do
    [
      maintainers: ["Membrane Team"],
      licenses: ["Apache-2.0"],
      links: %{
        "GitHub" => @github_url
      }
    ]
  end

  defp docs do
    [
      main: "readme",
      extras: ["README.md", "LICENSE"],
      formatters: ["html"],
      source_ref: "v#{@version}",
      nest_modules_by_prefix: [
        ExVision,
        ExVision.Classification,
        ExVision.Detection,
        ExVision.Segmentation,
        ExVision.Types
      ]
    ]
  end
end
