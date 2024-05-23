import Config

config :ortex, Ortex.Native, features: ["coreml"]

config :ex_vision,
  server_url: "EX_VISION_HOSTING_URI" |> System.get_env("http://localhost:8000") |> URI.new!(),
  cache_path: System.get_env("EX_VISION_CACHE_DIR", "models")
