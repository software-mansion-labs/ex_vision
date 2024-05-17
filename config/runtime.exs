import Config

config :ex_vision,
  server_url: "EX_VISION_HOSTING_URI" |> System.get_env("http://localhost:8000") |> URI.new!(),
  cache_path: System.get_env("EX_VISION_CACHE_DIR", "/tmp/ex_vision/cache")
