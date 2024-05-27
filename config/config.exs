import Config

config :nx, default_backend: EXLA.Backend
config :logger, level: :debug

config :ex_vision,
  server_url: URI.new!("https://ai.swmansion.com/exvision/files")

import_config "#{config_env()}.exs"
