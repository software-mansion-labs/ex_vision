import Config

config :nx, default_backend: EXLA.Backend

config :ortex, Ortex.Native, features: ["coreml"]

import_config "#{config_env()}.exs"
