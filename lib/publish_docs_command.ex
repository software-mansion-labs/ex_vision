defmodule Mix.Tasks.PublishDocs do
  @moduledoc "The hello mix task: `mix help hello`"
  use Mix.Task
  require Logger

  @shortdoc "Simply calls the Hello.say/0 function."
  def run(_) do
    "mix docs" |> String.to_charlist() |> :os.cmd
    "find ~+ doc/dist -name sidebar* -print0 | xargs -0 sed -i -E 's/nested_title\":\"\.([a-zA-Z_0-9]*)\"/nested_title\":\"\\\1\"/g'" |> String.to_charlist() |> :os.cmd
    Logger.info("replaced prefixes")
    "mix hex.publish docs" |> String.to_charlist() |> :os.cmd
    Logger.info("published")
  end
end
