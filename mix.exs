defmodule PowerFlowSolver.MixProject do
  use Mix.Project

  @version "0.1.15"
  @source_url "https://github.com/voltageAI/power_flow_solver"

  def project do
    [
      app: :power_flow_solver,
      version: @version,
      elixir: "~> 1.17",
      start_permanent: Mix.env() == :prod,
      description: description(),
      package: package(),
      deps: deps(),
      docs: docs()
    ]
  end

  def application do
    [
      extra_applications: [:logger]
    ]
  end

  defp description do
    """
    High-performance power flow solver using Rust NIFs. Includes Newton-Raphson solver
    with Q-limit enforcement and sparse linear algebra operations optimized for large
    power systems.
    """
  end

  defp package do
    [
      files: ~w(lib native .formatter.exs mix.exs README.md LICENSE CHANGELOG.md),
      licenses: ["MIT"],
      links: %{
        "GitHub" => @source_url
      }
    ]
  end

  defp deps do
    [
      {:decimal, "~> 2.0"},
      {:rustler, "~> 0.36.0"},
      {:ex_doc, "~> 0.31", only: :dev, runtime: false},
      {:power_system_parsers, github: "voltageAI/power_system_parsers", tag: "v0.3.0", only: :test}
    ]
  end

  defp docs do
    [
      main: "PowerFlowSolver",
      source_url: @source_url,
      source_ref: "v#{@version}",
      extras: ["README.md", "CHANGELOG.md"]
    ]
  end
end
