import Config

# Force build from source since precompiled NIFs are not available
config :rustler_precompiled, :force_build, power_flow_solver: true
