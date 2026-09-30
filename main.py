"""mkdocs-macros hooks for the OpenSky Genie workshop.

Reads catalog / schema / table from config.yml (loaded via the macros
`include_yaml` option) and exposes a convenience `fqn` variable so pages can
write `{{ fqn }}` instead of repeating the three-part name.
"""


def define_env(env):
    v = env.variables
    catalog = v.get("catalog", "marketplace")
    schema = v.get("schema", "opensky")
    table = v.get("table", "state_vectors")
    # Fully-qualified Unity Catalog table name, e.g. marketplace.opensky.state_vectors
    v["fqn"] = f"{catalog}.{schema}.{table}"
