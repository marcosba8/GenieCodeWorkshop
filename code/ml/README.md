# ML code (Step 7)

Reference notebooks for [Step 7: ML Models](../../docs/70-ml-models.md) — the kind of code
**Genie Code** generates from the prompts on that page. Both run on serverless compute /
Databricks Free Edition and take their catalog / schema / gold table as notebook widgets.

| Notebook | What it does | Prompt |
|----------|--------------|--------|
| `flight_phase_model.py` | Trains a flight-phase classifier (Ground / Climb / Descent / Cruise) on a gold table, tracks the run with MLflow, and registers the model to Unity Catalog. | Prompt 1 |
| `aircraft_features_lakebase.py` | Engineers per-aircraft features keyed by `icao24`, registers them as a Feature View in Unity Catalog, and publishes to a Lakebase online store for real-time lookups. | Prompt 2 |

Import a notebook into your workspace (**Workspace → Import**, or `databricks workspace import`),
set the widgets at the top, and run. They're a copy-paste starting point — Genie Code will produce
its own version tuned to your prompt and data.
