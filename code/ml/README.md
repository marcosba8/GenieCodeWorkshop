# ML code (Step 7)

Reference notebooks for [Step 7: ML Models](../../docs/70-ml-models.md) — the kind of code
**Genie Code** generates from the prompts on that page. Both run on serverless compute /
Databricks Free Edition and default to the same `marketplace.opensky.state_vectors` table the rest
of the workshop uses (catalog / schema / table are notebook widgets).

| Notebook | What it does | Prompt |
|----------|--------------|--------|
| `flight_phase_model.py` | Trains a flight-phase classifier (Ground / Climb / Descent / Cruise) on `state_vectors`, tracks the run with MLflow, and registers it to Unity Catalog as `flight_phase_model`. | Prompt 1 |
| `aircraft_features_lakebase.py` | Engineers per-aircraft features from `state_vectors` keyed by `icao24`, registers the `aircraft_features` Feature View in Unity Catalog, and publishes it to the `opensky_lakebase` online store for real-time lookups. | Prompt 2 |

> **Source vs. target:** both notebooks *read* from the read-only `marketplace.opensky` catalog and
> *write* (model, feature table) to a separate **writable** catalog/schema — the `target_catalog` /
> `target_schema` widgets (default `main` / `opensky_ml`). Set them to a catalog you can create in.

Import a notebook into your workspace (**Workspace → Import**, or `databricks workspace import`),
set the widgets at the top, and run. They're a copy-paste starting point — Genie Code will produce
its own version tuned to your prompt and data.
