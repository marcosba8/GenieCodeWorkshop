# 7. Train a machine learning model with Genie Code

## What you'll do

The [EDA](20-genie-eda.md) and [explore](30-genie-explore.md) steps profiled the OpenSky **state
vectors**. Now you use that same table — `{{ fqn }}` — to train a **machine learning model**, and,
as with every step so far, you describe what you want in plain English and **Genie Code** writes
and runs the code.

Genie Code includes a **[Data Science Agent](https://docs.databricks.com/aws/en/notebooks/ds-agent)**:
an AI partner that lives in the notebook, plans a multi-step ML workflow, generates the code,
runs it (with your approval), reads the outputs, and fixes its own errors as it goes. You stay in
control — it proposes each step and waits for you to approve before it runs.

You keep this deliberately simple: a **flight-phase classifier**. Using the same four-phase
definition from the [explore step](30-genie-explore.md) (Ground / Climb / Descent / Cruise), you
train a model that predicts an aircraft's flight phase from its altitude, speed, and vertical
rate, track the run with **MLflow**, and register the model to **Unity Catalog**.

## How does Genie Code help with machine learning?

Open the **Genie Code** side panel inside a notebook and give it a goal. The Data Science Agent:

- **Plans** the workflow (load data → prepare features → train → evaluate → register) and shows
  you the plan before touching anything.
- **Writes and runs** each cell, using `@table` context so it reads the right Unity Catalog table.
- **Uses the output** of each cell — metrics, errors, plots — to decide the next step, and
  repairs failing code automatically.
- **Tracks everything in MLflow** so every run, parameter, and metric is logged for you.

You approve steps one at a time, or turn on auto-approval once you trust the plan.

## Step-by-step guide

> **Step 1: Open Genie Code in a notebook**
>
> Create a new notebook on serverless compute and open the **Genie Code** panel. Add the state
> vectors table as context — `@{{ table }}` (i.e. `{{ fqn }}`). It's a large table, so let Genie
> sample it; the reference notebook caps training at ~200k rows so it runs quickly on Free Edition.
>
> **Step 2: Prompt Genie Code to train the model**
>
> Describe the label, the features, and where the model should land. Genie Code proposes the plan,
> then writes the feature-prep, training, evaluation, and registration cells:
>
> ```text
> Using @{{ table }}, train a model to classify each state vector's flight phase.
> Derive the label as: "Ground" when on_ground, "Climb" when vertical_rate > 1.5,
> "Descent" when vertical_rate < -1.5, otherwise "Cruise". Use baro_altitude,
> geo_altitude, velocity, and true_track as features. Sample the data to keep
> training fast. Track the run with MLflow autologging, print a classification
> report and confusion matrix, and register the best model to Unity Catalog as
> flight_phase_model.
> ```
>
> **Step 3: Approve and review**
>
> Approve the steps as Genie Code proposes them. When it finishes you have a trained classifier,
> an MLflow run with its parameters and metrics, and a model version registered in Unity Catalog.

## What you'll see

- An **MLflow run** under the notebook's experiment, with the model's accuracy / F1 and a
  confusion matrix logged as artifacts.
- A **registered model** in Unity Catalog named `flight_phase_model`, that any notebook, job, or
  serving endpoint can load. Register it in **your own writable catalog** — the shared `{{ catalog }}`
  marketplace catalog is read-only, so pick a catalog you can create objects in.
- A short **classification report** in the notebook output showing how well each phase is predicted.

A reference version of the notebook Genie Code produces lives in `code/ml/flight_phase_model.py`
so you have a copy-paste starting point.

## Serve features online with a Feature View and Lakebase

Training on the full table is a batch workflow. For **real-time** use — a model or app that needs
an aircraft's features the instant a request arrives — you precompute features once and serve them
from a low-latency store. On Databricks that store is **[Lakebase](https://docs.databricks.com/aws/en/oltp/)**,
a serverless Postgres online layer, and the features are defined as a **Feature View** in Unity
Catalog and synced to Lakebase for millisecond lookups.

Ask Genie Code to build it, in the same style as before:

> **Prompt: aircraft features as a Feature View served from Lakebase**
>
> ```text
> From @{{ table }}, engineer per-aircraft features keyed by icao24: average
> velocity, maximum baro_altitude, total number of state vectors, and the share
> of records where on_ground is true. Create a Feature View named aircraft_features
> in Unity Catalog, then publish it to a Lakebase online store (project
> opensky_lakebase) so a model or app can look up an aircraft's features in real
> time by icao24.
> ```

Genie Code creates the feature computation, registers the **Feature View** in Unity Catalog, and
sets up the **Lakebase** online store so the features are available for low-latency serving. The
same `icao24` key that identifies an aircraft in `{{ fqn }}` becomes the lookup key online — so the
[Databricks App](90-lakebase-apps.md) in a later step can fetch a plane's features in
milliseconds. A reference version lives in `code/ml/aircraft_features_lakebase.py`.

## Machine Learning — Beyond the Basics

A few things worth knowing once the basics work:

- **[AutoML](https://docs.databricks.com/aws/en/machine-learning/automl/)** — point it at a table
  and a target column and it trains and tunes a set of models for you, then hands back the
  best model *and* the editable notebook that produced it. Use it when you want a strong baseline
  and the code behind it, not a black box.
- **[Models in Unity Catalog](https://docs.databricks.com/aws/en/machine-learning/manage-model-lifecycle/)** —
  registered models are governed UC objects with versions, aliases, and lineage. Use it when you
  need to promote a model across environments with the same permissions and audit trail as your data.
- **[Model Serving](https://docs.databricks.com/aws/en/machine-learning/model-serving/)** — deploy
  the registered model behind a serverless REST endpoint that scales to zero. Use it when an app
  or downstream system needs online predictions.
- **[Feature Engineering in Unity Catalog](https://docs.databricks.com/aws/en/machine-learning/feature-store/)** —
  Feature Views make the same feature definitions available for both training and online serving,
  so the features a model trains on are exactly the ones it sees in production. Use it to avoid
  train/serve skew.

## Recap

You trained a **flight-phase classifier** on `{{ fqn }}` from a plain-English prompt: Genie Code
prepared the features, trained and evaluated the model, tracked it in **MLflow**, and registered
it to **Unity Catalog** — no hand-written training loop. You also defined a **Feature View** and
served it from **Lakebase** so the same features are available for real-time lookups. Next you put
governed data and features to work in an app.

---

### Tutorial navigation

| ← Previous | Overview | Next → |
|:---|:---:|---:|
| [6. Databricks Asset Bundles (DABs)](60-dabs.md) | [Table of contents](index.md) | [8. AI/BI Dashboards and Genie Agents](80-dashboards.md) |
