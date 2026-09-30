# Databricks notebook source
# MAGIC %md
# MAGIC # Flight-phase classifier (Step 7)
# MAGIC
# MAGIC A simple, reference version of the notebook **Genie Code** produces in
# MAGIC [Step 7: ML Models](../../docs/70-ml-models.md). It trains a classifier that predicts an
# MAGIC aircraft's **flight phase** (Ground / Climb / Descent / Cruise) from its altitude, speed,
# MAGIC and vertical rate, tracks the run with **MLflow**, and registers the model to **Unity
# MAGIC Catalog**.
# MAGIC
# MAGIC Runs on serverless compute / Databricks Free Edition. Keep it simple — this is a teaching
# MAGIC example, not a production model.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Parameters
# MAGIC
# MAGIC The **source** defaults to the OpenSky state vectors table used throughout the workshop
# MAGIC (`marketplace.opensky.state_vectors`). The **target** is where the model is registered — it
# MAGIC must be a **writable** catalog/schema, because the `marketplace` catalog is a read-only
# MAGIC Delta Sharing catalog. Change any of these if you personalized `config.yml`.

# COMMAND ----------

# Source (read) — the shared, read-only marketplace table.
dbutils.widgets.text("catalog", "marketplace", "Source catalog")
dbutils.widgets.text("schema", "opensky", "Source schema")
dbutils.widgets.text("table", "state_vectors", "Source table")
# Target (write) — your own WRITABLE catalog/schema (NOT the read-only marketplace catalog).
dbutils.widgets.text("target_catalog", "main", "Target catalog (writable)")
dbutils.widgets.text("target_schema", "opensky_ml", "Target schema (writable)")
dbutils.widgets.text("model_name", "flight_phase_model", "Registered model name")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
table = dbutils.widgets.get("table")
target_catalog = dbutils.widgets.get("target_catalog")
target_schema = dbutils.widgets.get("target_schema")
model_name = dbutils.widgets.get("model_name")

source_table = f"{catalog}.{schema}.{table}"
registered_model = f"{target_catalog}.{target_schema}.{model_name}"
print(f"Training on {source_table} -> registering {registered_model}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load and label the data
# MAGIC
# MAGIC Derive the four-phase label with the same definition used in the
# MAGIC [explore step](../../docs/30-genie-explore.md), then sample the large state-vectors table to
# MAGIC keep training fast on Free Edition.

# COMMAND ----------

from pyspark.sql import functions as F

df = (
    spark.table(source_table)
    .where(
        F.col("baro_altitude").isNotNull()
        & F.col("velocity").isNotNull()
        & F.col("vertical_rate").isNotNull()
    )
    .withColumn(
        "flight_phase",
        F.when(F.col("on_ground"), "Ground")
        .when(F.col("vertical_rate") > 1.5, "Climb")
        .when(F.col("vertical_rate") < -1.5, "Descent")
        .otherwise("Cruise"),
    )
    .select(
        "baro_altitude", "geo_altitude", "velocity", "true_track", "flight_phase"
    )
)

# Sample down to a laptop-sized training set (cap ~200k rows).
total = df.count()
fraction = min(1.0, 200_000 / total) if total else 1.0
pdf = df.sample(fraction, seed=42).toPandas().dropna()
print(f"{total:,} labeled rows; training on {len(pdf):,} sampled rows")
print(pdf["flight_phase"].value_counts())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train, evaluate, and register
# MAGIC
# MAGIC MLflow autologging captures the parameters, metrics, and model. We register the fitted
# MAGIC model to Unity Catalog so any notebook, job, or serving endpoint can load it.

# COMMAND ----------

import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

mlflow.set_registry_uri("databricks-uc")  # register models to Unity Catalog
mlflow.sklearn.autolog()

FEATURES = ["baro_altitude", "geo_altitude", "velocity", "true_track"]
X = pdf[FEATURES]
y = pdf["flight_phase"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

with mlflow.start_run(run_name="flight_phase_rf") as run:
    model = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print(classification_report(y_test, preds))
    print("Confusion matrix:\n", confusion_matrix(y_test, preds))

    mlflow.sklearn.log_model(
        model, artifact_path="model", registered_model_name=registered_model
    )

print(f"Done. Registered model: {registered_model}")
