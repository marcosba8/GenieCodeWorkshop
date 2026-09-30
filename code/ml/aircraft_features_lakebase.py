# Databricks notebook source
# MAGIC %md
# MAGIC # Aircraft features → Feature View → Lakebase (Step 7)
# MAGIC
# MAGIC Reference version of the second prompt in [Step 7: ML Models](../../docs/70-ml-models.md).
# MAGIC It engineers **per-aircraft features** from a gold table, registers them as a feature
# MAGIC table / **Feature View** in **Unity Catalog**, and then publishes them to a **Lakebase**
# MAGIC online store so a model or [Databricks App](../../docs/90-lakebase-apps.md) can look up an
# MAGIC aircraft's features in real time by `icao24`.
# MAGIC
# MAGIC Keep it simple — a handful of features keyed by `icao24`.

# COMMAND ----------

dbutils.widgets.text("catalog", "marketplace", "Catalog")
dbutils.widgets.text("schema", "opensky", "Schema")
dbutils.widgets.text("gold_table", "gold_apac", "Gold table")
dbutils.widgets.text("feature_table", "aircraft_features", "Feature table name")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
gold_table = dbutils.widgets.get("gold_table")
feature_table = f"{catalog}.{schema}.{dbutils.widgets.get('feature_table')}"

source_table = f"{catalog}.{schema}.{gold_table}"
print(f"Building {feature_table} from {source_table}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Engineer per-aircraft features
# MAGIC
# MAGIC One row per aircraft (`icao24`) — the primary key used for online lookups.

# COMMAND ----------

from pyspark.sql import functions as F

features = (
    spark.table(source_table)
    .groupBy("icao24")
    .agg(
        F.avg("velocity").alias("avg_velocity"),
        F.max("baro_altitude").alias("max_baro_altitude"),
        F.count("*").alias("num_state_vectors"),
        F.avg(F.col("on_ground").cast("double")).alias("ground_share"),
    )
)
display(features)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register the Feature View in Unity Catalog
# MAGIC
# MAGIC The Feature Engineering client writes the features as a governed UC table with `icao24` as
# MAGIC its primary key — the definition a Feature View reuses for both training and serving.

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient

fe = FeatureEngineeringClient()

fe.create_table(
    name=feature_table,
    primary_keys=["icao24"],
    df=features,
    description="Per-aircraft OpenSky features keyed by icao24 (Step 7).",
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Publish to Lakebase for online serving
# MAGIC
# MAGIC To serve these features with millisecond latency, sync the feature table into a
# MAGIC **[Lakebase](https://docs.databricks.com/aws/en/oltp/)** online store. In the UI: open the
# MAGIC feature table, choose **Publish to online store**, and pick your Lakebase Postgres project;
# MAGIC Databricks keeps the online copy in sync with the Delta table.
# MAGIC
# MAGIC Once published, a model or app looks features up by the `icao24` key at request time — the
# MAGIC same key the [Databricks App](../../docs/90-lakebase-apps.md) uses — so predictions use the
# MAGIC exact features the model trained on (no train/serve skew).
