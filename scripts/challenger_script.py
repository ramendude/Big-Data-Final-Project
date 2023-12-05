import sys
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job

args = getResolvedOptions(sys.argv, ["JOB_NAME"])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args["JOB_NAME"], args)

# Read CSV data directly into a DataFrame
data_frame = spark.read.csv(
    "s3://finalproject-raw-data/Challenger_Ranked_Games.csv",
    header=True,
    inferSchema=True,
)

# Write the result in Parquet format to Amazon S3
dataframe.write.parquet(
    "s3://finalproject-analytics/Challenger_dataset.parquet",
    mode="overwrite",
    compression="snappy",
)

job.commit()
