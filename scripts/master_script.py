# imports
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

# read data directly into dataframe
data_frame = spark.read.csv("s3://finalproject-raw-data/Master_Ranked_Games.csv", header=True, inferSchema=True,)

# ensure no duplicate entries
data = data_frame.drop_duplicates(subset='gameid', keep='first', inplace=True)
# Write the result in Parquet format to S3 bucket
data.write.parquet("s3://finalproject-analytics/Master_dataset.parquet", mode="overwrite", compression="snappy",)

job.commit()
