from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F

def user_filter(df: DataFrame) -> DataFrame:
    """
    Filters user data by grouping on user_id:
    - If there are records with label=1 in the group, keep the last record with label=1
    - Otherwise, keep the last record in the group

    Parameters:
        df: DataFrame containing user_id, label and other data columns

    Returns:
        Filtered DataFrame
    """
    # Print data before processing
    print("="*50)
    print("Data before processing:")
    df.show()
    print("="*50)

    # Step 1: Add row numbers to maintain original order
    df_with_rn = df.withColumn("rn", F.monotonically_increasing_id())

    # Step 2: Group by user_id and calculate key values
    grouped = df_with_rn.groupBy("user_id").agg(
        F.max(F.when(F.col("label") == 1, 1).otherwise(0)).alias("has_one"),
        F.max(F.when(F.col("label") == 1, F.col("rn")).otherwise(None)).alias("max_one_rn"),
        F.max("rn").alias("max_rn")
    )

    # Step 3: Determine which row to keep for each user_id
    selected_rn = grouped.withColumn(
        "selected_rn",
        F.when(F.col("has_one") == 1, F.col("max_one_rn")).otherwise(F.col("max_rn"))
    ).select("user_id", "selected_rn")

    # Step 4: Use aliases to resolve column ambiguity
    df_with_rn_alias = df_with_rn.alias("left")
    selected_rn_alias = selected_rn.alias("right")

    # Join to get final result
    result = df_with_rn_alias.join(
        selected_rn_alias,
        (df_with_rn_alias.user_id == selected_rn_alias.user_id) &
        (df_with_rn_alias.rn == selected_rn_alias.selected_rn),
        how="inner"
    ).select(
        df_with_rn_alias.user_id,
        df_with_rn_alias.label,
        df_with_rn_alias.other_data
    )

    # Print data after processing
    print("\n" + "="*50)
    print("Data after processing:")
    result.show()
    print("="*50)

    return result

# Example usage
if __name__ == "__main__":
    # Initialize SparkSession
    spark = SparkSession.builder \
        .appName("UserFilterExample") \
        .getOrCreate()

    # Define sample data
    data = [
        (1, 0, "a"),
        (1, 1, "b"),
        (1, 0, "c"),
        (2, 0, "d"),
        (2, 0, "e"),
        (3, 1, "f"),
        (3, 1, "g"),
        (3, 0, "h"),
        (4, 0, "i"),
        (4, 1, "j"),
        (4, 1, "m"),
        (4, 1, "n"),
        (4, 1, "q")
    ]

    # Create DataFrame
    df = spark.createDataFrame(data, ["user_id", "label", "other_data"])

    # Call user filter function
    filtered_result = user_filter(df)

    # Stop SparkSession
    spark.stop()
