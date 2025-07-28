from pyspark import SparkContext, SparkConf

def word_count():
    # 配置并初始化SparkContext
    conf = SparkConf().setAppName("WordCountWithOperators")
    sc = SparkContext(conf=conf)

    # 1. 读取文本文件，生成初始RDD
    # 支持本地文件（file:///路径）或HDFS文件（hdfs:///路径）
    lines = sc.textFile("input.txt")

    # 2. 转换操作：一系列算子组合
    word_counts = lines \
        .filter(lambda line: line.strip() != "") \
        .flatMap(lambda line: line.split()) \
        .map(lambda word: word.strip().lower().strip('.,!?;:"()[]')) \
        .filter(lambda word: word != "") \
        .map(lambda word: (word, 1)) \
        .reduceByKey(lambda a, b: a + b) \
        .sortBy(lambda x: x[1], ascending=False)

    # 3. 行动操作：获取并输出结果
    results = word_counts.collect()  # 将分布式计算结果收集到Driver端

    # 打印前100个结果
    for word, count in results[:100]:
        print(f"{word}: {count}")

    # 可选：将结果保存到文件
    word_counts.saveAsTextFile("output_wordcount_rdd")

    # 停止SparkContext
    sc.stop()

if __name__ == "__main__":
    word_count()
