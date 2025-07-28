from sklearn.preprocessing import LabelEncoder

# 示例数据
labels = ["1", "2", "33", "4", "4",33]

# 创建并拟合 LabelEncoder
encoder = LabelEncoder()
encoder.fit(labels)



# 转换标签
encoded_labels = encoder.transform(labels)



print(encoded_labels)
print(type(encoded_labels))  # <class 'numpy.ndarray'>
print(encoded_labels.dtype)  # dtype('int64')


print("*"*20)
encoded_labels = encoder.transform([33])

print(encoded_labels)
print(type(encoded_labels))  # <class 'numpy.ndarray'>
print(encoded_labels.dtype)  # dtype('int64')

print("*"*20)
encoded_labels = encoder.transform(["33"])

print(encoded_labels)
print(type(encoded_labels))  # <class 'numpy.ndarray'>
print(encoded_labels.dtype)  # dtype('int64')