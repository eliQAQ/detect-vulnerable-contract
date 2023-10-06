# confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

# classes = ['A','B','C','D','E']
# confusion_matrix = np.array([(9,1,3,4,0),(2,13,1,3,4),(1,4,10,0,13),(3,1,1,17,0),(0,0,0,1,14)],dtype=np.float64)


# 标签
classes = ['positive', 'negative']

classNamber = 2  # 类别数量

# 混淆矩阵
confusion_matrix = np.array([
    (0.9523, 0.0476),
    (0.0118, 0.9881)
], dtype=np.float64)

plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)  # 按照像素显示出矩阵
plt.title('confusion_matrix')  # 改图名
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=-45)
plt.yticks(tick_marks, classes)

thresh = confusion_matrix.max() / 2.
# iters = [[i,j] for i in range(len(classes)) for j in range((classes))]
# ij配对，遍历矩阵迭代器
iters = np.reshape([[[i, j] for j in range(classNamber)] for i in range(classNamber)], (confusion_matrix.size, 2))
for i, j in iters:
    plt.text(j, i, format(confusion_matrix[i, j]), va='center', ha='center')  # 显示对应的数字

plt.ylabel('Ture')
plt.xlabel('Prediction')
plt.tight_layout()
plt.show()

