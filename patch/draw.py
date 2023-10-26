import matplotlib.pyplot as plt

# 定义数据
x = ["300-700", "700-1100", "1100-1500", "1500-1900", "1900-"]  # x轴数据
y1 = [142.4, 246, 412, 524, 466.89]  # 第一条曲线的y轴数据
y2 = [165.2, 256.5, 429, 542, 484.89]  # 第二条曲线的y轴数据

# 创建折线图
plt.plot(x, y1, linestyle='-', linewidth=2, color='orangered', marker='s', label='Gas optimization')
plt.plot(x, y2, linestyle='--', linewidth=2, color='deepskyblue', marker='s', label='No gas optimization')
# 添加网格线
plt.grid(True)

# 添加标题和标签
plt.title("Increased bytecode length comparison")
plt.xlabel("Bytecode length/B")
plt.ylabel("Increased bytecode length/B")

# 添加图例
plt.legend()

# 显示图形
plt.show()