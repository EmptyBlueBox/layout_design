import cma

# 定义一个新的目标函数，例如：平方和函数


def my_fun(x):
    print(x)
    return sum(xi**2 for xi in x)

# 定义约束条件，例如：第一个变量x[0]必须大于等于1，第二个变量x[1]必须小于等于2


def constraints(x):
    return [1 - x[0], x[1] - 2]


# 创建适应度函数对象，使用自适应拉格朗日乘子法处理约束
cfun = cma.ConstrainedFitnessAL(my_fun, constraints)

# 初始解
x0 = [0, 3, 1, -1]  # 选择一个初始解

# 初始标准差
sigma0 = 1

# 使用CMA-ES算法进行优化
x, es = cma.fmin2(cfun, x0, sigma0, {'tolstagnation': 0}, callback=cfun.update)

# 获取优化后的解
x_optimized = es.result.xfavorite

# 显示约束条件的违反情况
constraint_violations = constraints(x_optimized)
print("优化后的解:", x_optimized)
print("约束条件的违反情况:", constraint_violations)
