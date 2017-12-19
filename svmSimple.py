# 简化版SMO算法(第二个α的选择是随机的)
import numpy as np
import matplotlib.pyplot as plt
def loadDataSet(fileName):
    dataMat, labelMat = [], []
    with open(fileName) as fr:
        for line in fr.readlines():
            lineArr = line.strip().split(',')
            dataMat.append([float(lineArr[0]), float(lineArr[1])])
            labelMat.append(float(lineArr[2]))
    return dataMat, labelMat
# 可视化数据集
def showClassifer(X, Y, alphas, w, b):
    classifiedPts = {'+1': [], '-1': []}
    for point, label in zip(X, Y):
        classifiedPts['+1'].append(point) if label == 1.0 else classifiedPts['-1'].append(point)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # 绘制数据点
    for label, pts in classifiedPts.items():
        pts = np.array(pts)
        ax.scatter(pts[:, 0], pts[:, 1], label=label)
    # 绘制分割线
    x10, _ = max(X, key=lambda x: x[0])
    x11, _ = min(X, key=lambda x: x[0])
    a1, a2 = w
    x20, x21 = (-b - a1 * x10) / a2, (-b - a1 * x11) / a2
    ax.plot([x10, x11], [x20, x21])
    # 绘制两条虚线边界
    d = 1.0 / np.sqrt(a1 * a1 + a2 * a2)
    xa20, xa21 = (-b - a1 * x10 + d) / a2, (-b - a1 * x11 + d) / a2
    xb20, xb21 = (-b - a1 * x10 - d) / a2, (-b - a1 * x11 - d) / a2
    ax.plot([x10, x11], [xa20, xa21], '--')
    ax.plot([x10, x11], [xb20, xb21], '--')
    # 绘制支持向量
    for i, alpha in enumerate(alphas):
        if abs(alpha) > 1e-3:
            x1, x2 = X[i]
            ax.scatter([x1], [x2], s=150, c='none', alpha=0.7, linewidth=1.5, edgecolor='#AB3319')
    plt.show()
# 从m中随机选取一个除了i之外剩余的数
def selectToJ(i, m):
    l = list(range(m))
    seq = l[: i] + l[i + 1:]
    return np.random.choice(seq)
# 根据aj的取值范围修剪aj的值
def clipForAj(aj, L, H):
    if aj < L:
        aj = L
    if aj > H:
        aj = H
    return aj
# 根据计算好的拉格朗乘子计算W向量
def getW(X, Y, alphas):
    alphas, X, Y = np.array(alphas), np.array(X), np.array(Y)
    yx = Y.reshape(1, -1).T * np.array([1, 1]) * X
    w = np.dot(yx.T, alphas)
    return w
# 超平面方程
def fx(x, X, Y, alphas, b):
    fx = np.matrix(alphas * Y) * np.matrix(X) * np.matrix(x).T + b
    return fx[0, 0]
# 简化版SMO算法
def simpleSMO(X, Y, C, toler, maxIter):
    X, Y = np.array(X), np.array(Y)  # 得到输入样本点全集和样本标签点全集
    m, n = X.shape  # 样本集合的行数和列数
    alphas = np.zeros(m)  # 初始化m个拉格朗乘子
    b = 0  # 初始化超平面方程的截距值
    iterNum = 0  # 已经迭代的次数
    while iterNum < maxIter:
        alphaPairsChanged = 0
        for i in range(m):
            # 步骤1：计算误差
            a_i, x_i, y_i = alphas[i], X[i], Y[i]
            fx_i = fx(x=x_i, X=X, Y=Y, alphas=alphas, b=b)
            E_i = fx_i - y_i  # 计算Ei误差
            # 如果误差较大，且a_i不满足KKT条件，则需要更新优化a_i
            if ((y_i * E_i < -toler) and (a_i < C)) or ((y_i * E_i > toler) and (a_i > 0)):
                j = selectToJ(i, m)  # 随机选择另一个与a_i成对优化的a_j
                a_j, x_j, y_j = alphas[j], X[j], Y[j]
                fx_j = fx(x=x_j, X=X, Y=Y, alphas=alphas, b=b)
                E_j = fx_j - y_j  # 计算Ej误差
                # 步骤2：计算a_j的上界H和下界L
                a_i_old, a_j_old = a_i.copy(), a_j.copy()  # 保留更新之前的a_i值和a_j值
                if y_i != y_j:
                    L = max(0, a_j_old - a_i_old)
                    H = min(C, C + a_j_old - a_i_old)
                else:
                    L = max(0, a_i_old + a_j_old - C)
                    H = min(C, a_j_old + a_i_old)
                if L == H:
                    print("L==H")
                    continue
                # 步骤3：根据多元函数推导出的一元函数后的系数eta，计算学习率eta
                K_ii, K_jj, K_ij = np.dot(x_i, x_i), np.dot(x_j, x_j), np.dot(x_i, x_j)  # 计算各个向量的点积
                eta = K_ii + K_jj - 2 * K_ij
                if eta <= 0:
                    print('eta <= 0')
                    continue
                # 步骤4：根据SMO算法推理出的一元函数的迭代公式，更新a_j（最复杂的推理过程）
                a_j_new = a_j_old + y_j * (E_i - E_j) / eta
                # 步骤5：根据a_j的取值范围修剪更新后的a_j
                a_j_new = clipForAj(aj=a_j_new, L=L, H=H)
                # 步骤6：根据约束条件推导出的迭代公式，更新a_i
                a_i_new = a_i_old + y_i * y_j * (a_j_old - a_j_new)
                if abs(a_j_new - a_j_old) < 0.00001:
                    print('alpha_j变化太小')
                    continue
                # 步骤7：根据支持向量点方程推理出的公式，更新b_i和b_j
                b_i = b - E_i - y_i * K_ii * (a_i_new - a_i_old) - y_j * K_ij * (a_j_new - a_j_old)
                b_j = b - E_j - y_i * K_ij * (a_i_new - a_i_old) - y_j * K_jj * (a_j_new - a_j_old)
                # 步骤8：根据b_i和b_j更新b
                if 0 < a_i_new < C:
                    b = b_i
                elif 0 < a_j_new < C:
                    b = b_j
                else:
                    b = (b_i + b_j) / 2.0
                alphas[i], alphas[j] = a_i_new, a_j_new  # 维护拉格朗乘子集合的第i个和第j乘子为更新后的a_i和a_j
                alphaPairsChanged += 1  # 统计优化的次数
                print('经过第{0}次迭代，样本{1}的alpha优化次数{2}'.format(iterNum, i, alphaPairsChanged))
        # 更新迭代次数
        if alphaPairsChanged == 0:
            iterNum += 1
        else:
            iterNum = 0
        print('迭代的次数为{0}'.format(iterNum))
    return alphas, b
# 测试数据集错误率
def loss(X, Y, alphas, w, b):
    X, Y = np.array(X), np.array(Y)
    m = np.shape(X)[0]
    svInd = np.nonzero(alphas > 0)[0]
    errorCount = 0
    for i in range(m):
        y_ = predict(x=X[i], w=w, b=b)
        if y_ != np.sign(Y[i]):
            errorCount += 1
    print("在{}个样本里，预测正确有{}个样本，支持向量个数{}个，样本错误率为{}%".format(m, m - errorCount, np.shape(X[svInd])[0], float(errorCount) / m * 100))
# 用计算出的向量w和偏移量b，预测样本x的归类
def predict(x, w, b):
    return np.sign(np.dot(w.T, x) + b)
if __name__ == '__main__':
    X, Y = loadDataSet('./data/dataLine.csv')
    alphas, b = simpleSMO(X=X, Y=Y, C=0.6, toler=0.001, maxIter=40)
    w = getW(X=X, Y=Y, alphas=alphas)
    loss(X=X, Y=Y, alphas=alphas, w=w, b=b)
    showClassifer(X=X, Y=Y, alphas=alphas, w=w, b=b)
