# 非线性数据SMO算法（将内积替换成核函数）
import numpy as np
import matplotlib.pyplot as plt
class optStructK:
    def __init__(self, X, Y, C, toler, kTup):
        self.X = X  # 输入样本维度全集
        self.Y = Y  # 输入样本标签全集
        self.C = C  # 松弛变量
        self.tol = toler  # 误差范围
        self.m = np.shape(X)[0]  # 样本的数量
        self.alphas = np.zeros(self.m)  # 拉格朗乘子集合
        self.b = 0.0  # 超平面方程截距
        self.eCache = np.mat(np.zeros((self.m, 2)))  # 误差缓存集合，两列，第一列代表标志位，代表误差是否有效
        self.K = np.mat(np.zeros((self.m, self.m)))  # 初始化核K
        for i in range(self.m):  # 计算所有数据的核K(x和xi的内积)
            self.K[:, i] = kernelTrans(self.X, self.X[i, :], kTup)
# 读取数据集
def loadDataSet(fileName):
    dataMat, labelMat = [], []
    with open(fileName) as fr:
        for line in fr.readlines():
            lineArr = line.strip().split(',')
            dataMat.append([float(lineArr[0]), float(lineArr[1])])
            labelMat.append(float(lineArr[2]))
    return dataMat, labelMat
# 可视化数据集
def showClassifer(X, Y, alphas, b):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    X, Y = np.array(X), np.array(Y)
    # x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
    # x2_min, x2_max = X[:, 1].min(), X[:, 1].max()
    # x1, x2 = np.mgrid[x1_min:x1_max:100j, x2_min:x2_max:100j]
    # grid_test = np.stack((x1.flat, x2.flat), axis=1)
    # grid_hat = []
    # for x in grid_test:
    #     grid_hat.append((np.matrix(alphas * Y) * np.matrix(kernelTrans(X, x, ('rbf', 1.3))) + b)[0, 0])
    # grid_hat = np.array(grid_hat)
    # grid_hat = grid_hat.reshape(x1.shape)
    # ax.pcolormesh(x1, x2, grid_hat, cmap=plt.cm.coolwarm)
    classifiedPts = {'+1': [], '-1': []}
    for point, label in zip(X, Y):
        classifiedPts['+1'].append(point) if label == 1.0 else classifiedPts['-1'].append(point)
    # 绘制数据点
    for label, pts in classifiedPts.items():
        pts = np.array(pts)
        ax.scatter(pts[:, 0], pts[:, 1], label=label)
    # 绘制支持向量
    svInd = np.nonzero(alphas > 0)[0]
    for sv in X[svInd]:
        plt.scatter(sv[0], sv[1], s=150, c='none', alpha=0.7, linewidth=1.5, edgecolor='#AB3319')
    plt.show()
# 核函数，将数据从低维特征空间映射到高维特征空间，并且直接在低维空间中计算内积隐式的表现出高维空间中的结果
def kernelTrans(X, x, kTup):
    m, n = np.shape(X)
    K = np.mat(np.zeros((m, 1)))  # 计算的核K
    if kTup[0] == 'lin':  # 线性函数，直接进行内积计算即可
        K = X * x.T
    elif kTup[0] == 'rbf':  # 非线性映射，计算高斯核K
        for j in range(m):
            K[j] = np.matrix((X[j, :] - x)) * np.matrix((X[j, :] - x)).T
        K = np.exp(K / (-1 * kTup[1] ** 2))
    return K
# 超平面方程
def fx(os, k):
    ret = np.matrix(os.alphas * os.Y) * os.K[:, k] + os.b
    return ret[0, 0]
# 根据aj的取值范围修剪aj的值
def clipForAj(aj, L, H):
    if aj < L:
        aj = L
    if aj > H:
        aj = H
    return aj
# 计算误差
def calcEk(os, k):
    return fx(os, k) - os.Y[k]
# 更新eCache中的误差
def updateEk(os, k):
    os.eCache[k] = [1, calcEk(os, k)]
# 定义启发式规则来选择第二个拉格朗乘子，选择标准是使得alpha_j有足够大的变化
def selectJK(i, os, Ei):
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    os.eCache[i] = [1, Ei]
    validEcacheList = np.nonzero(os.eCache[:, 0].A)[0]  # 得到有效位为1的误差列表编号
    if len(validEcacheList) > 1:
        for k in validEcacheList:  # 遍历列表找出最大的误差变化
            if k == i:  # 保证第二个alpha不等于第一个alpha
                continue
            Ek = calcEk(os, k)  # 计算第k个误差
            deltaE = abs(Ei - Ek)  # 第k个误差和第i个误差的增量
            if deltaE > maxDeltaE:  # 找到产生误差最大变化的alpha
                maxDeltaE = deltaE
                maxK = k  # 记录最大的步长
                Ej = Ek  # 记录最大误差值
        return maxK, Ej
    else:  # 如果找不到，就随机选择一个alpha
        l = list(range(os.m))
        j = np.random.choice(l[: i] + l[i + 1:])
        Ej = calcEk(os, j)
    return j, Ej
# 内循环，采用启发式选择两个alpha因子
def inner(i, os):
    # 步骤1：计算第一个因子的误差
    Ei = calcEk(os, i)
    # 第一个alpha因子的启发式选择方法，是不满足KKT条件
    if ((os.Y[i] * Ei < -os.tol) and (os.alphas[i] < os.C)) or ((os.Y[i] * Ei > os.tol) and (os.alphas[i] > 0)):
        j, Ej = selectJK(i, os, Ei)  # 第二个alpha因子的启发式选择方法，是寻找max|Ei-Ej|
        a_i_old, a_j_old = os.alphas[i].copy(), os.alphas[j].copy()
        # 步骤2：计算a_j的上界H和下界L
        if os.Y[i] != os.Y[j]:
            L = max(0, os.alphas[j] - os.alphas[i])
            H = min(os.C, os.C + os.alphas[j] - os.alphas[i])
        else:
            L = max(0, os.alphas[j] + os.alphas[i] - os.C)
            H = min(os.C, os.alphas[j] + os.alphas[i])
        if L == H:
            print('L==H')
            return 0
        # 步骤3：根据多元函数推导出的一元函数后的系数eta，计算学习率eta
        K_ii, K_jj, K_ij = os.K[i, i], os.K[j, j], os.K[i, j]  # 计算各个向量的点积
        eta = K_ii + K_jj - 2 * K_ij
        if eta <= 0:
            print('eta <= 0')
            return 0
        # 步骤4：根据SMO算法推理出的一元函数的迭代公式，更新a_j（最复杂的推理过程）
        os.alphas[j] = a_j_old + os.Y[j] * (Ei - Ej) / eta
        # 步骤5：根据a_j的取值范围修剪更新后的a_j
        os.alphas[j] = clipForAj(aj=os.alphas[j], L=L, H=H)
        updateEk(os, j)  # 更新第j个误差缓存
        if abs(os.alphas[j] - a_j_old) < 0.00001:
            print('alpha_j变化太小')
            return 0
        # 步骤6：根据约束条件推导出的迭代公式，更新a_i
        os.alphas[i] = a_i_old + os.Y[i] * os.Y[j] * (a_j_old - os.alphas[j])
        updateEk(os, i)  # 更新第i个误差缓存
        # 步骤7：根据支持向量点方程推理出的公式，更新b_i和b_j
        bi = os.b - Ei - os.Y[i] * K_ii * (os.alphas[i] - a_i_old) - os.Y[j] * K_ij * (os.alphas[j] - a_j_old)
        bj = os.b - Ej - os.Y[i] * K_ij * (os.alphas[i] - a_i_old) - os.Y[j] * K_jj * (os.alphas[j] - a_j_old)
        # 步骤8：根据b_i和b_j来更新b
        if 0 < os.alphas[i] < os.C:
            os.b = bi
        elif 0 < os.alphas[j] < os.C:
            os.b = bj
        else:
            os.b = (bi + bj) / 2.0
        return 1
    else:
        return 0
# 外层循环，优先顺序遍历间隔边界上的支持向量点，若无法优化则遍历整个数据集
def outter(X, Y, C, toler, maxIter, kTup = ('lin',0)):
    os = optStructK(X=np.array(X), Y=np.array(Y), C=C, toler=toler, kTup=kTup)
    alphaPairsChanged, iterNum = 0, 0
    entireSet = True
    while (iterNum < maxIter) and ((alphaPairsChanged > 0) or (entireSet is True)):
        alphaPairsChanged = 0
        if entireSet is True:
            # 遍历所有值
            for i in range(os.m):
                alphaPairsChanged += inner(i, os)
                print('遍历所有值，经过第{0}次迭代，第{1}个因子更新了{2}次'.format(iterNum, i, alphaPairsChanged))
            iterNum += 1
        else:
            # 遍历间隔边界上的支持向量点
            nonBoundIs = np.nonzero((np.mat(os.alphas).A > 0) * (np.mat(os.alphas).A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += inner(i, os)
                print('遍历间隔边界上的支持向量点，经过第{0}次迭代，第{1}个因子更新了{2}次'.format(iterNum, i, alphaPairsChanged))
            iterNum += 1
        if entireSet is True:
            entireSet = False
        elif alphaPairsChanged == 0:
            entireSet = True
        print('迭代的次数为{0}'.format(iterNum))
    return os.b, os.alphas
# 将模型参数持久化
def saveStore(b, alphas):
    np.savez('./model/model.npz', np.array([b]), alphas)
    print('模型保存完毕...')
# 从模型中读取参数
def laodStore():
    read = np.load('./model/model.npz')
    return read['arr_0'], read['arr_1']
# 测试数据集的错误率
def loss(X, Y, alphas, b):
    X, Y = np.array(X), np.array(Y)
    m = np.shape(X)[0]
    svInd = np.nonzero(alphas > 0)[0]
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(X[svInd], X[i, :], ('rbf', 1.3))  # 核函数计算x[i]内积，隐式计算出w
        predict = np.dot(alphas[svInd] * Y[svInd], kernelEval) + b  # 分类函数预测结果
        if np.sign(predict) != np.sign(Y[i]):
            errorCount += 1
    print('在{}个样本里，预测正确有{}个样本，支持向量个数{}个，样本错误率为{}%'.format(m, m - errorCount, np.shape(X[svInd])[0], round(float(errorCount) / m * 100, 2)))
if __name__ == '__main__':
    X, Y = loadDataSet('./data/dataRBF1.csv')  # 读取数据集
    # b, alphas = outter(X=X, Y=Y, C=200, toler=0.0001, maxIter=10000, kTup=('rbf', 1.3))
    # loss(X=X, Y=Y, alphas=alphas, b=b)
    # saveStore(b=b, alphas=alphas)  # 持久化模型参数
    b, alphas = laodStore()  # 从训练好的模型中取出参数
    loss(X=X, Y=Y, alphas=alphas, b=b)
    showClassifer(X=X, Y=Y, alphas=alphas, b=b)
