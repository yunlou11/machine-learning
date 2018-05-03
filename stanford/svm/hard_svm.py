# -- coding: utf-8 --
import numpy as np
import matplotlib.pyplot as plt


def load_data(path):
    data = np.loadtxt(path, dtype=float)
    return data


def draw_data_scatter(data):
    sample_size = np.shape(data)[0]
    for i in range(sample_size):
        category = int(data[i, 2])
        color = 'or' if category > 0 else 'og'
        plt.plot(data[i, 0], data[i, 1], color)


def draw_split_line(w, b):
    """
    绘制分界线
    :param w:
    :param b:
    :return:
    """
    x1 = np.linspace(0, 5, 10)
    x2 = (-b - w[0] * x1) / w[1]
    plt.plot(x1, x2)


def draw_vectors(alpha, x):
    """
    绘制支持向量点, alpha不为0的点为支持向量点
    :param alpha:
    :param x:
    :return:
    """
    sample_size = np.shape(alpha)[0]
    for i in range(sample_size):
        if alpha[i] > 0:
            plt.plot(x[i][0], x[i][1], 'x', markersize=10)


def get_w(alpha, y, x):
    """
    wi = sum(alpha_i * yi * xi)
    :param alpha:
    :param y:
    :param x:
    :return:
    """
    sample_size, figure_size = np.shape(x)
    w = np.zeros(figure_size)
    for i in range(sample_size):
        w += alpha[i] * y[i] * x[i]
    return w


def gx(alpha, y, kernel, i, b):
    """
    计算i样本到分界函数距离 gx = w.T * xi +b
    函数间隔 = yi * gx, 即函数间隔 = |gx|
    :param alpha:
    :param y:
    :param kernel: 核函数,本例中为 xi * xj
    :param i:
    :param b:
    :return:
    """
    return round(sum(alpha * y * kernel[i]) + b, 12)


def check_kkt(alpha, y, kernel, i, b):
    """
    检测i样本是否满足KKT条件, 即:
        (1) alpha_i > 0时, 函数间隔 = 1
        (2) alpha_i = 0时, 函数间隔 >= 1
    counter = |函数间隔-1| , 误划分的偏离程度
    返回是否满足KKT条件和counter
    :param alpha:
    :param y:
    :param kernel:
    :param i:
    :param b:
    :return:
    """
    counter = 0
    satisfied = True
    function_interval = y[i] * gx(alpha, y, kernel, i, b)
    if alpha[i] == 0:
        satisfied = function_interval >= 1
    elif alpha[i] > 0:
        satisfied = function_interval == 1
    if not satisfied:
        counter = abs(function_interval - 1)
    return satisfied, counter


def get_alpha_1(alpha, y, kernel, b):
    """
    获取外层循环的alpha_1,规则是:
        (1) 首先检测所有 alpha_i>0,即支持向量是否满足KKT条件
        (2) 如果支持向量都满足KKT,再检测剩余所有训练集
    获取违反KKT条件最严重的点
    该检验是在ε精度范围内进行的, 例如:
        alpha_i>0, 函数间隔=0.999 , 此时精度=0.01, 则可以认为满足KKT条件
    真实训练时,想要完全满足KKT条件还是很困难的,本例中训练数据较少,精度 1e-12 (主要float和1.0相等的比较,不设置精度,很难完全相等)
    :param alpha:
    :param y:
    :param kernel:
    :param b:
    :return:
    """
    sample_size = np.shape(y)[0]
    for j in range(2):
        satisfied = True
        max_counter = -1
        counter = -1
        alpha_i = -1
        for i in range(sample_size):
            tmp_satisfied = True
            if j == 0 and alpha[i] > 0:
                tmp_satisfied, counter = check_kkt(alpha, y, kernel, i, b)
            elif j == 1 and alpha[i] == 0:
                tmp_satisfied, counter = check_kkt(alpha, y, kernel, i, b)
            if not tmp_satisfied and counter > max_counter:
                alpha_i = i
                max_counter = counter
                satisfied = tmp_satisfied
        if not satisfied:
            break
    return alpha_i


def abs_max_index(data):
    """
    获取数组中,绝对值最大的下标索引
    :param data:
    :return:
    """
    size = np.size(data)
    max_abs_value = abs(data[0])
    max_index = 0
    for i in range(size):
        c_value = abs(data[i])
        if c_value > max_abs_value:
            max_abs_value = c_value
            max_index = i
    return max_index


def get_alpha_2_from_error(error_1, alpha, y, kernel, b):
    """
    alpha_2_new_unc = alpha_2_old + y2(E1 - E2) / (K11+K22-2K12)
    K11+K22-2k12 不变, 则alpha_2的变化量由 E1-E2决定,选择E2尽量使 |E1-E2|更大
    E2 = (w.T * x2 +b) - y2  = 预测值 - 实际值
    :param error_1:
    :param alpha:
    :param y:
    :param kernel:
    :param b:
    :return:
    """
    errors = []
    sample_size = np.shape(y)[0]
    for i in range(sample_size):
        e = gx(alpha, y, kernel, i, b) - y[i]
        errors.append(e)
    errors_1_2 = error_1 - np.array(errors)
    alpha_2 = abs_max_index(errors_1_2)
    return alpha_2, errors[alpha_2]


def get_alpha_2_from_all(kernel, alpha, y, b, alpha_1, error_1):
    """
    遍历所有训练集,获取使 alpha_1变化最大的样本点, 这里应该是使目标函数有足够下降的样本点,偷懒选择使alpha_1变化最大点
    :param kernel:
    :param alpha:
    :param y:
    :param b:
    :param alpha_1:
    :param error_1:
    :return:
    """
    sample_size = np.shape(alpha)[0]
    max_change = 0
    alpha_2 = -1
    error_2 = 0
    for i in range(sample_size):
        if i != alpha_1:
            tmp_error_2 = gx(alpha, y, kernel, i, b) - y[i]
            alpha_1_new = get_new_alpha_b(kernel, alpha, y, b, alpha_1, i, error_1, tmp_error_2)[0]
            change = abs(alpha_1_new - alpha[alpha_1])
            if change > max_change:
                alpha_2 = i
                error_2 = tmp_error_2
    return alpha_2, error_2


def get_new_alpha_b(kernel, alpha, y, b, alpha_1, alpha_2, error_1, error_2):
    """
    计算alpha_new 和 b_new
    :param kernel:
    :param alpha:
    :param y:
    :param b:
    :param alpha_1:
    :param alpha_2:
    :param error_1:
    :param error_2:
    :return:
    """
    # 简化计算所需参数名称
    k11 = kernel[alpha_1, alpha_1]
    k22 = kernel[alpha_2, alpha_2]
    k12 = kernel[alpha_1, alpha_2]
    y1 = y[alpha_1]
    y2 = y[alpha_2]
    alpha_1_old = alpha[alpha_1]
    alpha_2_old = alpha[alpha_2]
    # 计算 η
    eta = k11 + k22 - 2 * k12
    alpha_2_new_unc = alpha_2_old + (y2 * (error_1 - error_2)) / eta
    alpha_2_new = get_alpha_2_new(alpha_2_new_unc, alpha_1, alpha_2, y, alpha)
    alpha_1_new = alpha_1_old + y1 * y2 * (alpha_2_old - alpha_2_new)
    # 计算 b_new
    b_new_1 = -error_1 - y1 * k11 * (alpha_1_new - alpha_1_old) - y2 * k12 * (alpha_2_new - alpha_2_old) + b
    b_new_2 = -error_2 - y1 * k12 * (alpha_1_new - alpha_1_old) - y2 * k22 * (alpha_2_new - alpha_2_old) + b
    return alpha_1_new, alpha_2_new, b_new_1, b_new_2


def get_alpha_2_new(alpha_2_new_unc, alpha_1, alpha_2, y, alpha):
    """
    根据 alpha_2_new_unc获取 alpha_2_new, 约束条件
        (1) alpha_i >=0  i= 1,2,3...N
        (2) y1*alpha_1 + y2*alpha_2 = K  k常数
    所以 alpha_1 和 alpha_2在 斜率为 1或者 -1的直线上,并且满足>=0
        (1) y1!=y1时, alpha_2 = alpha_1 + k
            此时:alpha_1_old - alpha_2_old = K ,  alpha_2 > max(0, k)
        (2) y1 =y1时, alpha_2 = -alpha_1 + k
            此时:alpha_1_old + alpha_2_old = k,  0=< alpha_2 <= k
    当 alpha_2不满足约束条件时, alpha_2取边界值, 对应的alpha_1一定满足约束条件
    :param alpha_2_new_unc:
    :param alpha_1:
    :param alpha_2:
    :param y:
    :param alpha:
    :return:
    """
    y1 = int(y[alpha_1])
    y2 = int(y[alpha_2])
    low = max(0, alpha[alpha_2] - alpha[alpha_1])
    high = alpha[alpha_2] + alpha[alpha_1]
    if y1 != y2 and alpha_2_new_unc < low:
        alpha_2_new_unc = low
    elif y1 == y2:
        if alpha_2_new_unc > high:
            alpha_2_new_unc = high
        elif alpha_2_new_unc < 0:
            alpha_2_new_unc = 0
    return alpha_2_new_unc


def smo(alpha, y, kernel, b):
    """
    根据SMO序列最小最优化算法求解 SVM的最优解,有3部分偷懒
        (1)已经确定alpha_1的情况下, 寻找alpha_2按照以下规则:
            1)首先寻找使 |E1-E2|最大的点,若此alpha_2无法使目标函数产生足够下降,则下一步
            2)遍历所有支持向量,依次实验,选择使目标函数产生足够下降的样本,找不到则下一步
            3)遍历整个训练集,依次实验每个样本
        本例中如果1)中无法满足, 则直接遍历整个样本数据集
        (2) 如果以上三步都无法获取使目标函数足够下降的alpha_2, 则抛弃alpha_1,重新选择另外一个alpha_1
            本例中没有这一步
        (3) 目标函数有足够的下降 被 alpha_1有足够大的变化代替
    如果只使用|E1-E2|选择alpha_2,则很容易停在一个中间值, 例如选择的alpha_2因为超出边界,则选择边界值0,使
    alpha_2_new = alpha_2_old = 0此时, alpha_1_new = alpha_1_old 无变化, 下次循环依然会选择本次的
    alpha_2和alpha_1,依然无变化,优化停止

    算法停止条件是所有样本在精度内满足KKT条件, 本例中指定了最大循环次数,当无法获得alpha_1时,表明所有样本满足KKT条件
    :param alpha:
    :param y:
    :param kernel:
    :param b:
    :return:
    """
    stop = False
    alpha_1 = get_alpha_1(alpha, y, kernel, b)
    if alpha_1 >= 0:
        error_1 = gx(alpha, y, kernel, alpha_1, b) - y[alpha_1]
        alpha_2, error_2 = get_alpha_2_from_error(error_1, alpha, y, kernel, b)
        alpha_1_new, alpha_2_new, b_new_1, b_new_2 = get_new_alpha_b(kernel, alpha, y, b,
                                                                     alpha_1, alpha_2, error_1, error_2)
        if abs(alpha_1_new - alpha[alpha_1]) < 1e-5:
            alpha_2, error_2 = get_alpha_2_from_all(kernel, alpha, y, b, alpha_1, error_1)
            alpha_1_new, alpha_2_new, b_new_1, b_new_2 = get_new_alpha_b(kernel, alpha, y, b,
                                                                         alpha_1, alpha_2, error_1, error_2)
        if alpha_2 >= 0:
            alpha[alpha_2] = alpha_2_new
            alpha[alpha_1] = alpha_1_new
            if alpha_1_new == 0 and alpha_2_new == 0:
                b = (b_new_1 + b_new_2) / 2
            elif alpha_1_new > 0:
                b = b_new_1
            elif alpha_2_new > 0:
                b = b_new_2
    else:
        stop = True
    return alpha, b, stop


def train(alpha, y, kernel, b, iter_count):
    for i in range(iter_count):
        alpha, b, stop = smo(alpha, y, kernel, b)
        if stop:
            print "iterate count:", i
            break
    return alpha, b


def main():
    plt.figure("mode")
    train_data = load_data("../../doc/hard_svm_data.txt")
    sample_size = np.shape(train_data)[0]
    alpha = np.zeros(sample_size)
    b = 0
    x = train_data[:, 0:2]
    y = train_data[:, 2]
    kernel = np.dot(x, x.T).T
    draw_data_scatter(train_data)
    alpha, b = train(alpha, y, kernel, b, 10)
    w = get_w(alpha, y, x)
    print alpha
    print w, b
    draw_split_line(w, b)
    draw_vectors(alpha, x)
    plt.show()

if __name__ == '__main__':
    main()