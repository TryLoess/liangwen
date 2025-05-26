import numpy as np

def softmax(logits):
    """把lgbm原始得分 logits 转换为 softmax 概率分布（概率和为1）"""
    e = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    # keepdims=True 让结果保持原来的维度结构，方便后续矩阵运算和广播。
    return e / np.sum(e, axis=1, keepdims=True)

def soft_fbeta_loss_multiclass(beta=0.5, eps=1e-10, num_class=3):
    def loss(y_pred, y_val):
        # 这里的y_pred是加了raw_score=True后的LightGBM的原始得分 logits
        # 可能输出如下：
        # [[-1.2,  2.3,  0.5],   # 第1个样本，类别0/1/2的原始分数
        #  [ 1.1, -0.8,  0.2],   # 第2个样本，类别0/1/2的原始分数
        # 需要检查一下是不是这样，是的话形状如下
        # y_val.shape == (n,)
        # y_pred.shape == (n, 3)
        y_prob = softmax(y_pred) # 将 logits 转换为概率分布

        # 初始化梯度和二阶导数数组
        grad = np.zeros_like(y_prob)
        hess = np.ones_like(y_prob) * 1e-8  # 避免除零错误

        beta2 = beta ** 2

        # Fβ不可导，所以用可导的 soft Fβ 来代替 Fβ
        # 这里的 soft Fβ 是指在计算 Fβ 时，使用了平滑的方式来处理精确率和召回率

        for c in range(num_class):
            y_val_c = (y_val == c).astype(float)   # 判断每个样本是否属于当前类别
            y_pred_c = y_prob[:, c]                # 取出所有样本对当前类别 c 的预测概率

            # soft TP, FP, FN
            soft_TP = np.sum(y_val_c * y_pred_c)
            soft_FP = np.sum((1 - y_val_c) * y_pred_c)
            soft_FN = np.sum(y_val_c * (1 - y_pred_c))
            
            # 易知soft_TP + soft_FP = np.sum(y_pred_c)
            # 易知soft_TP + soft_FN = np.sum(y_val_c)
            D1 = np.sum(y_pred_c) + eps
            D2 = np.sum(y_val_c) + eps

            # 对比一下          
            # P（Precision） = TP / (TP + FP)
            # R（Recall） = TP / (TP + FN)

            # Soft Precision and Recall
            soft_Precision = soft_TP / D1
            soft_Recall = soft_TP / D2

            # Soft Fβ
            soft_Fβ = (1 + beta2) * soft_Precision * soft_Recall / (beta2 * soft_Precision + soft_Recall + eps)

            # 计算梯度grad
            dTP = y_val_c
            dFP = (1 - y_val_c)
            dFN = -y_val_c
            # dP = (dTP * (soft_TP + soft_FP + eps) - soft_TP * (dTP + dFP)) / ((soft_TP + soft_FP + eps) ** 2)
            # dR = (dTP * (soft_TP + soft_FN + eps) - soft_TP * (dTP + dFN)) / ((soft_TP + soft_FN + eps) ** 2)
            # 简化后如下：
            P = soft_Precision
            R = soft_Recall
            dP = (dTP * D1 - soft_TP) / (D1 ** 2)
            # dR = (dTP * D2 - soft_TP * (dTP + dFN)) / (D2 ** 2)
            dR = dTP / D2
            dF = (1 + beta2) * (dP * R + P * dR) * (beta2 * P + R + eps) - (1 + beta2) * P * R * (beta2 * dP + dR)
            dF /= (beta2 * P + R + eps) ** 2

            # LightGBM 只能“最小化”损失，而最大化 Fβ 就等价于最小化 -Fβ。
            # 所以这里要给梯度取负号
            grad[:, c] = -dF 
            
            # 接下来计算二阶导数 hess
            # 令 df=N/D 简化运算
            N = (1 + beta2) * (dP * R + P * dR) * (beta2 * P + R + eps) - (1 + beta2) * P * R * (beta2 * dP + dR)
            D = (beta2 * P + R + eps) ** 2
            
            # 二阶导数
            # dD1=1
            # dD2=0
            # d2P = (-2 * dTP * D1 + 2 * soft_TP) / (D1 ** 3)
            d2P = -2 * dP/ D1
            d2R = 0
            
            # dN = N对y_pred_c的导数，按乘积和链式法则展开
            # dN = (1 + beta2) * (
            #     (d2P * R + 2 * dP * dR + P * d2R) * (beta2 * P + R + eps)
            #     + (dP * R + P * dR) * (beta2 * dP + dR)
            #     - (dP * R + P * dR) * (beta2 * dP + dR)
            #     - P * R * (beta2 * d2P + d2R)
            # )
            dN = (1 + beta2) * (
                (d2P * R + 2 * dP * dR) * (beta2 * P + R + eps)
                - P * R * beta2 * d2P
            )
            dD = 2 * (beta2 * P + R + eps) * (beta2 * dP + dR)
            
            hess[:, c] = -(dN * D - N * dD) / (D ** 2)  # 取负号，保持与梯度方向一致

        return grad.flatten(), hess.flatten()

    return loss
