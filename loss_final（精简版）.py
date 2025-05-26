import numpy as np

def softmax(logits):
    """把lgbm原始得分 logits 转换为 softmax 概率分布（概率和为1）"""
    e = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)

def soft_fbeta_loss_multiclass(beta=0.5, eps=1e-10, num_class=3):
    def loss(y_pred, data):
        """计算 soft Fβ 损失函数的梯度和二阶导数"""
        y_true= data.get_label() # 这里好像是只能传递dataset.get_label()，不能传递y_true这一列
        y_pred = y_pred.reshape(-1, num_class) # 变成 (N, num_class) 的形状
        y_prob = softmax(y_pred)
        grad = np.zeros_like(y_prob)
        hess = np.ones_like(y_prob) * eps  # 避免除零错误
        beta2 = beta ** 2
        
        for c in range(num_class):
            y_true_c = (y_true == c).astype(float)
            y_pred_c = y_prob[:, c]

            soft_TP = np.sum(y_true_c * y_pred_c)
            D1 = np.sum(y_pred_c) + eps
            D2 = np.sum(y_true_c) + eps

            P = soft_TP / D1
            R = soft_TP / D2

            dTP = y_true_c
            dP = (dTP * D1 - soft_TP) / (D1 ** 2)
            dR = dTP / D2
            d2P = -2 * dP / D1
            d2R = 0

            N = (1 + beta2) * (dP * R + P * dR) * (beta2 * P + R + eps) - (1 + beta2) * P * R * (beta2 * dP + dR)
            D = (beta2 * P + R + eps) ** 2

            dN = (1 + beta2) * (
                (d2P * R + 2 * dP * dR) * (beta2 * P + R + eps)
                - P * R * beta2 * d2P
            )
            dD = 2 * (beta2 * P + R + eps) * (beta2 * dP + dR)

            grad[:, c] = -N / D
            hess[:, c] = -(dN * D - N * dD) / (D ** 2)

        return grad.flatten(), hess.flatten()

    return loss