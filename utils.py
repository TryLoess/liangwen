import csv
import os
import time
import inspect
import zipfile

def get_dir_name(parent=False):
    """可以直接获取d当前src的目录，True是src的上一级目录"""
    if parent:
        return os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
    return os.path.dirname(os.path.abspath(__file__))

def print_info(*args, sep=' ', end='\n', file=None, flush=False):
    caller_frame = inspect.currentframe().f_back
    filename = caller_frame.f_code.co_filename
    filename = "\ "[0].join(filename.split(r"\ "[0])[-2:])
    line_no = caller_frame.f_lineno
    prefix = f"[{filename}: {line_no}]: "
    now_time = time.strftime('%m-%d %H:%M', time.localtime())
    message = sep.join(map(str, args))
    print(f"[{now_time}]{prefix} {message}", end=end, file=file, flush=flush)
    
def together_csv(file_path, save_path, condition: int):
    """

    :param file_path: 存放csv的总文件夹
    :param save_path: 路径+名字,推荐使用绝对路径，使用get_dir_name + 文件名
    :param condition: 要求sym等于condition，相当于只合并相应股票的csv.负1会提取所有内容
    """
    if file_path[-1] != "/":
        file_path += "/"

    if condition == -1:
        all_files = [file for file in os.listdir(file_path)]
    else:
        all_files = [file for file in os.listdir(file_path) if file.split("_")[1] == f"sym{condition}"]
    length = len(all_files)
    with open(save_path, 'w', encoding="utf-8") as f_output:
        csv_output = csv.writer(f_output)
        count = 0
        for file in all_files:
            start_time = time.time()
            count += 1
            with open(file_path + file, newline='', encoding="utf-8") as f_input:
                csv_input = csv.reader(f_input)
                header = next(csv_input)
                if f_output.tell() == 0:
                    csv_output.writerow(header)
                for row in csv_input:
                    csv_output.writerow(row)
            delta = time.time() - start_time
            print("当前还剩下: %i待处理\n预计时间需要: %f" % ((length - count), (length - count) * delta))
            print("------------------分割线--------------------")



def zip_files_only(folder_path, output_zip_path):
    with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            if os.path.isfile(item_path):  # 只打包文件
                arcname = os.path.basename(item_path)
                zipf.write(item_path, arcname)
    print(f"打包完成，生成文件：{output_zip_path}")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def conditional_proba_plot(df, feature, target, bins=50):
    """"绘制条件概率图 + 每个分箱的样本数柱状图+条件概率*分箱样本数(加权)图
    
    参数：
    df -- 数据集，包含特征列和目标列
    feature -- 需要分析的特征列名（字符串）
    target -- 目标列名（二分类变量）
    bins -- 分箱数量，默认为50个箱（可根据数据调整）
    """
    df_copy = df.copy()
    # 对特征列进行等频分箱
    
    # 给特征列添加轻微的噪声
    eps = np.random.rand(len(df_copy)) * 1e-8  # 小幅度噪声
    df_copy['feature_with_noise'] = df_copy[feature] + eps

    # 使用噪声数据来分箱
    # 如果不加噪声的话,duplicates='drop'无法做到等频分箱,因为相同数据点会被加到一起
    # 加上噪声后就可以改为duplicates='raise'了
    df['bin'] = pd.qcut(df_copy['feature_with_noise'], q=bins, duplicates='raise')
    # # 对特征列进行等宽分箱
    # df['bin'] = pd.cut(df_copy[feature], bins=bins, duplicates='drop')
    # 等宽分箱导致几乎全集中在最左边效果很差,还是采用等频分箱
    
    # 每个 bin 的正样本概率
    prob_table = df.groupby('bin', observed=True)[target].mean()
    # 每个 bin 的样本数
    count_table = df.groupby('bin', observed=True)[target].count()
    # 每个 bin 的概率 × 样本数（加权指标）
    positive_count = df.groupby('bin', observed=True)[target].sum()

    # 汇总为 DataFrame
    summary = pd.DataFrame({
        f'proba_of_{target}': prob_table,
        'count': count_table,
        f'positive_count': positive_count
    })


    # 创建图形
    fig, axes = plt.subplots(2,1, figsize=(14, 6))
    ax2 = axes[0].twinx()
    x_vals = summary.index.astype(str)

    # 条件概率图（蓝色折线）
    sns.lineplot(x=x_vals, y=summary[f'proba_of_{target}'], color='blue', marker='o', label='proba', ax=axes[0])
    # 柱状图：样本数（灰）
    sns.barplot(x=x_vals, y=summary['count'], color='gray', alpha=0.3, ax=ax2, label='count')

    # 总体is_amount_0概率
    p = df[target].mean()

    # 添加水平虚线（is_amount_0的比例）
    axes[0].axhline(p, color='red', linestyle='--', label=f'P({target}) = {p:.4f}')

    axes[0].set_ylabel(f'proba_of_{target}', color='blue')
    ax2.set_ylabel('count', color='gray')
    axes[0].set_xlabel(f'{feature}_bins', fontsize=12)
    axes[0].tick_params(axis='x', rotation=90)
    axes[0].legend(loc='upper left')
    
    # 副图:proba*count
    sns.lineplot(x=x_vals, y=summary['positive_count'], color='purple', marker='o', ax=axes[1])
    sns.barplot(x=x_vals, y=summary['positive_count'], color='yellow', alpha=0.3, ax=axes[1])
    axes[1].set_ylabel('positive_count (count * proba)', color='purple')
    axes[1].set_xlabel('')
    axes[1].set_xticks([])
    # 总标题和布局
    fig.suptitle(f'{feature} vs {target} (Proba, Count, Count × Proba)', fontsize=16)

    # 自动调整布局
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)  # 为标题留空间
    plt.show()




def proba_plot_by_sym(df, feature, target, bins=50):
    df = df.copy()
    fig, axes = plt.subplots(5, 2, figsize=(18, 20), sharex=False, sharey=False)
    
    # 也可以用df['sym'].unique()
    for i in range(5):
        df_sym = df[df['sym'] == i].copy()
        
        # 给特征列添加轻微的噪声
        eps = np.random.rand(len(df_sym)) * 1e-8  # 小幅度噪声
        df_sym['feature_with_noise'] = df_sym[feature] + eps
        df_sym['bin'] = pd.qcut(df_sym['feature_with_noise'], q=bins, duplicates='raise')

        prob = df_sym.groupby('bin', observed=True)[target].mean()
        count = df_sym.groupby('bin', observed=True)[target].count()
        positive = df_sym.groupby('bin', observed=True)[target].sum()

        summary = pd.DataFrame({
            'proba': prob,
            'count': count,
            'positive_count': positive
        })
        x_vals = summary.index.astype(str)
        
        ax1 = axes[i, 0]
        ax2 = ax1.twinx()

        sns.lineplot(x=x_vals, y=summary['proba'], color='blue', marker='o', label='proba', ax=ax1)
        sns.barplot(x=x_vals, y=summary['count'], color='gray', alpha=0.3, ax=ax2)
        
        # 总体is_amount_0概率
        p = df[target].mean()
        
        # 获取每个分箱的边界
        bin_edges = [interval.left for interval in df['bin'].cat.categories]


        # 添加水平虚线（is_amount_0的比例）
        ax1.axhline(p, color='red', linestyle='--', label=f'P({target}) = {p:.4f}')
        
        ax1.set_ylabel('proba', color='blue')
        ax2.set_ylabel('count', color='gray')
        ax1.set_title(f'sym = {i} - {feature} vs proba & count')
        ax1.tick_params(axis='x', rotation=90)

        ax3 = axes[i, 1]
        sns.lineplot(x=x_vals, y=summary['positive_count'], color='purple', marker='o', ax=ax3)
        sns.barplot(x=x_vals, y=summary['positive_count'], color='yellow', alpha=0.3, ax=ax3)

        ax3.set_ylabel('positive_count', color='purple')
        ax3.set_title(f'sym = {i} - positive_count')
        ax3.set_xticks([])  # 不显示副图的x轴刻度

    plt.tight_layout()
    plt.show()

def compute_trend_tstat(group, col, window=20, long=True, epsilon=1e-8):
    """
    计算滑动窗口的趋势指标（t-statistic）：均值/标准差，乘以sqrt(window)以强化长期效应
    参数：
        group (DataFrame): 一段时间序列数据，需包含 col 列和 'time'
        col (str): 要计算差分的列名，如 'mid_diff'
        window (int): 滑动窗口长度（tick 数）
        long (bool): 是否乘以 sqrt(window) 来放大长期效应
        min_periods (int): rolling 最少窗口数，默认 window//2
        epsilon (float): 防止除以零的小常数
    返回：
        DataFrame: 增加一列 'trend_tstat_{window}'
    """
    # 排序 & 计算差分
    group = group.sort_values('time')
    group[f'{col}_diff'] = group[col].diff()

    # 滑动窗口均值与标准差
    group[f'rolling_{window}_{col}_diff_sum'] = group[f'{col}_diff'].rolling(window=window).sum()
    group[f'rolling_{window}_{col}_diff_std']  = group[f'{col}_diff'].rolling(window=window).std()

    # 计算 t-statistic
    t_stat = (group[f'rolling_{window}_{col}_diff_sum'] / (group[f'rolling_{window}_{col}_diff_std'] + epsilon)) / window

    # 放大长期效应
    if long:
        t_stat = t_stat * np.sqrt(window)

    group[f'trend_tstat_{window}'] = t_stat
    return group

