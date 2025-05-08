import csv
import inspect
import os
import time

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


