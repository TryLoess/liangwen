import pandas as pd
from src.test_utils import *
from need_zip_dir.Predictor import *
import json


base_dir = get_dir_name(True)
with open(base_dir + "/need_zip_dir/config.json", "r") as f:
    config = json.load(f)
feature = config["feature"]
data = pd.read_csv(base_dir + "/feature_set/no_feature_sym0.csv")[feature]
print("初始化完成")
pre = Predictor()
pre.predict([data.iloc[:100], data.iloc[100:200]])