import json
import os
import re

# 输入和输出文件路径
input_json_path = "FINE_TUNE_DATA/wsi_cls_jsons/tissuenet.json"
output_json_path = "FINE_TUNE_DATA/wsi_cls_jsons/tissuenet_updated.json"

# 补丁目录的基础路径
patch_base_dir = "/mnt/sdb/ljw/dataset/TISSUENET/tissue-net_fp"

# 从feat_path中提取slide ID的正则表达式
pattern = r'/(C\d+_B\d+_S\d+(?:_pyr)?)/'

# 加载原始JSON数据
with open(input_json_path, 'r') as f:
    data = json.load(f)

# 处理每个数据集
for dataset_key in data.keys():
    print(f"处理 {dataset_key} 数据集...")
    for i, item in enumerate(data[dataset_key]):
        # 提取slide ID
        match = re.search(pattern, item["feat_path"])
        if match:
            slide_id = match.group(1)
            
            # 检查对应的patch目录是否存在
            patch_dir = os.path.join(patch_base_dir, slide_id)
            if os.path.exists(patch_dir):
                item["patch_dir"] = patch_dir
            else:
                item["patch_dir"] = "NULL"
        else:
            item["patch_dir"] = "NULL"
        
        # 每处理100个项目打印一次进度
        if (i + 1) % 100 == 0:
            print(f"已处理 {i + 1}/{len(data[dataset_key])} 个项目")

# 保存更新后的JSON数据
with open(output_json_path, 'w') as f:
    json.dump(data, f, indent=4)

print(f"处理完成，更新后的JSON已保存到 {output_json_path}") 