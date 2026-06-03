import re
import ast
import json


def extract_data_from_txt(txt_path):
    """
    从本地answer.txt中提取指定字段，返回JSON格式数据
    :param txt_path: TXT文件的本地路径
    :return: 格式化的JSON字符串
    """
    # 初始化存储结果的列表
    result = []
    # 初始化单条记录的临时字典
    record = {}

    # 正则表达式匹配 耗时、数字、字典行
    time_pattern = re.compile(r'(\d+\.\d+)')  # 匹配小数
    dict_pattern = re.compile(r"^\{'retrieval_parent_id':")  # 匹配数据字典行
    accuracy_pattern = re.compile(r'^[01]$')  # 匹配单独的0/1

    # 读取本地TXT文件（utf-8编码，兼容中文）
    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 逐行解析
    for line in lines:
        line = line.strip()  # 去除首尾空格/换行
        if not line:
            continue

        # 1. 提取 检索耗时
        if "检索耗时" in line:
            time = float(time_pattern.search(line).group(1))
            record["检索耗时"] = time

        # 2. 提取 retrieval_parent_id 和 answer（字典行）
        elif dict_pattern.match(line):
            # 安全解析字典字符串
            data_dict = ast.literal_eval(line)
            record["retrieval_parent_id"] = data_dict["retrieval_parent_id"]
            record["answer"] = data_dict["answer"]

        # 3. 提取 生成回答耗时
        elif "生成回答耗时" in line:
            time = float(time_pattern.search(line).group(1))
            record["生成回答耗时"] = time

        # 4. 提取 accuracy（0/1）
        elif accuracy_pattern.match(line):
            record["accuracy"] = int(line)
            # 单条记录完成，加入结果集，清空临时字典
            result.append(record.copy())
            record.clear()

    # 转为格式化JSON字符串（ensure_ascii=False 保留中文，indent=4 格式化）
    json_result = json.dumps(result, ensure_ascii=False, indent=4)
    return json_result


# ------------------- 只需修改这里的TXT文件路径 -------------------
TXT_FILE_PATH = "AWSD数据集answer.txt"  # 替换为你的文件绝对/相对路径
# ----------------------------------------------------------------

# 执行提取并输出JSON
if __name__ == "__main__":
    json_data = extract_data_from_txt(TXT_FILE_PATH)
    print("提取完成，JSON格式结果：\n")
    print(json_data)

    # 可选：将结果保存为本地JSON文件
    with open("提取结果.json", "w", encoding="utf-8") as f:
        f.write(json_data)
    print("\n✅ 结果已保存为：提取结果.json")