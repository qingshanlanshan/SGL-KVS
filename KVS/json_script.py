import json

# 读取JSON文件
def analyze_instructions(json_file_path):
    # 读取JSON文件
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    instructions = data['instructions']
    
    # 统计总数
    total_count = len(instructions)
    print(f"总共有 {total_count} 条instructions")
    
    # 计算每条instruction的单词数量
    word_counts = []
    for instruction in instructions:
        word_count = len(instruction.split())
        word_counts.append(word_count)
    
    # 排序获取前10%最长的instructions
    word_counts.sort(reverse=True)  # 降序排列
    top_10_percent_count = max(1, int(total_count * 0.1))  # 至少取1个
    top_10_percent_lengths = word_counts[:top_10_percent_count]
    
    # 计算前10%的平均长度
    avg_length = sum(top_10_percent_lengths) / len(top_10_percent_lengths)
    
    print(f"前10%最长的instructions（共{top_10_percent_count}条）的平均长度: {avg_length:.2f} 个单词")
    
    # 显示一些额外统计信息
    print(f"最长instruction: {max(word_counts)} 个单词")
    print(f"最短instruction: {min(word_counts)} 个单词")
    print(f"所有instructions的平均长度: {sum(word_counts)/len(word_counts):.2f} 个单词")

# 使用示例
if __name__ == "__main__":
    # 修改这里的文件路径为你的JSON文件路径
    json_file_path = "/home/mengke/code/data_preprocess/final_data.json"
    
    try:
        analyze_instructions(json_file_path)
    except FileNotFoundError:
        print(f"文件 {json_file_path} 不存在，请检查文件路径")
    except KeyError:
        print("JSON文件格式不正确，请确保包含'instructions'字段")
    except Exception as e:
        print(f"发生错误: {e}")