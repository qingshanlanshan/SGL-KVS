#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import sys
from collections import defaultdict

def parse_log_line(line):
    """
    解析日志行，提取记录编号和每次迭代时间 (s/it)
    支持例如：
      76%|███████▌| 12182/16000 [4:34:20<1:25:33,  1.34s/it][2025-08-10 11:06:14]
      100%|█████████▉| 4997/5000 [1:21:43<00:03,  1.04s/it][2025-08-14 20:52:43] Prefill ...
    允许在 s/it] 后有追加内容（缺失换行的情况）。
    """

    import re

    # 如果是 Capturing batches 开头的行，直接跳过
    if line.strip().startswith("Capturing batches"):
        return None, None

    # 主匹配：... [ ...,  1.23s/it] 后可跟任意内容
    pat_s_per_it = re.compile(
        r'(?P<pct>\d+)%\|.*?\|\s*(?P<cur>\d+)/\d+\s*\[[^\]]*?,\s*(?P<tpi>\d+(?:\.\d+)?)s/it\].*'
    )
    m = pat_s_per_it.search(line)
    if m:
        return int(m.group('cur')), float(m.group('tpi'))

    # 兜底：如果是 it/s 形式，则换算为 s/it
    pat_it_per_s = re.compile(
        r'(?P<pct>\d+)%\|.*?\|\s*(?P<cur>\d+)/\d+\s*\[[^\]]*?,\s*(?P<itps>\d+(?:\.\d+)?)it/s\].*'
    )
    m2 = pat_it_per_s.search(line)
    if m2:
        itps = float(m2.group('itps'))
        tpi = (1.0 / itps) if itps > 0 else float("inf")
        return int(m2.group('cur')), tpi

    return None, None

def analyze_log_file(filename):
    """
    分析日志文件，支持多个 0-1000 run，每 1000 个记录统计一次
    """
    group_times = defaultdict(list)

    run_id = 0
    last_record = -1

    try:
        with open(filename, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                record_num, time_per_iter = parse_log_line(line.strip())

                if record_num is not None and time_per_iter is not None:
                    # 如果编号比上次小（例如重新回到 0），则认为是新的 run
                    print(record_num)
                    if record_num < last_record:
                        run_id += 1
                    elif record_num == last_record:
                        continue
                    last_record = record_num


                    group_times[run_id].append(time_per_iter)
                    print(group_times[run_id])
                if last_record >= 10:
                    break
    
    
    

    except FileNotFoundError:
        print(f"错误: 找不到文件 '{filename}'")
        return
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return

    if not group_times:
        print("没有找到符合格式的日志记录")
        return

    print("\n" + "=" * 70)
    print("每1000个记录的时间统计 (按 run 分开):")
    print("=" * 70)
    print(f"{'Run':<5} {'Range':<15} {'Avg/s':<10} {'Min/s':<10} {'Max/s':<10} ")
    print("-" * 70)

    for run_id in sorted(group_times.keys()):
        times = group_times[run_id]
        avg_time = sum(times) / len(times)
        print(len(times))
        min_time = min(times)
        max_time = max(times)

        start_record = run_id * 1000
        end_record = start_record + 999
        range_str = f"{start_record}-{end_record}"

        print(f"{run_id:<5} {range_str:<15} {avg_time:<10.3f} {min_time:<10.3f} {max_time:<10.3f} ")


def main():
    """
    主函数
    """
    if len(sys.argv) != 2:
        print("使用方法: python log_analyzer.py <日志文件路径>")
        print("示例: python log_analyzer.py my_log.txt")
        return

    log_file = sys.argv[1]
    print(f"正在分析日志文件: {log_file}")
    print("-" * 40)

    analyze_log_file(log_file)


if __name__ == "__main__":
    main()
