import json
import os
from collections import defaultdict

def evaluate_retrieval(dataset_path: str, result_path: str, output_path: str) -> None:
    """
    评估检索结果：计算每个问题的召回率、倒数排名，并按类型分组统计。

    Args:
        dataset_path: 原始数据集文件路径（包含 parent_id, type）
        result_path: 提取结果文件路径（包含 retrieval_parent_id 和 accuracy）
        output_path: 输出评估结果的 JSON 文件路径
    """
    dataset_path="AWSD数据集JSON.json"
    result_path = "提取结果.json"
    output_path = "evaluation_result"
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    with open(result_path, 'r', encoding='utf-8') as f:
        results = json.load(f)

    assert len(dataset) == len(results), \
        f"数据集与提取结果条目数不一致: {len(dataset)} vs {len(results)}"

    evaluation = []
    total_questions = len(dataset)

    # 用于累计统计的值（总体）
    sum_recall = 0.0
    sum_accuracy = 0.0
    sum_reciprocal_rank = 0.0

    # 按类型分组统计的累加器
    type_stats = defaultdict(lambda: {"sum_recall": 0.0, "sum_accuracy": 0.0, "sum_rr": 0.0, "count": 0})

    for idx, (data_item, res_item) in enumerate(zip(dataset, results), start=1):
        q_type = data_item.get('type', '未知类型')
        parent_ids = data_item.get('parent_id', [])
        retrieval_ids = res_item.get('retrieval_parent_id', [])
        accuracy = res_item.get('accuracy', 0)

        # 1. 计算召回率 = 检索结果中命中的正确ID数量 / 全集正确ID数量
        parent_set = set(parent_ids)
        hit_count = sum(1 for pid in retrieval_ids if pid in parent_set)
        recall = hit_count / len(parent_ids) if parent_ids else 0.0

        # 2. 计算倒数排名（首个相关结果的位置）
        rank = 0
        for r_idx, pid in enumerate(retrieval_ids, start=1):
            if pid in parent_set:
                rank = r_idx
                break
        reciprocal_rank = 1.0 / rank if rank > 0 else 0.0

        # 累计统计（总体）
        sum_recall += recall
        sum_accuracy += accuracy
        sum_reciprocal_rank += reciprocal_rank

        # 按类型累计
        type_stats[q_type]["sum_recall"] += recall
        type_stats[q_type]["sum_accuracy"] += accuracy
        type_stats[q_type]["sum_rr"] += reciprocal_rank
        type_stats[q_type]["count"] += 1

        # 构建输出条目
        entry = {
            "question_number": idx,
            "type": q_type,
            "parent_id": parent_ids,
            "retrieval_parent_id": retrieval_ids,
            "recall": recall,
            "reciprocal_rank": reciprocal_rank,
            "accuracy": accuracy
        }
        evaluation.append(entry)

    # 计算总体平均值
    mean_recall = sum_recall / total_questions if total_questions > 0 else 0.0
    mean_accuracy = sum_accuracy / total_questions if total_questions > 0 else 0.0
    mrr = sum_reciprocal_rank / total_questions if total_questions > 0 else 0.0

    # 计算各类型的平均值
    type_summary = {}
    for q_type, stats in type_stats.items():
        cnt = stats["count"]
        type_summary[q_type] = {
            "count": cnt,
            "mean_recall": stats["sum_recall"] / cnt if cnt > 0 else 0.0,
            "mean_accuracy": stats["sum_accuracy"] / cnt if cnt > 0 else 0.0,
            "mrr": stats["sum_rr"] / cnt if cnt > 0 else 0.0
        }

    # 输出详细结果到 JSON 文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(evaluation, f, ensure_ascii=False, indent=2)

    # 打印统计信息
    print(f"评估完成！共 {total_questions} 个问题。")
    print("\n=== 总体指标 ===")
    print(f"平均召回率 (Mean Recall): {mean_recall:.4f}")
    print(f"平均准确率 (Mean Accuracy): {mean_accuracy:.4f}")
    print(f"平均倒数排名 (MRR): {mrr:.4f}")

    print("\n=== 按类型分类指标 ===")
    # 定义五种类型的顺序
    type_order = ["单事实检索型", "表格问答型", "公式问答型", "图文关联型", "多源整合型"]
    for q_type in type_order:
        if q_type in type_summary:
            stats = type_summary[q_type]
            print(f"\n【{q_type}】（数量：{stats['count']}）")
            print(f"  平均召回率: {stats['mean_recall']:.4f}")
            print(f"  平均准确率: {stats['mean_accuracy']:.4f}")
            print(f"  MRR: {stats['mrr']:.4f}")

    print(f"\n结果详情已保存至: {output_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_file = os.path.join(script_dir, "AWSD数据集JSON.json")
    result_file = os.path.join(script_dir, "提取结果.json")
    output_file = os.path.join(script_dir, "evaluation_results.json")

    evaluate_retrieval(dataset_file, result_file, output_file)