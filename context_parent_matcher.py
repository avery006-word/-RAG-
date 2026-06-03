import json
import os
import pickle
import re
import sys


def normalize_text(text):
    text = re.sub(r'<[^>]+>', '', text or '')
    text = text.replace('……', '')
    text = re.sub(r'\s+', '', text)
    return text.lower()


def build_ngrams(text, n=8):
    if len(text) < n:
        return {text} if text else set()
    return {text[i:i + n] for i in range(len(text) - n + 1)}


def score_match(query_grams, parent_grams):
    if not query_grams or not parent_grams:
        return 0
    return len(query_grams & parent_grams)


def split_context(context, item_type):
    if item_type != '多源整合型':
        return [context]
    parts = [part.strip() for part in re.split(r'……+', context or '')]
    return [part for part in parts if len(normalize_text(part)) >= 12] or [context]


def pick_parent_ids(context, item_type, parent_grams_list, max_ids=3):
    picked = []
    for part in split_context(context, item_type):
        query = normalize_text(part)
        query_grams = build_ngrams(query)
        if not query_grams:
            continue

        scored = [
            (score_match(query_grams, grams), idx)
            for idx, grams in enumerate(parent_grams_list)
        ]
        best_score, best_idx = max(scored, default=(0, None))
        # 过滤掉只有零星重合的弱匹配。
        if best_idx is None or best_score / len(query_grams) < 0.18:
            continue
        if best_idx not in picked:
            picked.append(best_idx)
        if len(picked) >= max_ids:
            break

    if picked:
        return picked

    query_grams = build_ngrams(normalize_text(context))
    scored = [
        (score_match(query_grams, grams), idx)
        for idx, grams in enumerate(parent_grams_list)
    ]
    best_score, best_idx = max(scored, default=(0, None))
    if best_idx is not None and query_grams and best_score / len(query_grams) >= 0.18:
        return [best_idx]
    return []

def main(dataset_path=None):
    if dataset_path is None:
        dataset_path = os.path.join('qa_dataset', 'AWSD-Reactive-Burn-Model', 'AWSD数据集JSON.json')

    with open(dataset_path, 'r', encoding='utf-8') as f:
        items = json.load(f)

    kb_dir = os.path.join('knowledge_base', 'AWSD-Reactive-Burn-Model')
    with open(os.path.join(kb_dir, 'parent_data.pkl'), 'rb') as f:
        parents = pickle.load(f)

    parent_texts = list(parents.values()) if isinstance(parents, dict) else list(parents)
    parent_grams_list = [build_ngrams(normalize_text(text)) for text in parent_texts]

    for it in items:
        it['parent_id'] = pick_parent_ids(
            it.get('context', ''),
            it.get('type', ''),
            parent_grams_list,
        )

    with open(dataset_path, 'w', encoding='utf-8') as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else None)