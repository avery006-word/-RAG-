# test6plus2.py
# 基于 test6plus.py 增强：多查询重写（Multi-Query Rewrite）
# 所有问题统一扩展为 3 条子查询后分别召回，汇总去重子块后用"查询1"统一 Cross-Encoder 重排保留 Top3，其余逻辑保持一致

import json
import re
import pickle
import faiss
import time
import numpy as np
import jieba
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from qwen_agent.llm import get_chat_model

FAISS_INDEX_PATH = "knowledge_base/AWSD-Reactive-Burn-Model/faiss_parent_child_index.bin"
CHILD_DATA_PATH = "knowledge_base/AWSD-Reactive-Burn-Model/child_data.pkl"
PARENT_DATA_PATH = "knowledge_base/AWSD-Reactive-Burn-Model/parent_data.pkl"

print("加载知识库...")
index = faiss.read_index(FAISS_INDEX_PATH)
with open(CHILD_DATA_PATH, 'rb') as f:
    child_data = pickle.load(f)
with open(PARENT_DATA_PATH, 'rb') as f:
    parent_chunks = pickle.load(f)

child_chunks = child_data['chunks']
child_to_parent = child_data['parent_ids']

def get_parent_ids_by_child_idx(idx: int):
    p = child_to_parent[idx]
    if isinstance(p, list):
        return p
    return [p]

BM25_RECALL_K = 6
VECTOR_RECALL_K = 6
RRF_K = 50
W_BM25 = 0.4
W_VECTOR = 0.6
RERANK_TOP_CHILD_N = 3
RERANKER_TOP_CANDIDATES = 10

EMBED_MODEL_PATH = r"D:\anaconda\conda_envs\qwen-agent\models\Qwen3-Embedding-0.6B"
print("加载 Embedding 模型...")
embed_model = SentenceTransformer(EMBED_MODEL_PATH, device="cpu")

def encode_query(query):
    emb = embed_model.encode([query], prompt_name="query", normalize_embeddings=True)
    return emb.astype('float32')

print("构建 BM25 索引（child 粒度）...")
child_tokenized = [jieba.lcut(t) for t in child_chunks]
bm25 = BM25Okapi(child_tokenized)

RERANKER_MODEL_PATH = r"D:\anaconda\conda_envs\qwen-agent\models\bge-reranker-v2-m3"
print("加载 Cross-Encoder reranker...")
cross_encoder = CrossEncoder(RERANKER_MODEL_PATH, device="cpu")

def simple_progress_bar(current, total, prefix=""):
    bar_len = 30
    filled_len = int(round(bar_len * current / float(total)))
    bar = '█' * filled_len + '-' * (bar_len - filled_len)
    print(f"\r{prefix}[{bar}] {current}/{total}", end="", flush=True)
    if current == total:
        print()

# ========== 多查询重写相关 ==========

# 查询扩展提示词：原问题作首条，优先问题拆解、其次同义词扩展，补充2条，共3条
REWRITE_PROMPT = """你是兵器科技文献检索查询生成器，仅输出符合规则的JSON，禁止任何解释、序号、额外内容。

【核心规则】
1. 条目规范：单条为学术文献检索规范表述，以规范术语为主；禁用口语、原问题未提及的额外对象/场景/限定；条目间语义独立无重复。
2. 必须将用户原问题原样作为首条查询，接着判断问题是否为可拆解的复杂问题，如果是，则将其拆解为2个简单子问题；如果不是，则生成2条与原问题表述不同但指向相关专业概念的查询。总共3条子查询。

【输出格式】
{"queries": ["查询1", "查询2", "查询3"]}

【示例1：需要拆解的问题】
用户问题：某型坦克在高原环境下的发动机故障率以及对应的维修措施有哪些？
输出：{"queries": ["某型坦克在高原环境下的发动机故障率以及对应的维修措施有哪些？", "某型坦克高原环境发动机故障率", "某型坦克发动机高原故障维修措施"]}
【示例2：需要同义扩展的问题】
用户问题：标枪反坦克导弹为什么能够攻击坦克顶部装甲？
输出：{"queries":  ["为什么低压下的冲击转爆轰实验难以用AWSD模型捕捉？", "标枪导弹的“攻顶模式”是如何实现的？", "标枪导弹是如何实现从上方打击坦克的？"]}

仅输出JSON，无任何其他内容。

用户问题：{用户问题}"""


def _parse_queries_json(text: str) -> list:
    """从 LLM 输出中解析 queries 列表，最多取 3 条"""
    m = re.search(r"\{[\s\S]*\}", text.strip())
    if not m:
        return []
    try:
        data = json.loads(m.group())
        qs = data.get("queries", [])
        return [str(x).strip() for x in qs if str(x).strip()]
    except json.JSONDecodeError:
        return []


def expand_queries(user_question: str, llm) -> list:
    """调用 LLM 将原问题扩展为 3 条检索查询（原问题+2条），失败时回退到原问题"""
    prompt = REWRITE_PROMPT.replace("{用户问题}", user_question.strip())
    resp = llm.chat(messages=[{"role": "user", "content": prompt}], stream=False)
    raw = resp[-1].get("content", "") if resp else ""
    parsed = _parse_queries_json(raw)
    uq = user_question.strip()
    if not parsed:
        return [uq] * 3
    # 强制第1条为原问题
    if parsed[0] != uq:
        parsed = [uq] + [x for x in parsed if x != uq]
    # 补足3条
    while len(parsed) < 3:
        parsed.append(uq)
    return parsed[:3]


# ========== 检索函数（拆分自 test6plus.py 的 retrieve_context） ==========

def _recall_child_ids(query: str) -> list:
    """单条查询 -> 经 RRF 融合后的子块候选列表（不含重排）"""
    query_tokens = jieba.lcut(query)
    bm25_scores = bm25.get_scores(query_tokens)
    bm25_top_idx = np.argsort(bm25_scores)[-BM25_RECALL_K:][::-1]
    bm25_ranked_children = [int(i) for i in bm25_top_idx if i != -1]

    query_vec = encode_query(query)
    _, vec_indices = index.search(query_vec, VECTOR_RECALL_K)
    vec_ranked_children = [int(i) for i in vec_indices[0] if i != -1]

    rrf_scores = {}
    for r, cid in enumerate(bm25_ranked_children):
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + W_BM25 / (RRF_K + r + 1)
    for r, cid in enumerate(vec_ranked_children):
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + W_VECTOR / (RRF_K + r + 1)

    fused_candidates = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
    return fused_candidates[:RERANKER_TOP_CANDIDATES]


def retrieve_context_multi(queries: list) -> str:
    """对多条子查询分别召回子块，合并去重后用 queries[0] 重排，保留 Top3 子块"""
    # 收集所有子查询的召回子块并去重（保序）
    seen_child = set()
    all_candidates = []
    for q in queries:
        for cid in _recall_child_ids(q):
            if cid not in seen_child:
                seen_child.add(cid)
                all_candidates.append(cid)

    # 用"查询1"（queries[0]）对合并后的候选子块进行 Cross-Encoder 重排
    query1 = queries[0]
    cand_texts = [child_chunks[cid] for cid in all_candidates]
    pairs = [[query1, doc] for doc in cand_texts]
    batch_size = 3
    ce_scores = []
    print("Cross-Encoder 重排序中：")
    for i in range(0, len(pairs), batch_size):
        batch_scores = cross_encoder.predict(pairs[i:i+batch_size])
        ce_scores.extend(batch_scores)
        simple_progress_bar(min(i+batch_size, len(pairs)), len(pairs), prefix="Rerank ")

    # 保留 Top3 子块，映射到父块（去重保序）
    ranked_by_ce = sorted(zip(all_candidates, ce_scores), key=lambda x: float(x[1]), reverse=True)
    top_child_ids = [int(cid) for cid, _ in ranked_by_ce[:RERANK_TOP_CHILD_N]]

    parent_ordered = []
    seen_parent = set()
    for cid in top_child_ids:
        for pid in get_parent_ids_by_child_idx(cid):
            if pid not in seen_parent:
                seen_parent.add(pid)
                parent_ordered.append(pid)
    return "\n\n".join(parent_chunks[pid] for pid in parent_ordered)


# ========== LLM 及对话循环 ==========

llm_cfg = {
    'model': 'qwen2.5:1.5b',
    'model_server': 'http://localhost:11434/v1',
    'api_key': 'EMPTY'
}
llm = get_chat_model(llm_cfg)
system_message = """你是一个技术支持专家，请基于提供的专业文献，准确、专业地回答用户问题。
在生成回答时，请遵循以下原则，以确保答案的质量和专业性:
1.  **准确性**: 确保答案完全基于提供的知识库内容，避免提供超出范围的信息。
2.  **完整性**: 如果知识库中包含多个相关点，请尽量覆盖所有相关内容，确保回答全面。
3.  **引用**: 适当引用知识库中的具体句子或段落来支持你的回答。
4.  **简洁性**: 保持回答简明扼要，不要冗长或偏离主题。
5.  **专业性**: 使用正式的语言风格，确保回答符合专业标准，适合用于正式场合。"""

messages = []
print("文档问答助手已启动（输入'quit'退出）")
while True:
    query = input("\n用户: ")
    if query.lower() == 'quit':
        break

    messages.append({'role': 'user', 'content': query})

    # 多查询扩展 + 检索并计时
    start_time = time.time()
    sub_queries = expand_queries(query, llm)
    print(f"扩展查询: {sub_queries}")
    context = retrieve_context_multi(sub_queries)
    elapsed_time = time.time() - start_time
    print("上下文信息：", context)
    print(f"扩展+检索耗时: {elapsed_time:.2f} 秒")

    user_prompt_with_context = f"上下文信息：\n{context}\n\n问题：{query}\n请基于上下文回答问题。"
    messages[-1] = {'role': 'user', 'content': user_prompt_with_context}

    # 生成回答计时
    print("助手生成回答中...")
    gen_start = time.time()
    response = llm.chat(messages=messages, stream=False)
    gen_elapsed = time.time() - gen_start
    assistant_msg = response[-1]
    print(f"助手: {assistant_msg['content']}")
    print(f"生成回答耗时: {gen_elapsed:.2f} 秒")
    messages.append(assistant_msg)
