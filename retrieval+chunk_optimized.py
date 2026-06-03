# test6plus.py
# 基于 test6.py 增强：
# 1. Cross-Encoder 重排序分批处理 pairs
# 2. 输出检索耗时和生成回答耗时
# 3. 检索和生成环节添加简单进度条
# 其余逻辑保持一致，便于对比

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

BM25_RECALL_K = 10
VECTOR_RECALL_K = 10
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

def retrieve_context(query: str):
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
    fused_candidates = fused_candidates[:RERANKER_TOP_CANDIDATES]
    # Cross-Encoder rerank 分批处理
    cand_texts = [child_chunks[cid] for cid in fused_candidates]
    pairs = [[query, doc] for doc in cand_texts]
    batch_size = 3  # 可根据显存/性能调整
    ce_scores = []
    print("Cross-Encoder 重排序中：")
    for i in range(0, len(pairs), batch_size):
        batch_pairs = pairs[i:i+batch_size]
        batch_scores = cross_encoder.predict(batch_pairs)
        ce_scores.extend(batch_scores)
        simple_progress_bar(min(i+batch_size, len(pairs)), len(pairs), prefix="Rerank ")
    ranked_by_ce = sorted(
        zip(fused_candidates, ce_scores),
        key=lambda x: float(x[1]),
        reverse=True,
    )
    top_child_ids = [int(cid) for cid, _ in ranked_by_ce[:RERANK_TOP_CHILD_N]]
    parent_ordered = []
    seen_parent = set()
    for cid in top_child_ids:
        for pid in get_parent_ids_by_child_idx(cid):
            if pid not in seen_parent:
                seen_parent.add(pid)
                parent_ordered.append(pid)
    contexts = [parent_chunks[pid] for pid in parent_ordered]
    return parent_ordered, "\n\n".join(contexts)

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

print("文档问答助手已启动（输入'quit'退出）")
while True:
    query = input("\n用户: ")
    if query.lower() == 'quit':
        break
    # 检索并计时
    start_time = time.time()
    retrieval_parent_id, context = retrieve_context(query)
    elapsed_time = time.time() - start_time
    print(f"检索耗时: {elapsed_time:.2f} 秒")
    user_prompt_with_context = f"上下文信息：\n{context}\n\n问题：{query}\n请基于上下文回答问题。"
    current_messages = [
        {'role': 'system', 'content': system_message},
        {'role': 'user', 'content': user_prompt_with_context}
    ]
    # 生成回答计时
    print("助手生成回答中...")
    gen_start = time.time()
    response = llm.chat(messages=current_messages, stream=False)
    gen_elapsed = time.time() - gen_start
    assistant_msg = response[-1]
    print({
        "retrieval_parent_id": retrieval_parent_id,
        "answer": assistant_msg['content']
    })
    print(f"生成回答耗时: {gen_elapsed:.2f} 秒")
