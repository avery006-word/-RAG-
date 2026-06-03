from qwen_agent.llm import get_chat_model
import pickle
import faiss
from sentence_transformers import SentenceTransformer
import time
import json
# +检索时间+知识库（父子分段）+分块策略优化+parent_chunks减小并解决子块同时存在于多个父块的问题
# 加载之前构建的知识库
FAISS_INDEX_PATH = "knowledge_base/AWSD-Reactive-Burn-Model/faiss_parent_child_index.bin"
CHILD_DATA_PATH = "knowledge_base/AWSD-Reactive-Burn-Model/child_data.pkl"
PARENT_DATA_PATH = "knowledge_base/AWSD-Reactive-Burn-Model/parent_data.pkl"
INPUT_QA_JSON_PATH = "qa_dataset/AWSD-Reactive-Burn-Model/AWSD数据集JSON.json"
OUTPUT_QA_JSON_PATH = "qa_dataset/AWSD-Reactive-Burn-Model/AWSD数据集JSON_with_answer.json"

print("加载知识库...")
index = faiss.read_index(FAISS_INDEX_PATH)
with open(CHILD_DATA_PATH, 'rb') as f:
    child_data = pickle.load(f)
with open(PARENT_DATA_PATH, 'rb') as f:
    parent_chunks = pickle.load(f)

child_chunks = child_data['chunks']
child_to_parent = child_data['parent_ids']

# 子块 idx -> 父块 id 列表（兼容旧版返回 int 的情况）
def get_parent_ids_by_child_idx(idx: int):
    p = child_to_parent[idx]
    if isinstance(p, list):
        return p
    return [p]

# 加载 embedding 模型用于查询编码
EMBED_MODEL_PATH = r"D:\anaconda\conda_envs\qwen-agent\models\Qwen3-Embedding-0.6B"
print("加载 Embedding 模型...")
embed_model = SentenceTransformer(EMBED_MODEL_PATH, device="cpu")


def encode_query(query):
    """编码查询向量，归一化"""
    emb = embed_model.encode([query], prompt_name="query", normalize_embeddings=True)
    return emb.astype('float32')


def retrieve_context(query, top_k=4):
    """检索相关父块上下文"""
    # 编码查询
    query_vec = encode_query(query)
    # 在 FAISS 中搜索 top_k 个子块
    distances, indices = index.search(query_vec, top_k)
    # 获取对应父块索引（按检索顺序去重）
    parent_indices = []
    seen_parent_ids = set()
    for idx in indices[0]:
        if idx != -1:
            for pid in get_parent_ids_by_child_idx(idx):
                if pid not in seen_parent_ids:
                    seen_parent_ids.add(pid)
                    parent_indices.append(pid)
    # 根据父块索引获取文本
    contexts = [parent_chunks[pid] for pid in parent_indices]
    # 合并上下文（可以按原始顺序，但这里简单合并）
    context_text = "\n\n".join(contexts)
    return context_text, parent_indices


# 创建 LLM 实例
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

def answer_single_question(query):
    """单轮问答：不保留历史上下文，返回检索父块ID和回答。"""
    start_time = time.time()
    context, retrieval_parent_ids = retrieve_context(query)
    elapsed_time = time.time() - start_time
    print(f"检索耗时: {elapsed_time:.2f} 秒")

    user_prompt_with_context = f"上下文信息：\n{context}\n\n问题：{query}\n请基于上下文回答问题。"
    response = llm.chat(
        messages=[
            {'role': 'system', 'content': system_message},
            {'role': 'user', 'content': user_prompt_with_context},
        ],
        stream=False
    )
    assistant_msg = response[-1]
    answer = assistant_msg.get('content', '')
    return retrieval_parent_ids, answer


def run_batch_qa(input_json_path=INPUT_QA_JSON_PATH, output_json_path=OUTPUT_QA_JSON_PATH):
    """按数据集顺序逐条问答，并将结果写回新的JSON文件。"""
    with open(input_json_path, 'r', encoding='utf-8') as f:
        qa_data = json.load(f)

    total = len(qa_data)
    for i, item in enumerate(qa_data, start=1):
        query = item.get("question", "")
        if not query:
            item["retrieval_parent_id"] = []
            item["answer"] = ""
            continue

        print(f"\n[{i}/{total}] 问题: {query}")
        retrieval_parent_ids, answer = answer_single_question(query)
        item["retrieval_parent_id"] = retrieval_parent_ids
        item["answer"] = answer

    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(qa_data, f, ensure_ascii=False, indent=2)

    print(f"\n处理完成，结果已保存到: {output_json_path}")


if __name__ == "__main__":
    run_batch_qa()