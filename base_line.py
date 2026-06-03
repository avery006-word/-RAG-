from qwen_agent.llm import get_chat_model
import pickle
import faiss
from sentence_transformers import SentenceTransformer
import time
# +检索时间+知识库(父子分块策略)
# 加载之前构建的知识库
FAISS_INDEX_PATH = "segment_processing1/faiss_parent_child_index.bin"
CHILD_DATA_PATH = "segment_processing1/child_data.pkl"
PARENT_DATA_PATH = "segment_processing1/parent_data.pkl"

print("加载知识库...")
index = faiss.read_index(FAISS_INDEX_PATH)
with open(CHILD_DATA_PATH, 'rb') as f:
    child_data = pickle.load(f)
with open(PARENT_DATA_PATH, 'rb') as f:
    parent_chunks = pickle.load(f)

child_chunks = child_data['chunks']
child_to_parent = child_data['parent_ids']

# 加载 embedding 模型用于查询编码
EMBED_MODEL_PATH = r"D:\anaconda\conda_envs\qwen-agent\models\Qwen3-Embedding-0.6B"
print("加载 Embedding 模型...")
embed_model = SentenceTransformer(EMBED_MODEL_PATH, device="cpu")


def encode_query(query):
    """编码查询向量，归一化"""
    emb = embed_model.encode([query], prompt_name="query", normalize_embeddings=True)
    return emb.astype('float32')


def retrieve_context(query, top_k=3):
    """检索相关父块上下文"""
    # 编码查询
    query_vec = encode_query(query)
    # 在 FAISS 中搜索 top_k 个子块
    distances, indices = index.search(query_vec, top_k)
    # 获取对应父块索引（去重）
    parent_indices = set()
    for idx in indices[0]:
        if idx != -1:
            parent_indices.add(child_to_parent[idx])
    # 根据父块索引获取文本
    contexts = [parent_chunks[pid] for pid in parent_indices]
    # 合并上下文（可以按原始顺序，但这里简单合并）
    context_text = "\n\n".join(contexts)
    return context_text


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

# ==================== 对话循环 ====================
messages = []
print("文档问答助手已启动（输入'quit'退出）")
while True:
    query = input("\n用户: ")
    if query.lower() == 'quit':
        break

    messages.append({'role': 'user', 'content': query})

    # 检索并计时
    start_time = time.time()
    context = retrieve_context(query)  # 使用你定义的检索函数
    elapsed_time = time.time() - start_time
    print(f"检索耗时: {elapsed_time:.2f} 秒")

    # 构造带上下文的 prompt
    user_prompt_with_context = f"上下文信息：\n{context}\n\n问题：{query}\n请基于上下文回答问题。"
    # 替换最后一条用户消息为带上下文的版本（因为原消息已添加，需要替换）
    messages[-1] = {'role': 'user', 'content': user_prompt_with_context}

    # 调用 LLM 获取回答（非流式，收集完整回答）
    print("上下文信息：", context)
    print("助手: ", end="", flush=True)
    response = llm.chat(messages=messages, stream=False)  # 返回的是列表，包含完整消息
    assistant_msg = response[-1]  # 取最后一条 assistant 消息
    print(assistant_msg['content'])
    messages.append(assistant_msg)  # 将助手回答加入历史