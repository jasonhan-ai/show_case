"""
测试Qdrant向量操作。
"""
import uuid
import numpy as np
from sentence_transformers import SentenceTransformer
from src.qdrant_utils import QdrantClientConfig, QdrantOperations

def normalize_vector(v):
    """标准化向量"""
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def get_embeddings(model, texts):
    """获取文本向量，添加模型需要的特殊前缀"""
    processed_texts = [f"为这个句子生成表示：{text}" for text in texts]
    embeddings = model.encode(processed_texts)
    return [normalize_vector(emb) for emb in embeddings]

def main():
    # 示例中文网文标题
    titles = [
        "重生之都市修仙",
        "我在修仙界开网店",
        "修真聊天群",
        "斗破苍穹",
        "完美世界",
        
        "凡人修仙传",
        "遮天",
        "我是大明星",
        "都市之最强狂兵",
        "超级战神在都市"
    ]
    
    # 加载文本向量模型
    print("加载文本向量模型...")
    model = SentenceTransformer('BAAI/bge-large-zh-v1.5')
    
    # 生成文本向量
    print("生成文本向量...")
    vectors = get_embeddings(model, titles)
    
    # 初始化Qdrant客户端
    print("初始化Qdrant客户端...")
    config = QdrantClientConfig()
    client = config.get_client()
    ops = QdrantOperations(client)
    
    # 创建集合
    collection_name = "novel_titles"
    vector_size = len(vectors[0])
    print(f"删除已存在的集合 {collection_name}...")
    ops.delete_collection(collection_name)
    print(f"创建新的集合 {collection_name}...")
    ops.create_collection(collection_name, vector_size)
    
    # 准备元数据和ID
    ids = [str(uuid.uuid4()) for _ in titles]
    payloads = [{"title": title} for title in titles]
    
    # 上传向量
    print("上传向量数据...")
    ops.upsert_points(
        collection_name=collection_name,
        vectors=vectors,
        ids=ids,
        payload=payloads
    )
    
    # 测试搜索
    print("\n开始测试搜索...")
    test_queries = [
        "修仙小说",
        "都市类小说",
        "玄幻小说",
        "网络小说",
        "仙侠小说"
    ]
    
    for query in test_queries:
        print(f"\n搜索查询: {query}")
        query_vector = get_embeddings(model, [query])[0]
        results = ops.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=3
        )
        
        print("搜索结果:")
        for res in results:
            print(f"- {res.payload['title']} (相似度: {res.score:.4f})")

if __name__ == "__main__":
    main() 