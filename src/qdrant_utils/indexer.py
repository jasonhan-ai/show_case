"""
索引管理器模块。
"""
from typing import List, Dict, Any, Optional
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from .embeddings import TextEmbedding

class TextIndexer:
    """文本索引管理器类"""
    
    def __init__(
        self,
        embedding_model: TextEmbedding,
        qdrant_ops: QdrantClient,
        collection_name: str
    ):
        """
        初始化索引管理器。
        
        参数：
            embedding_model: 文本向量生成模型
            qdrant_ops: Qdrant 客户端实例
            collection_name: 集合名称
        """
        self.embedding_model = embedding_model
        self.qdrant_ops = qdrant_ops
        self.collection_name = collection_name
    
    def create_index(self, force: bool = False) -> bool:
        """
        创建索引。
        
        参数：
            force: 如果为True，则强制重新创建索引
        
        返回：
            bool: 成功返回True
        """
        if force:
            self.qdrant_ops.delete_collection(self.collection_name)
        
        try:
            # 检查集合是否已存在
            collections = self.qdrant_ops.client.get_collections()
            if self.collection_name in [c.name for c in collections.collections]:
                if not force:
                    return False
            
            # 创建新集合
            return self.qdrant_ops.create_collection(
                collection_name=self.collection_name,
                vector_size=self.embedding_model.vector_size
            )
        except Exception as e:
            print(f"创建索引失败：{e}")
            return False
    
    def add_texts(self, texts: List[str]) -> bool:
        """
        添加文本到索引
        :param texts: 文本列表
        :return: 是否成功添加
        """
        try:
            # 生成向量
            vectors = self.embedding_model.generate_vector(texts)
            
            # 添加向量
            return self.add_vectors(vectors, texts)
        except Exception as e:
            print(f"添加文本失败：{str(e)}")
            return False
    
    def search(self, query: str, limit: int = 10, score_threshold: float = 0.0) -> List[Dict]:
        """
        搜索相似文本
        :param query: 查询文本
        :param limit: 返回结果数量限制
        :param score_threshold: 相似度阈值
        :return: 搜索结果列表
        """
        try:
            # 生成查询文本的向量
            query_vector = self.embedding_model.generate_vector([query])[0]
            
            # 执行搜索
            results = self.qdrant_ops.query_points(
                collection_name=self.collection_name,
                vector=query_vector.tolist(),
                limit=limit,
                score_threshold=score_threshold
            )
            return [
                {
                    "id": point.id,
                    "score": point.score,
                    "payload": point.payload
                }
                for point in results
            ]
        except Exception as e:
            print(f"搜索失败：{str(e)}")
            return []
    
    def search_batch(self, queries: List[str], limit: int = 10, score_threshold: float = 0.0) -> List[List[Dict]]:
        """
        批量搜索相似文本
        :param queries: 查询文本列表
        :param limit: 每个查询返回的结果数量限制
        :param score_threshold: 相似度阈值
        :return: 搜索结果列表的列表
        """
        try:
            # 生成查询文本的向量
            query_vectors = self.embedding_model.generate_vector(queries)
            
            # 执行批量搜索
            results = []
            for query_vector in query_vectors:
                result = self.qdrant_ops.query_points(
                    collection_name=self.collection_name,
                    vector=query_vector.tolist(),
                    limit=limit,
                    score_threshold=score_threshold
                )
                results.append([
                    {
                        "id": point.id,
                        "score": point.score,
                        "payload": point.payload
                    }
                    for point in result
                ])
            return results
        except Exception as e:
            print(f"批量搜索失败：{str(e)}")
            return []
    
    def search_by_vector(
        self,
        vector: np.ndarray,
        limit: int = 10,
        score_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        使用向量搜索相似文本。
        
        参数：
            vector: 查询向量
            limit: 返回的最大结果数
            score_threshold: 最小相似度阈值
        
        返回：
            List[Dict]: 搜索结果列表
        """
        try:
            results = self.qdrant_ops.query_points(
                collection_name=self.collection_name,
                vector=vector.tolist(),
                limit=limit,
                score_threshold=score_threshold or 0.0
            )
            
            return [
                {
                    "id": point.id,
                    "score": point.score,
                    "payload": point.payload
                }
                for point in results
            ]
        except Exception as e:
            print(f"向量搜索失败：{e}")
            return []
    
    def add_vectors(self, vectors: List[np.ndarray], texts: List[str]) -> bool:
        """
        添加向量到索引
        :param vectors: 向量列表
        :param texts: 文本列表
        :return: 是否成功添加
        """
        try:
            # 构建点数据
            points = [
                {
                    "id": i,
                    "vector": vector.tolist(),
                    "payload": {"title": text}
                }
                for i, (vector, text) in enumerate(zip(vectors, texts))
            ]
            
            # 添加点数据
            return self.qdrant_ops.upsert_points_batch(
                collection_name=self.collection_name,
                points=points
            )
        except Exception as e:
            print(f"添加向量失败：{str(e)}")
            return False 