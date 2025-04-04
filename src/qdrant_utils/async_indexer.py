"""
异步索引管理器模块。
"""
from typing import List, Dict, Any, Optional
import asyncio
from .embeddings import TextEmbedding
from .async_operations import AsyncQdrantOperations

class AsyncTextIndexer:
    """异步文本索引管理器类"""
    
    def __init__(
        self,
        embedding_model: TextEmbedding,
        operations: AsyncQdrantOperations,
        collection_name: str
    ):
        """
        初始化异步索引管理器。
        
        Args:
            embedding_model: 文本向量生成模型
            operations: 异步 Qdrant 操作类实例
            collection_name: 集合名称
        """
        self.embedding_model = embedding_model
        self.operations = operations
        self.collection_name = collection_name
    
    async def create_index(self, force: bool = False) -> bool:
        """
        创建索引。
        
        Args:
            force: 是否强制重建索引
        
        Returns:
            bool: 是否成功创建
        """
        if force:
            await self.operations.delete_collection(self.collection_name)
        
        vector_size = self.embedding_model.vector_size
        return await self.operations.create_collection(
            collection_name=self.collection_name,
            vector_size=vector_size
        )
    
    async def add_texts_batch(
        self,
        texts: List[str],
        batch_size: int = 32
    ) -> bool:
        """
        批量添加文本到索引
        :param texts: 文本列表
        :param batch_size: 批处理大小
        :return: 是否成功添加
        """
        try:
            # 生成向量
            vectors = self.embedding_model.generate_vector(texts)
            
            # 构建点数据
            points = [
                {
                    "id": i,
                    "vector": vector.tolist(),
                    "payload": {"title": text}
                }
                for i, (vector, text) in enumerate(zip(vectors, texts))
            ]
            
            # 分批处理
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                success = await self.operations.upsert_points_batch(
                    collection_name=self.collection_name,
                    points=batch
                )
                if not success:
                    return False
            
            return True
        except Exception as e:
            print(f"添加文本失败：{str(e)}")
            return False
    
    async def search_batch(
        self,
        queries: List[str],
        limit: int = 10,
        score_threshold: float = 0.0,
        batch_size: int = 10
    ) -> List[List[Dict]]:
        """
        批量搜索相似文本
        :param queries: 查询文本列表
        :param limit: 每个查询返回的结果数量限制
        :param score_threshold: 相似度阈值
        :param batch_size: 批处理大小
        :return: 搜索结果列表的列表
        """
        try:
            # 生成查询文本的向量
            query_vectors = self.embedding_model.generate_vector(queries)
            
            # 构建搜索请求
            requests = [
                {
                    "collection_name": self.collection_name,
                    "vector": vector.tolist(),
                    "limit": limit,
                    "score_threshold": score_threshold
                }
                for vector in query_vectors
            ]
            
            # 分批处理搜索请求
            results = []
            for i in range(0, len(requests), batch_size):
                batch_requests = requests[i:i + batch_size]
                batch_results = await self.operations.search_batch(requests=batch_requests)
                results.extend(batch_results)
            
            return results
        except Exception as e:
            print(f"搜索失败: {str(e)}")
            return []
    
    async def search(
        self,
        query: str,
        limit: int = 10,
        score_threshold: float = 0.0
    ) -> List[Dict]:
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
            
            # 构建搜索请求
            request = {
                "collection_name": self.collection_name,
                "vector": query_vector.tolist(),
                "limit": limit,
                "score_threshold": score_threshold
            }
            
            # 执行搜索
            results = await self.operations.search_batch([request])
            return results[0] if results else []
        except Exception as e:
            print(f"搜索失败: {str(e)}")
            return [] 