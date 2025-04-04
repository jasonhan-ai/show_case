"""
异步 Qdrant 操作模块。
"""
from typing import List, Dict, Any, Optional
import numpy as np
from qdrant_client.async_qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct, SearchRequest

class AsyncQdrantOperations:
    """异步 Qdrant 操作类"""
    
    def __init__(self, client: AsyncQdrantClient):
        """
        初始化异步操作类。
        
        Args:
            client: 异步 Qdrant 客户端实例
        """
        self.client = client
    
    async def delete_collection(self, collection_name: str) -> bool:
        """
        删除集合。
        
        Args:
            collection_name: 集合名称
        
        Returns:
            bool: 是否成功删除
        """
        try:
            await self.client.delete_collection(collection_name)
            return True
        except Exception as e:
            print(f"删除集合失败: {e}")
            return False
    
    async def create_collection(
        self,
        collection_name: str,
        vector_size: int
    ) -> bool:
        """
        创建集合。
        
        Args:
            collection_name: 集合名称
            vector_size: 向量维度
        
        Returns:
            bool: 是否成功创建
        """
        try:
            await self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE
                )
            )
            return True
        except Exception as e:
            print(f"创建集合失败: {e}")
            return False
    
    async def upsert_points_batch(
        self,
        collection_name: str,
        points: List[Dict]
    ) -> bool:
        """
        批量上传向量数据
        :param collection_name: 集合名称
        :param points: 点数据列表，每个点包含以下字段：
            - id: 点ID
            - vector: 向量数据
            - payload: 附加数据
        :return: 是否成功上传
        """
        try:
            await self.client.upsert(
                collection_name=collection_name,
                wait=True,
                points=[
                    PointStruct(
                        id=point["id"],
                        vector=point["vector"],
                        payload=point["payload"]
                    )
                    for point in points
                ]
            )
            return True
        except Exception as e:
            print(f"上传失败: {str(e)}")
            return False
    
    async def search_batch(self, requests: List[Dict]) -> List[List[Dict]]:
        """
        批量搜索向量
        :param requests: 搜索请求列表，每个请求包含以下字段：
            - collection_name: 集合名称
            - vector: 查询向量
            - limit: 返回结果数量限制
            - score_threshold: 相似度阈值
        :return: 搜索结果列表的列表
        """
        try:
            results = []
            for request in requests:
                result = await self.query_points(
                    collection_name=request["collection_name"],
                    vector=request["vector"],
                    limit=request["limit"],
                    score_threshold=request["score_threshold"]
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
            print(f"搜索失败: {str(e)}")
            return []

    async def query_points(
        self,
        collection_name: str,
        vector: List[float],
        limit: int = 10,
        score_threshold: float = 0.0
    ) -> List[PointStruct]:
        """
        搜索相似向量
        :param collection_name: 集合名称
        :param vector: 查询向量
        :param limit: 返回结果数量限制
        :param score_threshold: 相似度阈值
        :return: 搜索结果列表
        """
        try:
            results = await self.client.query_points(
                collection_name=collection_name,
                vector=vector,
                limit=limit,
                score_threshold=score_threshold
            )
            return results
        except Exception as e:
            print(f"搜索失败: {str(e)}")
            return [] 