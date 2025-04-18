"""
Qdrant向量操作模块，用于数据导入和检索。
"""
from typing import List, Dict, Any, Optional
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from qdrant_client.models import Distance, VectorParams

class QdrantOperations:
    """用于处理Qdrant向量操作的类。"""
    
    def __init__(self, client: QdrantClient):
        """
        初始化QdrantOperations。
        
        参数：
            client: 配置好的QdrantClient实例
        """
        self.client = client
    
    def delete_collection(self, collection_name: str) -> bool:
        """
        删除指定的集合。
        
        参数：
            collection_name: 集合名称
            
        返回：
            bool: 成功返回True
        """
        try:
            self.client.delete_collection(collection_name=collection_name)
            return True
        except Exception as e:
            print(f"删除集合时出错：{e}")
            return False

    def create_collection(
        self,
        collection_name: str,
        vector_size: int,
        distance: Distance = Distance.COSINE
    ) -> bool:
        """
        在Qdrant中创建新的集合。
        
        参数：
            collection_name: 集合名称
            vector_size: 向量维度大小
            distance: 距离度量方式
            
        返回：
            bool: 成功返回True
        """
        try:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=distance)
            )
            return True
        except Exception as e:
            if "already exists" in str(e):
                return False
            print(f"创建集合时出错：{e}")
            return False
    
    def upsert_points(
        self,
        collection_name: str,
        vectors: List[List[float]],
        ids: Optional[List[str]] = None,
        payload: Optional[List[Dict[str, Any]]] = None
    ) -> bool:
        """
        向集合中上传向量数据。
        
        参数：
            collection_name: 集合名称
            vectors: 要上传的向量列表
            ids: 可选的向量ID列表
            payload: 可选的向量元数据列表
            
        返回：
            bool: 成功返回True
        """
        try:
            if ids is None:
                ids = [str(i) for i in range(len(vectors))]
            
            points = [
                rest.PointStruct(
                    id=id_,
                    vector=vector,
                    payload=payload[i] if payload else None
                )
                for i, (id_, vector) in enumerate(zip(ids, vectors))
            ]
            
            self.client.upsert(
                collection_name=collection_name,
                points=points
            )
            return True
        except Exception as e:
            print(f"上传向量时出错：{e}")
            return False
    
    def search(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 10,
        score_threshold: Optional[float] = None
    ) -> List[rest.ScoredPoint]:
        """
        在集合中搜索相似向量。
        
        参数：
            collection_name: 集合名称
            query_vector: 查询向量
            limit: 最大返回结果数
            score_threshold: 最小相似度阈值
            
        返回：
            List[ScoredPoint]: 搜索结果列表
        """
        try:
            return self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold
            )
        except Exception as e:
            print(f"搜索时出错：{e}")
            return []

    def upsert_points_batch(
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
            self.client.upsert(
                collection_name=collection_name,
                wait=True,
                points=[
                    rest.PointStruct(
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

    def query_points(
        self,
        collection_name: str,
        vector: List[float],
        limit: int = 10,
        score_threshold: float = 0.0
    ) -> List[rest.ScoredPoint]:
        """
        搜索相似向量
        :param collection_name: 集合名称
        :param vector: 查询向量
        :param limit: 返回结果数量限制
        :param score_threshold: 相似度阈值
        :return: 搜索结果列表
        """
        try:
            results = self.client.search(
                collection_name=collection_name,
                query_vector=vector,
                limit=limit,
                score_threshold=score_threshold
            )
            return results
        except Exception as e:
            print(f"搜索失败: {str(e)}")
            return []

    def query_batch_points(
        self,
        requests: List[Dict]
    ) -> List[List[rest.ScoredPoint]]:
        """
        批量搜索相似向量
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
                result = self.query_points(
                    collection_name=request["collection_name"],
                    vector=request["vector"],
                    limit=request["limit"],
                    score_threshold=request["score_threshold"]
                )
                results.append(result)
            return results
        except Exception as e:
            print(f"批量搜索失败: {str(e)}")
            return [] 