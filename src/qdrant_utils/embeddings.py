"""
向量生成模块。
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Dict
import numpy as np
from sentence_transformers import SentenceTransformer

def normalize_vector(v: np.ndarray) -> np.ndarray:
    """标准化向量"""
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

class TextEmbedding(ABC):
    """文本向量生成抽象基类"""
    
    @abstractmethod
    def encode(self, texts: List[str]) -> List[np.ndarray]:
        """
        将文本转换为向量。
        
        参数：
            texts: 要转换的文本列表
            
        返回：
            List[np.ndarray]: 向量列表
        """
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """返回向量维度"""
        pass

class BGEEmbedding(TextEmbedding):
    """BGE模型的向量生成实现"""
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-large-zh-v1.5",
        normalize: bool = True,
        prefix: str = "为这个句子生成表示："
    ):
        """
        初始化BGE向量生成器。
        
        参数：
            model_name: 模型名称
            normalize: 是否标准化向量
            prefix: 文本前缀
        """
        self.model = SentenceTransformer(model_name)
        self.normalize = normalize
        self.prefix = prefix
        self._dimension = self.model.encode("测试").shape[0]
    
    def encode(self, texts: List[str]) -> List[np.ndarray]:
        """
        将文本转换为向量。
        
        参数：
            texts: 要转换的文本列表
            
        返回：
            List[np.ndarray]: 向量列表
        """
        processed_texts = [f"{self.prefix}{text}" for text in texts]
        embeddings = self.model.encode(processed_texts)
        if self.normalize:
            return [normalize_vector(emb) for emb in embeddings]
        return [emb for emb in embeddings]
    
    @property
    def dimension(self) -> int:
        """返回向量维度"""
        return self._dimension

class Text2VecEmbedding(TextEmbedding):
    """Text2Vec模型的向量生成实现"""
    
    def __init__(
        self,
        model_name: str = "shibing624/text2vec-base-chinese",
        normalize: bool = True
    ):
        """
        初始化Text2Vec向量生成器。
        
        参数：
            model_name: 模型名称
            normalize: 是否标准化向量
        """
        self.model = SentenceTransformer(model_name)
        self.normalize = normalize
        self._dimension = self.model.encode("测试").shape[0]
    
    def encode(self, texts: List[str]) -> List[np.ndarray]:
        """
        将文本转换为向量。
        
        参数：
            texts: 要转换的文本列表
            
        返回：
            List[np.ndarray]: 向量列表
        """
        embeddings = self.model.encode(texts)
        if self.normalize:
            return [normalize_vector(emb) for emb in embeddings]
        return [emb for emb in embeddings]
    
    @property
    def dimension(self) -> int:
        """返回向量维度"""
        return self._dimension 