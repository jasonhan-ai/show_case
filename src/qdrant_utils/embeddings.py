"""
文本向量生成模块。
"""
from typing import List
import numpy as np
import torch
from abc import ABC, abstractmethod
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F

class TextEmbedding(ABC):
    """文本向量生成基类"""
    
    @property
    @abstractmethod
    def vector_size(self) -> int:
        """向量维度"""
        pass
    
    @abstractmethod
    def generate_vector(self, texts: List[str]) -> List[np.ndarray]:
        """
        生成文本的向量表示
        :param texts: 文本列表
        :return: 向量列表
        """
        pass

class BGEEmbedding(TextEmbedding):
    """BGE 文本向量生成类"""
    
    def __init__(self, model_name: str = "BAAI/bge-large-zh-v1.5"):
        """
        初始化 BGE 向量生成器。
        
        参数：
            model_name: 模型名称
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self._vector_size = self.model.config.hidden_size
    
    @property
    def vector_size(self) -> int:
        """向量维度"""
        return self._vector_size
    
    def generate_vector(self, texts: List[str]) -> List[np.ndarray]:
        """
        生成文本的向量表示
        :param texts: 文本列表
        :return: 向量列表
        """
        # 对文本进行编码
        encoded_input = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )

        # 生成向量
        with torch.no_grad():
            model_output = self.model(**encoded_input)
            embeddings = model_output[0][:, 0]  # 使用 [CLS] token 的输出作为句子表示
            embeddings = F.normalize(embeddings, p=2, dim=1)  # L2 归一化

        return [embedding.numpy() for embedding in embeddings]

class Text2VecEmbedding(TextEmbedding):
    """Text2Vec 文本向量生成类"""
    
    def __init__(self, model_name: str = "shibing624/text2vec-base-chinese"):
        """
        初始化 Text2Vec 向量生成器。
        
        参数：
            model_name: 模型名称
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self._vector_size = self.model.config.hidden_size
    
    @property
    def vector_size(self) -> int:
        """向量维度"""
        return self._vector_size
    
    def generate_vector(self, texts: List[str]) -> List[np.ndarray]:
        """
        生成文本的向量表示
        :param texts: 文本列表
        :return: 向量列表
        """
        # 对文本进行编码
        encoded_input = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )

        # 生成向量
        with torch.no_grad():
            model_output = self.model(**encoded_input)
            embeddings = model_output[0][:, 0]  # 使用 [CLS] token 的输出作为句子表示
            embeddings = F.normalize(embeddings, p=2, dim=1)  # L2 归一化

        return [embedding.numpy() for embedding in embeddings] 