"""
Qdrant客户端配置模块。
"""
from typing import Optional
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

load_dotenv()

class QdrantClientConfig:
    """Qdrant客户端配置类。"""
    
    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        api_key: Optional[str] = None,
        https: bool = True
    ):
        """
        初始化Qdrant客户端配置。
        
        参数：
            host: Qdrant服务器主机地址
            port: Qdrant服务器端口
            api_key: 认证用的API密钥
            https: 是否使用HTTPS
        """
        self.host = host or os.getenv("QDRANT_HOST", "localhost")
        self.port = port or int(os.getenv("QDRANT_PORT", "6333"))
        self.api_key = api_key or os.getenv("QDRANT_API_KEY")
        self.https = https
        
    def get_client(self) -> QdrantClient:
        """
        获取配置好的Qdrant客户端实例。
        
        返回：
            QdrantClient: 配置好的客户端实例
        """
        return QdrantClient(
            host=self.host,
            port=self.port,
            api_key=self.api_key,
            https=self.https
        ) 