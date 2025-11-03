"""Base interface for Large Language Model engines."""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any


class LLMResponse:
    """LLM 回應結果。"""
    
    def __init__(self, content: str, metadata: Optional[Dict[str, Any]] = None):
        self.content = content
        self.metadata = metadata or {}
    
    def __str__(self) -> str:
        return self.content
    
    def __repr__(self) -> str:
        return f"LLMResponse(content='{self.content[:50]}...', metadata={self.metadata})"


class LLMEngine(ABC):
    """Large Language Model 引擎的基礎類別。"""
    
    @abstractmethod
    def query(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        """
        向 LLM 發送查詢。
        
        Args:
            prompt: 使用者的提示訊息
            system_prompt: 系統提示訊息（可選）
            
        Returns:
            LLMResponse: LLM 的回應
        """
        pass
    
    @abstractmethod
    def chat(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
    ) -> LLMResponse:
        """
        進行多輪對話。
        
        Args:
            messages: 對話歷史，格式為 [{"role": "user|assistant", "content": "..."}]
            system_prompt: 系統提示訊息（可選）
            
        Returns:
            LLMResponse: LLM 的回應
        """
        pass
