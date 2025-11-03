"""Tool 基礎框架 - 讓 LLM 能夠調用外部工具。"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from dataclasses import dataclass


@dataclass
class ToolParameter:
    """工具參數定義。"""
    name: str
    type: str  # "string", "number", "boolean", "object", "array"
    description: str
    required: bool = True
    default: Optional[Any] = None


@dataclass
class ToolResult:
    """工具執行結果。"""
    success: bool
    data: Any
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __str__(self) -> str:
        if self.success:
            return f"成功: {self.data}"
        else:
            return f"失敗: {self.error}"


class BaseTool(ABC):
    """
    工具基礎類別。
    
    所有 LLM 可調用的工具都應該繼承這個類別。
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """工具名稱（唯一識別符）。"""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """工具描述（告訴 LLM 這個工具的功能）。"""
        pass
    
    @property
    @abstractmethod
    def parameters(self) -> list[ToolParameter]:
        """工具需要的參數列表。"""
        pass
    
    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """
        執行工具。
        
        Args:
            **kwargs: 工具參數
            
        Returns:
            ToolResult: 執行結果
        """
        pass
    
    def to_schema(self) -> Dict[str, Any]:
        """
        將工具轉換為 JSON Schema 格式（供 LLM 使用）。
        
        Returns:
            工具的 schema 定義
        """
        properties = {}
        required = []
        
        for param in self.parameters:
            properties[param.name] = {
                "type": param.type,
                "description": param.description,
            }
            if param.default is not None:
                properties[param.name]["default"] = param.default
            
            if param.required:
                required.append(param.name)
        
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            }
        }
    
    def validate_parameters(self, **kwargs) -> tuple[bool, Optional[str]]:
        """
        驗證參數是否正確。
        
        Args:
            **kwargs: 要驗證的參數
            
        Returns:
            (是否有效, 錯誤訊息)
        """
        # 檢查必要參數
        for param in self.parameters:
            if param.required and param.name not in kwargs:
                return False, f"缺少必要參數: {param.name}"
        
        return True, None
    
    def __repr__(self) -> str:
        return f"<Tool: {self.name}>"
