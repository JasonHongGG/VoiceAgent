"""Tool 管理器 - 管理和執行 LLM Tools。"""

from typing import Dict, List, Optional, Any
import json
import re

from .base import BaseTool, ToolResult


class ToolManager:
    """
    工具管理器。
    
    負責：
    1. 註冊和管理可用的工具
    2. 生成工具的 schema 供 LLM 使用
    3. 解析 LLM 的工具調用請求
    4. 執行工具並返回結果
    """
    
    def __init__(self):
        """初始化工具管理器。"""
        self._tools: Dict[str, BaseTool] = {}
        print("[ToolManager] Initialized")
    
    def register_tool(self, tool: BaseTool) -> None:
        """
        註冊一個工具。
        
        Args:
            tool: 要註冊的工具實例
        """
        if tool.name in self._tools:
            print(f"[ToolManager] Warning: Tool '{tool.name}' already registered, overwriting")
        
        self._tools[tool.name] = tool
        print(f"[ToolManager] Registered tool: {tool.name}")
    
    def unregister_tool(self, tool_name: str) -> bool:
        """
        取消註冊一個工具。
        
        Args:
            tool_name: 工具名稱
            
        Returns:
            是否成功取消註冊
        """
        if tool_name in self._tools:
            del self._tools[tool_name]
            print(f"[ToolManager] Unregistered tool: {tool_name}")
            return True
        return False
    
    def get_tool(self, tool_name: str) -> Optional[BaseTool]:
        """
        獲取指定的工具。
        
        Args:
            tool_name: 工具名稱
            
        Returns:
            工具實例，如果不存在則返回 None
        """
        return self._tools.get(tool_name)
    
    def list_tools(self) -> List[str]:
        """
        列出所有已註冊的工具名稱。
        
        Returns:
            工具名稱列表
        """
        return list(self._tools.keys())
    
    def get_tools_schema(self) -> List[Dict[str, Any]]:
        """
        獲取所有工具的 schema（供 LLM 使用）。
        
        Returns:
            工具 schema 列表
        """
        return [tool.to_schema() for tool in self._tools.values()]
    
    def get_tools_description(self) -> str:
        """
        獲取所有工具的描述文字（供 LLM system prompt 使用）。
        
        Returns:
            工具描述文字
        """
        if not self._tools:
            return ""
        
        descriptions = [
            "# 可用工具\n",
            "你是一個使用**繁體中文**對話的智能助理，可以使用以下工具來協助使用者。",
            "**重要規則：**",
            "1. 當使用者的需求符合工具描述時，你必須調用相應的工具",
            "2. 所有對話必須使用繁體中文",
            "3. 調用工具後，用簡短的中文告訴使用者操作已完成\n"
        ]
        
        for tool in self._tools.values():
            descriptions.append(f"## 工具：{tool.name}")
            descriptions.append(tool.description)
            descriptions.append("\n**參數說明：**")
            for param in tool.parameters:
                required = "【必要】" if param.required else "【可選】"
                descriptions.append(f"- {param.name} ({param.type}) {required}: {param.description}")
            descriptions.append("")
        
        descriptions.append(
            "# 工具調用格式\n"
            "當你需要使用工具時，**必須**以以下 JSON 格式回應（不要加任何其他文字）：\n"
            "```tool\n"
            "{\n"
            '  "tool_name": "工具名稱",\n'
            '  "parameters": {\n'
            '    "參數名": "參數值"\n'
            "  }\n"
            "}\n"
            "```\n"
            "\n"
            "**範例：**\n"
            "使用者說：「幫我記帳，今天吃了200元的牛肉麵」\n"
            "你應該回應：\n"
            "```tool\n"
            "{\n"
            '  "tool_name": "accounting_agent",\n'
            '  "parameters": {\n'
            '    "toolInput": "今天吃了200元的牛肉麵"\n'
            "  }\n"
            "}\n"
            "```\n"
        )
        
        return "\n".join(descriptions)
    
    def execute_tool(self, tool_name: str, **parameters) -> ToolResult:
        """
        執行指定的工具。
        
        Args:
            tool_name: 工具名稱
            **parameters: 工具參數
            
        Returns:
            ToolResult: 執行結果
        """
        tool = self.get_tool(tool_name)
        
        if tool is None:
            return ToolResult(
                success=False,
                data=None,
                error=f"工具 '{tool_name}' 不存在",
            )
        
        print(f"[ToolManager] Executing tool: {tool_name} with parameters: {parameters}")
        
        try:
            result = tool.execute(**parameters)
            return result
        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                error=f"執行工具時發生錯誤: {str(e)}",
            )
    
    def parse_tool_call(self, text: str) -> Optional[tuple[str, Dict[str, Any]]]:
        """
        從 LLM 回應中解析工具調用。
        
        Args:
            text: LLM 的回應文字
            
        Returns:
            (tool_name, parameters) 如果找到工具調用，否則返回 None
        """
        # 尋找 ```tool ... ``` 區塊
        tool_pattern = r'```tool\s*\n(.*?)\n```'
        match = re.search(tool_pattern, text, re.DOTALL)
        
        if not match:
            return None
        
        try:
            tool_data = json.loads(match.group(1))
            tool_name = tool_data.get("tool_name")
            parameters = tool_data.get("parameters", {})
            
            if tool_name:
                print(f"[ToolManager] Parsed tool call: {tool_name}")
                return tool_name, parameters
        except json.JSONDecodeError as e:
            print(f"[ToolManager] Failed to parse tool call: {e}")
        
        return None
    
    def has_tools(self) -> bool:
        """
        檢查是否有已註冊的工具。
        
        Returns:
            是否有工具
        """
        return len(self._tools) > 0
    
    def __len__(self) -> int:
        """返回已註冊的工具數量。"""
        return len(self._tools)
    
    def __repr__(self) -> str:
        return f"<ToolManager: {len(self._tools)} tools registered>"
