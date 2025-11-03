"""記帳工具 - 通過 WebHook 呼叫記帳 Agent。"""

import requests
import json
from typing import Any, Dict

from .base import BaseTool, ToolParameter, ToolResult


class AccountingAgentWebHook(BaseTool):
    """
    記帳代理工具。
    
    當使用者有記帳相關需求時，調用此工具。
    例如：
    - "幫我記帳，今天中午吃了200元的牛肉麵"
    - "記錄一下，昨天買了一杯50元的咖啡"
    - "我今天下午兩點肚子餓，去學校後面吃了牛肉麵花了200元，幫我記帳"
    """
    
    def __init__(
        self,
        webhook_url: str = "https://6aeda076cc90.ngrok-free.app/webhook/7a336883-1379-438d-aa08-95d1af38ef80",
        timeout: int = 30,
    ):
        """
        初始化記帳工具。
        
        Args:
            webhook_url: WebHook URL
            timeout: 請求超時時間（秒）
        """
        self._webhook_url = webhook_url
        self._timeout = timeout
        
        print(f"[AccountingAgentWebHook] Initialized with URL: {webhook_url}")
    
    @property
    def name(self) -> str:
        return "accounting_agent"
    
    @property
    def description(self) -> str:
        return (
            "記帳工具。當使用者提到記帳、記錄支出、花費、消費等需求時使用此工具。"
            "此工具會將使用者的記帳需求轉發給專門的記帳系統處理。"
        )
    
    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="toolInput",
                type="string",
                description=(
                    "清楚描述使用者的記帳需求。"
                    "應該包含：時間、項目、金額等資訊。"
                    "例如：'今天下午兩點吃200元的牛肉麵'、'昨天買50元的咖啡'"
                ),
                required=True,
            )
        ]
    
    def execute(self, **kwargs) -> ToolResult:
        """
        執行記帳操作。
        
        Args:
            toolInput: 記帳需求描述
            
        Returns:
            ToolResult: 執行結果
        """
        # 驗證參數
        is_valid, error = self.validate_parameters(**kwargs)
        if not is_valid:
            return ToolResult(
                success=False,
                data=None,
                error=error,
            )
        
        tool_input = kwargs.get("toolInput")
        
        print(f"[AccountingAgentWebHook] Processing: {tool_input}")
        
        try:
            # 發送 POST 請求到 WebHook
            response = requests.post(
                self._webhook_url,
                json={"toolInput": tool_input},
                headers={"Content-Type": "application/json"},
                timeout=self._timeout,
            )
            
            # 檢查回應
            if response.status_code == 200:
                try:
                    result_data = response.json()
                except json.JSONDecodeError:
                    result_data = {"raw_response": response.text}
                
                print(f"[AccountingAgentWebHook] Success: {result_data}")
                
                return ToolResult(
                    success=True,
                    data=result_data,
                    metadata={
                        "status_code": response.status_code,
                        "tool_input": tool_input,
                    }
                )
            else:
                error_msg = f"WebHook 請求失敗 (HTTP {response.status_code})"
                print(f"[AccountingAgentWebHook] {error_msg}")
                print(f"[AccountingAgentWebHook] Response: {response.text}")
                
                return ToolResult(
                    success=False,
                    data=None,
                    error=error_msg,
                    metadata={
                        "status_code": response.status_code,
                        "response_text": response.text,
                    }
                )
        
        except requests.exceptions.Timeout:
            error_msg = f"WebHook 請求超時 (>{self._timeout}s)"
            print(f"[AccountingAgentWebHook] {error_msg}")
            return ToolResult(
                success=False,
                data=None,
                error=error_msg,
            )
        
        except requests.exceptions.RequestException as e:
            error_msg = f"WebHook 請求失敗: {str(e)}"
            print(f"[AccountingAgentWebHook] {error_msg}")
            return ToolResult(
                success=False,
                data=None,
                error=error_msg,
            )
        
        except Exception as e:
            error_msg = f"執行工具時發生錯誤: {str(e)}"
            print(f"[AccountingAgentWebHook] {error_msg}")
            return ToolResult(
                success=False,
                data=None,
                error=error_msg,
            )
