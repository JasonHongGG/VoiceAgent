"""Ollama-based LLM implementation."""

import json
from typing import Optional, List, Dict, Any, Iterator
import requests

from .base import LLMEngine, LLMResponse


class OllamaLLM(LLMEngine):
    """使用 Ollama API 的 LLM 引擎。"""
    
    def __init__(
        self,
        api_url: str,
        model: str,
        default_system_prompt: str,
        timeout: int = 60,
    ):
        """
        初始化 Ollama LLM 引擎。
        
        Args:
            api_url: Ollama API 的基礎 URL (例如 "http://localhost:11434")
            model: 使用的模型名稱
            default_system_prompt: 預設的系統提示訊息
            timeout: 請求超時時間（秒）
        """
        self.api_url = api_url.rstrip("/")
        self.model = model
        self.default_system_prompt = default_system_prompt 
        self.timeout = timeout
        
        print(f"[OllamaLLM] Initialized with model '{model}' at {api_url}")
    
    def query(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        """
        向 LLM 發送查詢。
        
        Args:
            prompt: 使用者的提示訊息
            system_prompt: 系統提示訊息（可選，會覆蓋預設值）
            
        Returns:
            LLMResponse: LLM 的回應
        """
        system = system_prompt if system_prompt is not None else self.default_system_prompt
        
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        return self.chat(messages)
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
    ) -> LLMResponse:
        """
        進行多輪對話。
        
        Args:
            messages: 對話歷史，格式為 [{"role": "user|assistant|system", "content": "..."}]
            system_prompt: 系統提示訊息（可選，會插入到對話開頭）
            
        Returns:
            LLMResponse: LLM 的回應
        """
        # 如果提供了 system_prompt，將其插入到對話開頭
        if system_prompt is not None:
            messages = [{"role": "system", "content": system_prompt}] + messages
        
        # 準備請求
        url = f"{self.api_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": False,
        }
        
        headers = {
            "Content-Type": "application/json"
        }
        
        print(f"[OllamaLLM] Sending request to {url}")
        
        try:
            response = requests.post(
                url,
                headers=headers,
                data=json.dumps(payload),
                timeout=self.timeout
            )
            
            # 檢查是否成功
            if response.status_code == 200:
                result = response.json()
                content = result.get("message", {}).get("content", "")
                
                print(f"[OllamaLLM] Received response: {content[:100]}...")
                
                return LLMResponse(
                    content=content,
                    metadata={
                        "model": result.get("model"),
                        "created_at": result.get("created_at"),
                        "done": result.get("done"),
                    }
                )
            else:
                error_msg = f"請求失敗: {response.status_code}"
                print(f"[OllamaLLM] {error_msg}")
                print(f"[OllamaLLM] Response: {response.text}")
                return LLMResponse(
                    content=error_msg,
                    metadata={"error": True, "status_code": response.status_code}
                )
        
        except requests.exceptions.Timeout:
            error_msg = f"請求超時 (>{self.timeout}s)"
            print(f"[OllamaLLM] {error_msg}")
            return LLMResponse(
                content=error_msg,
                metadata={"error": True, "timeout": True}
            )
        
        except Exception as exc:
            error_msg = f"請求發生錯誤: {str(exc)}"
            print(f"[OllamaLLM] {error_msg}")
            return LLMResponse(
                content=error_msg,
                metadata={"error": True, "exception": str(exc)}
            )
    
    def query_stream(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None
    ) -> Iterator[str]:
        """
        流式查詢 LLM（逐字生成）。
        
        Args:
            prompt: 使用者的提示訊息
            system_prompt: 系統提示訊息（可選）
            
        Yields:
            str: 逐步生成的文字片段
        """
        system = system_prompt if system_prompt is not None else self.default_system_prompt
        
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        yield from self.chat_stream(messages)
    
    def chat_stream(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
    ) -> Iterator[str]:
        """
        流式多輪對話（逐字生成）。
        
        Args:
            messages: 對話歷史
            system_prompt: 系統提示訊息（可選）
            
        Yields:
            str: 逐步生成的文字片段
        """
        # 如果提供了 system_prompt，將其插入到對話開頭
        if system_prompt is not None:
            messages = [{"role": "system", "content": system_prompt}] + messages
        
        # 準備請求 (啟用串流)
        url = f"{self.api_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,  # 啟用串流模式
            "think": False,
        }
        
        headers = {
            "Content-Type": "application/json"
        }
        
        print(f"[OllamaLLM] Sending streaming request to {url}")
        
        try:
            response = requests.post(
                url,
                headers=headers,
                data=json.dumps(payload),
                timeout=self.timeout,
                stream=True  # 啟用串流接收
            )
            
            if response.status_code == 200:
                # 逐行讀取串流回應
                for line in response.iter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            # 提取訊息內容
                            message = data.get("message", {})
                            content = message.get("content", "")
                            
                            if content:
                                yield content
                            
                            # 檢查是否完成
                            if data.get("done", False):
                                print(f"[OllamaLLM] Streaming completed")
                                break
                        except json.JSONDecodeError:
                            continue
            else:
                error_msg = f"串流請求失敗: {response.status_code}"
                print(f"[OllamaLLM] {error_msg}")
                yield error_msg
        
        except requests.exceptions.Timeout:
            error_msg = f"串流請求超時 (>{self.timeout}s)"
            print(f"[OllamaLLM] {error_msg}")
            yield error_msg
        
        except Exception as exc:
            error_msg = f"串流請求發生錯誤: {str(exc)}"
            print(f"[OllamaLLM] {error_msg}")
            yield error_msg
