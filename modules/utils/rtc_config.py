"""WebRTC configuration utilities for STUN/TURN server setup."""

import os
from typing import Dict, List, Any


def build_ice_servers_from_env() -> List[Dict[str, Any]]:
    """從環境變數產生 ICE 伺服器設定。

    支援：
        - RTC_STUN_URLS: 逗號分隔，例如
            "stun:stun.l.google.com:19302,stun:stun1.l.google.com:19302"
        - RTC_TURN_URL:  TURN 入口，例如
            "turns:turn.example.com:443?transport=tcp" 或 "turn:turn.example.com:3478"
        - RTC_TURN_USERNAME / RTC_TURN_PASSWORD: TURN 認證
    
    Returns:
        ICE 伺服器設定列表
    """
    ice_servers = []

    # 處理 STUN URLs
    stun_urls = os.getenv("RTC_STUN_URLS")
    if stun_urls:
        urls = [u.strip() for u in stun_urls.split(",") if u.strip()]
        if urls:
            ice_servers.append({"urls": urls})

    # 處理 TURN 設定
    turn_url = os.getenv("RTC_TURN_URL")
    turn_user = os.getenv("RTC_TURN_USERNAME")
    turn_pass = os.getenv("RTC_TURN_PASSWORD")
    if turn_url:
        turn_entry = {"urls": [turn_url]}
        if turn_user and turn_pass:
            turn_entry["username"] = turn_user
            turn_entry["credential"] = turn_pass
        ice_servers.append(turn_entry)

    # 若未指定，預設提供 Google STUN 提升存活率
    if not ice_servers:
        ice_servers = [{"urls": ["stun:stun.l.google.com:19302"]}]

    return ice_servers


def get_client_rtc_config() -> Dict[str, Any]:
    """取得客戶端 RTC 配置。
    
    允許透過環境變數提供 STUN/TURN 設定，並可在瀏覽器端強制走 TURN (relay)
    以避免 ngrok/企業網無法轉發 UDP 造成的 ICE/DTLS/SRTP 連線問題。
    
    Returns:
        客戶端 RTC 配置字典
    """
    config = {"iceServers": build_ice_servers_from_env()}
    
    # 例："relay" 強制走 TURN
    ice_policy = os.getenv("RTC_ICE_TRANSPORT_POLICY")
    if ice_policy:
        # 只會作用在瀏覽器端 (RTCPeerConnection)
        config["iceTransportPolicy"] = ice_policy
    
    return config


def get_server_rtc_config() -> Dict[str, Any]:
    """取得伺服器端 RTC 配置。
    
    伺服器端 aiortc 也可以帶相同 iceServers
    （雖然 iceTransportPolicy 不適用於 server）
    
    Returns:
        伺服器端 RTC 配置字典
    """
    return {"iceServers": build_ice_servers_from_env()}
