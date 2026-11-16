from __future__ import annotations

from typing import Any

import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic.dataclasses import dataclass

from app.core.config import settings

# 使用 HTTPBearer 安全方案，会自动从 Authorization: Bearer <token> 提取
security = HTTPBearer()

class InvalidAccessTokenError(ValueError):
    pass

@dataclass(frozen=True, slots=True)
class UserContext:
    user_id: int
    username: str | None = None

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """核心依赖：解析 Token 并返回用户对象"""
    token = credentials.credentials
    try:
        # 1. 解码 Token
        payload = jwt.decode(token, settings.jwt_secret, algorithms=['HS256'])
        user_id: int = _as_int(payload.get("user_id") or payload.get("sub"))
        username: str = payload.get("username")
        if user_id is None:
            raise HTTPException(status_code=401, detail="无效的认证凭据")
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token 已过期，请重新登录")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="无效的 Token")

    return UserContext(user_id=user_id, username=str(username)), token



def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None
