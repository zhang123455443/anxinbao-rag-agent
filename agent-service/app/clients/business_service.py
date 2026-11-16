from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from uuid import UUID

import httpx


class BusinessServiceError(RuntimeError):
    def __init__(self, status_code: int, body: Any, message: str | None = None) -> None:
        self.status_code = status_code
        self.body = body
        super().__init__(message or f"business-service returned HTTP {status_code}")


class BusinessServiceClient:
    def __init__(
        self,
        base_url: str,
        *,
        http_client: httpx.AsyncClient | None = None,
        timeout_seconds: float = 10.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._external_client = http_client is not None
        self._client = http_client or httpx.AsyncClient(
            base_url=self.base_url,
            timeout=timeout_seconds,
        )

    async def aclose(self) -> None:
        if not self._external_client:
            await self._client.aclose()

    async def get_product(self, product_id: int) -> dict[str, Any]:
        return await self._request("GET", f"/api/v1/products/{product_id}")

    async def list_products(
        self,
        *,
        category: str | None = None,
        product_name: str | None = None,
        page: int = 1,
        page_size: int = 20,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "page": page,
            "page_size": page_size,
        }
        if category:
            params["category"] = category
        if product_name:
            params["product_name"] = product_name
        return await self._request("GET", "/api/v1/products", params=params)

    async def list_candidate_products(
        self,
        categories: list[str],
        *,
        premium_min: float | None = None,
        limit_per_category: int = 5,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {
            "categories": categories,
            "limit_per_category": limit_per_category,
        }
        if premium_min is not None:
            params["premium_min"] = premium_min
        response = await self._request(
            "GET",
            "/api/v1/products/candidates",
            params=params,
        )
        return response.get("data", [])

    async def get_product_pricing_schema(self, product_id: int) -> dict[str, Any]:
        return await self._request("GET", f"/api/v1/products/{product_id}/pricing-schema")

    async def calculate_premium(
        self,
        user_token: str,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        return await self._request(
            "POST",
            "/api/v1/premium/calculate",
            user_token=user_token,
            json=payload,
        )

    async def save_current_insurance_plan(
        self,
        user_token: str,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        return await self._request(
            "PUT",
            "/api/v1/insurance-plans/current",
            user_token=user_token,
            json=payload,
        )

    async def get_current_insurance_plan(self, user_token: str) -> dict[str, Any]:
        return await self._request(
            "GET",
            "/api/v1/insurance-plans/current",
            user_token=user_token,
        )

    async def get_insurance_plan(self, user_token: str, plan_id: str | UUID) -> dict[str, Any]:
        return await self._request(
            "GET",
            f"/api/v1/insurance-plans/{plan_id}",
            user_token=user_token,
        )

    async def mark_insurance_plan_item_insured(
        self,
        user_token: str,
        plan_id: str | UUID,
        item_id: str | UUID,
    ) -> dict[str, Any]:
        return await self._request(
            "PATCH",
            f"/api/v1/insurance-plans/{plan_id}/items/{item_id}/insured",
            user_token=user_token,
        )

    async def list_policies(
        self,
        user_token: str,
        *,
        status: str = "active",
    ) -> dict[str, Any]:
        return await self._request(
            "GET",
            "/api/v1/policies",
            user_token=user_token,
            params={"status": status},
        )

    async def get_policy(self, user_token: str, policy_id: str | UUID) -> dict[str, Any]:
        return await self._request(
            "GET",
            f"/api/v1/policies/{policy_id}",
            user_token=user_token,
        )

    async def get_claim_guide(
        self,
        user_token: str,
        policy_id: str | UUID,
    ) -> dict[str, Any]:
        result = await self._request(
            "GET",
            f"/api/v1/claim-guides/{policy_id}",
            user_token=user_token,
        )
        return result

    async def list_claims(
        self,
        user_token: str,
        *,
        policy_id: str | UUID | None = None,
        status: str | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {}
        if policy_id is not None:
            params["policy_id"] = str(policy_id)
        if status:
            params["status"] = status
        return await self._request(
            "GET",
            "/api/v1/claims",
            user_token=user_token,
            params=params,
        )

    async def get_claim(self, user_token: str, claim_id: str | UUID) -> dict[str, Any]:
        return await self._request(
            "GET",
            f"/api/v1/claims/{claim_id}",
            user_token=user_token,
        )

    async def _request(
        self,
        method: str,
        path: str,
        *,
        user_token: str | None = None,
        json: Mapping[str, Any] | None = None,
        params: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        headers = self._auth_headers(user_token)
        response = await self._client.request(method, path, headers=headers, json=json, params=params)
        if response.status_code >= 400:
            raise BusinessServiceError(response.status_code, self._response_body(response))
        body = self._response_body(response)
        return body if isinstance(body, dict) else {"data": body}

    def _auth_headers(self, user_token: str | None) -> dict[str, str]:
        if not user_token:
            return {}
        return {"Authorization": f"Bearer {user_token}"}

    def _response_body(self, response: httpx.Response) -> Any:
        try:
            return response.json()
        except ValueError:
            return response.text
