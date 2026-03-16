import httpx
import logging
import asyncio

logger = logging.getLogger(__name__)

class HTTPClient:
    def __init__(
        self,
        url: str,
        timeout: float = 10.0,
        max_keepalive_connections: int = 8,
        max_connections: int = 32,
    ):
        self.base_url = url

        self.client = httpx.AsyncClient(
            base_url=url,
            timeout=timeout,
            limits=httpx.Limits(
                max_keepalive_connections=max_keepalive_connections,
                max_connections=max_connections,
            ),
        )

    async def close(self):
        if self.client:
            await self.client.aclose()
            logger.info(f"HTTP Client closed for {self.base_url}")

    async def _post(self, endpoint: str, query: dict):
        try:
            response = await self.client.post(url=endpoint, json=query)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(
                f"HTTP Error {e.response.status_code} for {self.base_url}: {e}"
            )
            raise
        except httpx.RequestError as e:
            logger.error(f"Connection Error for {self.base_url}: {e}")
            raise

class QwenClient(HTTPClient):
    def __init__(
        self,
        url: str,
        model_name: str, 
        timeout: float = 60.0,  
        max_keepalive_connections: int = 8,
        max_connections: int = 32,
    ):
        super().__init__(
            url=url,
            timeout=timeout,
            max_keepalive_connections=max_keepalive_connections,
            max_connections=max_connections,
        )
        self.model_name = model_name

    async def generate_responses(
        self, 
        texts: list[str], 
        temperature: float = 0.7, 
        max_tokens: int = 2000
    ):
        tasks = []
        for text in texts:

            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "user", "content": text}
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "extra_body": {
                    "chat_template_kwargs": {"enable_thinking": False}
                }
            }
            
            tasks.append(self._post("/v1/chat/completions", payload))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        output = []
        for res in results:
            if isinstance(res, Exception):
                logger.error(f"Qwen request failed: {res}")
                output.append(None)
                continue

            if isinstance(res, dict):
                try:
                    
                    content = res["choices"][0]["message"]["content"]
                    output.append(content)
                except (KeyError, IndexError) as e:
                    logger.warning(f"Unexpected Qwen response format: {res}. Error: {e}")
                    output.append(None)
            else:
                logger.warning(f"Unexpected Qwen response type: {type(res)}")
                output.append(None)

        return output