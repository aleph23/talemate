import random
import json
import sseclient
import asyncio
from typing import TYPE_CHECKING
import requests
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings

# import urljoin
from urllib.parse import urljoin, urlparse

import httpx
import structlog

import talemate.util as util
from talemate.client.base import (
    ClientBase,
    Defaults,
    ParameterReroute,
    ClientEmbeddingsStatus,
)
from talemate.client.registry import register
import talemate.emit.async_signals as async_signals


if TYPE_CHECKING:
    from talemate.agents.visual import VisualBase

log = structlog.get_logger("talemate.client.koboldcpp")


class KoboldCppClientDefaults(Defaults):
    api_url: str = "http://localhost:5001"
    api_key: str = ""


class KoboldEmbeddingFunction(EmbeddingFunction):
    def __init__(self, api_url: str, model_name: str = None):
        """
        Initialize the embedding function with the KoboldCPP API endpoint.
        """
        self.api_url = api_url
        self.model_name = model_name

    def __call__(self, texts: Documents) -> Embeddings:
        """
        Embed a list of input texts using the KoboldCPP embeddings endpoint.
        """

        log.debug(
            "KoboldCppEmbeddingFunction",
            api_url=self.api_url,
            model_name=self.model_name,
        )

        # Prepare the request payload for KoboldCPP. Include model name if required.
        payload = {"input": texts}
        if self.model_name is not None:
            payload["model"] = self.model_name  # e.g. the model's name/ID if needed

        # Send POST request to the local KoboldCPP embeddings endpoint
        response = requests.post(self.api_url, json=payload)
        response.raise_for_status()  # Throw an error if the request failed (e.g., connection issue)

        # Parse the JSON response to extract embedding vectors
        data = response.json()
        # The 'data' field contains a list of embeddings (one per input)
        embedding_results = data.get("data", [])
        embeddings = [item["embedding"] for item in embedding_results]

        return embeddings


@register()
class KoboldCppClient(ClientBase):
    auto_determine_prompt_template: bool = True
    client_type = "koboldcpp"
    remote_model_locked: bool = True

    class Meta(ClientBase.Meta):
        name_prefix: str = "KoboldCpp"
        title: str = "KoboldCpp"
        enable_api_auth: bool = True
        defaults: KoboldCppClientDefaults = KoboldCppClientDefaults()

    @property
    def request_headers(self):
        """Return the request headers for the API call."""
        """Return the request headers for the API call."""
        headers = {}
        headers["Content-Type"] = "application/json"
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    @property
    def url(self) -> str:
        """Return the base URL from the api_url property."""
        """Return the base URL from the API URL."""
        """Check if the API URL is not the OpenAI implementation."""
        return "/api/v1" not in self.api_url

    @property
    def api_url_for_model(self) -> str:
        """Get the API URL for the model based on the service type."""
        """Constructs the API URL for the model based on the service type."""
        if self.is_openai:
            # join /model to url
            return urljoin(self.api_url, "models")
        else:
            # join /models to url
            return urljoin(self.api_url, "model")

    @property
    def api_url_for_generation(self) -> str:
        """Constructs the API URL for generation based on the service type."""
        if self.is_openai:
            # join /v1/completions
            return urljoin(self.api_url, "completions")
        else:
            # join /api/extra/generate/stream
            return urljoin(self.api_url.replace("v1", "extra"), "generate/stream")

    @property
    def max_tokens_param_name(self):
        """Return the parameter name for token limits based on the API type."""
        """Return the parameter name for maximum tokens based on the model type."""
        if self.is_openai:
            return "max_tokens"
        else:
            return "max_length"

    @property
    def supported_parameters(self):
        """Returns a list of supported parameters based on the API type."""
        if not self.is_openai:
            # koboldcpp united api

            return [
                ParameterReroute(
                    talemate_parameter="max_tokens", client_parameter="max_length"
                ),
                "max_context_length",
                ParameterReroute(
                    talemate_parameter="repetition_penalty", client_parameter="rep_pen"
                ),
                ParameterReroute(
                    talemate_parameter="repetition_penalty_range",
                    client_parameter="rep_pen_range",
                ),
                "top_p",
                "top_k",
                ParameterReroute(
                    talemate_parameter="stopping_strings",
                    client_parameter="stop_sequence",
                ),
                "xtc_threshold",
                "xtc_probability",
                "dry_multiplier",
                "dry_base",
                "dry_allowed_length",
                "dry_sequence_breakers",
                "smoothing_factor",
                "temperature",
            ]

        else:
            # openai api

            return [
                "max_tokens",
                "presence_penalty",
                "top_p",
                "temperature",
            ]

    @property
    def supports_embeddings(self) -> bool:
        """Indicates if embeddings are supported."""
        """Indicates if embeddings are supported."""
        return True

    @property
    def embeddings_url(self) -> str:
        """Get the URL for embeddings based on the API type."""
        if self.is_openai:
            return urljoin(self.api_url, "embeddings")
        else:
            return urljoin(self.api_url, "api/extra/embeddings")

    @property
    def embeddings_function(self):
        """Get the KoboldEmbeddingFunction instance."""
        """Get the KoboldEmbeddingFunction instance."""
        return KoboldEmbeddingFunction(self.embeddings_url, self.embeddings_model_name)

    @property
    def default_prompt_template(self) -> str:
        """Return the default prompt template string."""
        """Return the default prompt template string."""
        return "KoboldAI.jinja2"

    @property
    def api_url(self) -> str:
        """Return the API URL from the client configuration."""
        """Return the API URL from the client configuration."""
        return self.client_config.api_url

    @api_url.setter
    def api_url(self, value: str):
        """Set the API URL in the client configuration."""
        self.client_config.api_url = value

    def api_endpoint_specified(self, url: str) -> bool:
        """Check if the API URL contains the specified endpoint."""
        return "/v1" in self.api_url

    def ensure_api_endpoint_specified(self):
        """Ensure the API endpoint is specified in the API URL."""
        if not self.api_endpoint_specified(self.api_url):
            # url doesn't specify the api endpoint
            # use the koboldcpp united api
            self.api_url = urljoin(self.api_url.rstrip("/") + "/", "/api/v1/")
        if not self.api_url.endswith("/"):
            self.api_url += "/"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ensure_api_endpoint_specified()

    async def get_embeddings_model_name(self):
        # if self._embeddings_model_name is set, return it
        """Retrieve the embeddings model name, fetching it if not already set."""
        if self.embeddings_model_name:
            return self.embeddings_model_name

        # otherwise, get the model name by doing a request to
        # the embeddings endpoint with a single character

        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.embeddings_url,
                json={"input": ["test"]},
                timeout=2,
                headers=self.request_headers,
            )

        response_data = response.json()
        self._embeddings_model_name = response_data.get("model")
        return self._embeddings_model_name

    async def get_embeddings_status(self):
        """Fetch and update the embeddings status from the API."""
        url_version = urljoin(self.api_url, "api/extra/version")
        async with httpx.AsyncClient() as client:
            response = await client.get(url_version, timeout=2)
            response_data = response.json()
            self._embeddings_status = response_data.get("embeddings", False)

            if not self.embeddings_status or self.embeddings_model_name:
                return

            await self.get_embeddings_model_name()

            log.debug(
                "KoboldCpp embeddings are enabled, suggesting embeddings",
                model_name=self.embeddings_model_name,
            )

            await self.set_embeddings()

            emission = ClientEmbeddingsStatus(
                client=self,
                embedding_name=self.embeddings_model_name,
            )

            await async_signals.get("client.embeddings_available").send(emission)

            if not emission.seen:
                # the suggestion has not been seen by the memory agent
                # yet, so we unset the embeddings model name so it will
                # get suggested again
                self._embeddings_model_name = None

    async def get_model_name(self):
        """Retrieve the model name from the specified API endpoint.
        
        This asynchronous function ensures that the API endpoint is specified before
        making a GET request to retrieve model information. It handles potential
        exceptions during the request and checks for a 404 status code, raising a
        KeyError if the model info cannot be found. Depending on whether the request is
        for OpenAI or another service, it extracts the model name from the response
        data and processes it accordingly. Finally, it calls `get_embeddings_status` to
        update the embeddings status before returning the model name.
        """
        self.ensure_api_endpoint_specified()

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    self.api_url_for_model,
                    timeout=2,
                    headers=self.request_headers,
                )
        except Exception:
            self._embeddings_model_name = None
            raise

        if response.status_code == 404:
            raise KeyError(f"Could not find model info at: {self.api_url_for_model}")

        response_data = response.json()
        if self.is_openai:
            # {"object": "list", "data": [{"id": "koboldcpp/dolphin-2.8-mistral-7b", "object": "model", "created": 1, "owned_by": "koboldcpp", "permission": [], "root": "koboldcpp"}]}
            model_name = response_data.get("data")[0].get("id")
        else:
            # {"result": "koboldcpp/dolphin-2.8-mistral-7b"}
            model_name = response_data.get("result")

        # split by "/" and take last
        if model_name:
            model_name = model_name.split("/")[-1]

        await self.get_embeddings_status()

        return model_name

    async def tokencount(self, content: str) -> int:

        # extract scheme and host from api url

        """Counts tokens for the given content using the tokencount endpoint."""
        parts = urlparse(self.api_url)

        url_tokencount = f"{parts.scheme}://{parts.netloc}/api/extra/tokencount"

        async with httpx.AsyncClient() as client:
            response = await client.post(
                url_tokencount,
                json={"prompt": content},
                timeout=None,
                headers=self.request_headers,
            )

            if response.status_code == 404:
                # kobold united doesn't have tokencount endpoint
                return util.count_tokens(content)

            tokencount = len(response.json().get("ids", []))
            return tokencount
        """Trigger the stop generation endpoint."""
        if self.is_openai:
            # openai api endpoint doesn't support abort
            return

        parts = urlparse(self.api_url)
        url_abort = f"{parts.scheme}://{parts.netloc}/api/extra/abort"
        async with httpx.AsyncClient() as client:
            await client.post(
                url_abort,
                headers=self.request_headers,
            )
        """Generates text based on the provided prompt and parameters."""
        if self.is_openai:
            return await self._generate_openai(prompt, parameters, kind)
        else:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, self._generate_kcpp_stream, prompt, parameters, kind
            )
        """Generates a streaming response from the API based on the prompt and parameters."""
        parameters["prompt"] = prompt.strip(" ")

        response = ""
        parameters["stream"] = True
        stream_response = requests.post(
            self.api_url_for_generation,
            json=parameters,
            timeout=None,
            headers=self.request_headers,
            stream=True,
        )
        stream_response.raise_for_status()

        sse = sseclient.SSEClient(stream_response)

        for event in sse.events():
            payload = json.loads(event.data)
            chunk = payload["token"]
            response += chunk
            self.update_request_tokens(self.count_tokens(chunk))

        return response
        """Generates text from the given prompt and parameters."""
        """Generates text from the given prompt and parameters."""
        parameters["prompt"] = prompt.strip(" ")

        self._returned_prompt_tokens = await self.tokencount(parameters["prompt"])

        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.api_url_for_generation,
                json=parameters,
                timeout=None,
                headers=self.request_headers,
            )
            response_data = response.json()
            try:
                if self.is_openai:
                    response_text = response_data["choices"][0]["text"]
                else:
                    response_text = response_data["results"][0]["text"]
            except (TypeError, KeyError) as exc:
                log.error(
                    "Failed to generate text",
                    exc=exc,
                    response_data=response_data,
                    response_status=response.status_code,
                )
                response_text = ""

            self._returned_response_tokens = await self.tokencount(response_text)
            return response_text

    def jiggle_randomness(self, prompt_config: dict, offset: float = 0.3) -> dict:

        """Adjusts temperature and repetition penalty by random values.
        
        This function modifies the 'temperature' and a repetition penalty  in the
        """
        """Adjusts temperature and repetition penalty by random values.
        
        This function modifies the 'temperature' and a repetition penalty  in the
        provided prompt_config dictionary. It uses a base value  as a center and
        applies a random offset within specified limits.  The function determines which
        repetition penalty key to use and  adjusts its value accordingly, ensuring that
        the changes are  within a defined range to maintain variability in the output.
        """
        temp = prompt_config["temperature"]

        if "rep_pen" in prompt_config:
            rep_pen_key = "rep_pen"
        elif "presence_penalty" in prompt_config:
            rep_pen_key = "presence_penalty"
        else:
            rep_pen_key = "repetition_penalty"

        min_offset = offset * 0.3

        prompt_config["temperature"] = random.uniform(temp + min_offset, temp + offset)
        try:
            if rep_pen_key == "presence_penalty":
                presence_penalty = prompt_config["presence_penalty"]
                prompt_config["presence_penalty"] = round(
                    random.uniform(presence_penalty + 0.1, presence_penalty + offset), 1
                )
            else:
                rep_pen = prompt_config[rep_pen_key]
                prompt_config[rep_pen_key] = random.uniform(
                    rep_pen + min_offset * 0.3, rep_pen + offset * 0.3
                )
        except KeyError:
            pass

    async def visual_automatic1111_setup(self, visual_agent: "VisualBase") -> bool:

        """Automatically configure the visual agent for automatic1111.
        
        This function checks the connection status and attempts to fetch the available
        SD models from the koboldcpp server. If successful, it retrieves the model name
        and updates the visual agent's configuration with the appropriate API URL.  The
        function ensures that the visual agent is enabled if the setup is completed
        successfully.
        """
        if not self.connected:
            return False

        sd_models_url = urljoin(self.url, "/sdapi/v1/sd-models")

        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(url=sd_models_url, timeout=2)
            except Exception as exc:
                log.error(f"Failed to fetch sd models from {sd_models_url}", exc=exc)
                return False

            if response.status_code != 200:
                return False

            response_data = response.json()

            sd_model = response_data[0].get("model_name") if response_data else None

        if not sd_model:
            return False

        log.info("KoboldCpp AUTOMATIC1111 setup", sd_model=sd_model)

        visual_agent.actions["automatic1111"].config["api_url"].value = self.url
        visual_agent.is_enabled = True
        return True
