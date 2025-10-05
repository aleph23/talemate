import pydantic
import structlog
from mistralai import Mistral
from mistralai.models.sdkerror import SDKError

from talemate.client.base import (
    ClientBase,
    ErrorAction,
    CommonDefaults,
    ExtraField,
)
from talemate.client.registry import register
from talemate.client.remote import (
    EndpointOverride,
    EndpointOverrideMixin,
    endpoint_override_extra_fields,
)
from talemate.config.schema import Client as BaseClientConfig
from talemate.emit import emit

__all__ = [
    "MistralAIClient",
]
log = structlog.get_logger("talemate")

# Edit this to add new models / remove old models
SUPPORTED_MODELS = [
    "open-mistral-7b",
    "open-mixtral-8x7b",
    "open-mixtral-8x22b",
    "open-mistral-nemo",
    "mistral-small-latest",
    "mistral-medium-latest",
    "mistral-large-latest",
    "magistral-medium-2506",
]


class Defaults(EndpointOverride, CommonDefaults, pydantic.BaseModel):
    max_token_length: int = 16384
    model: str = "open-mixtral-8x22b"


class ClientConfig(EndpointOverride, BaseClientConfig):
    pass


@register()
class MistralAIClient(EndpointOverrideMixin, ClientBase):
    """
    OpenAI client for generating text.
    """

    client_type = "mistral"
    conversation_retries = 0
    # TODO: make this configurable?
    decensor_enabled = True
    config_cls = ClientConfig

    class Meta(ClientBase.Meta):
        name_prefix: str = "MistralAI"
        title: str = "MistralAI"
        manual_model: bool = True
        manual_model_choices: list[str] = SUPPORTED_MODELS
        requires_prompt_template: bool = False
        defaults: Defaults = Defaults()
        extra_fields: dict[str, ExtraField] = endpoint_override_extra_fields()

    @property
    def can_be_coerced(self) -> bool:
        """Indicates if coercion is possible based on reason_enabled."""
        return not self.reason_enabled

    @property
    def mistral_api_key(self):
        """Get the Mistral API key from the configuration."""
        return self.config.mistralai.api_key

    @property
    def supported_parameters(self):
        """Return a list of supported parameters."""
        return [
            "temperature",
            "top_p",
            "max_tokens",
        ]

    def emit_status(self, processing: bool = None):
        """Emit the current status of the client.
        
        This function updates the processing state and determines the current status
        based on the presence of an API key and a loaded model. If the API key is
        missing, it prepares an error action to prompt the user to set the key.
        Additionally, it gathers common status data and emits a message with the
        current status, including any relevant error messages.
        
        Args:
            processing (bool?): Indicates whether the client is currently
        """
        """Emit the current status of the client.
        
        This function updates the processing state and determines the current status
        based on the presence of an API key and a loaded model. If the API key is
        missing, it prepares an error action to prompt the user to set the key.
        Additionally, if no model is loaded, it sets the status to an error. Finally,
        it emits the status along with relevant data to the client.
        
        Args:
            processing (bool?): Indicates whether the client is currently
                processing. If provided, it updates the internal processing state.
        """
        error_action = None
        error_message = None
        if processing is not None:
            self.processing = processing

        if self.mistral_api_key:
            status = "busy" if self.processing else "idle"
        else:
            status = "error"
            error_message = "No API key set"
            error_action = ErrorAction(
                title="Set API Key",
                action_name="openAppConfig",
                icon="mdi-key-variant",
                arguments=[
                    "application",
                    "mistralai_api",
                ],
            )

        if not self.model_name:
            status = "error"
            error_message = "No model loaded"

        self.current_status = status
        data = {
            "error_action": error_action.model_dump() if error_action else None,
            "meta": self.Meta().model_dump(),
            "enabled": self.enabled,
            "error_message": error_message,
        }
        data.update(self._common_status_data())
        emit(
            "client_status",
            message=self.client_type,
            id=self.name,
            details=self.model_name,
            status=status if self.enabled else "disabled",
            data=data,
        )

    def response_tokens(self, response: str):
        """Returns the completion tokens from the response usage."""
        return response.usage.completion_tokens

    def prompt_tokens(self, response: str):
        """Returns the prompt tokens from the given response."""
        return response.usage.prompt_tokens

    async def status(self):
        """Emit the current status."""
        self.emit_status()

    def clean_prompt_parameters(self, parameters: dict):
        """Cleans and clamps the temperature parameter in the given dictionary."""
        super().clean_prompt_parameters(parameters)
        # clamp temperature to 0.1 and 1.0
        # Unhandled Error: Status: 422. Message: {"object":"error","message":{"detail":[{"type":"less_than_equal","loc":["body","temperature"],"msg":"Input should be less than or equal to 1","input":1.31,"ctx":{"le":1.0},"url":"https://errors.pydantic.dev/2.6/v/less_than_equal"}]},"type":"invalid_request_error","param":null,"code":null}
        if "temperature" in parameters:
            parameters["temperature"] = min(1.0, max(0.1, parameters["temperature"]))

    async def generate(self, prompt: str, parameters: dict, kind: str):

        """Generate text from a given prompt and parameters using the Mistral API.
        
        This asynchronous function constructs a message payload based on the provided
        prompt and parameters, including optional coercion prompts if applicable. It
        initializes a client for the Mistral API and streams the response from the chat
        model, handling token counts for both the prompt and the completion. The
        function also includes error handling for API-related issues and logs relevant
        information throughout the process.
        
        Args:
            prompt (str): The input text prompt to generate a response for.
            parameters (dict): Additional parameters to customize the generation process.
            kind (str): The type of message or context for the generation.
        
        Returns:
            str: The generated text response from the Mistral API.
        
        """
        """Generate text from the given prompt and parameters using the Mistral API.
        
        This asynchronous function constructs a message payload based on the provided
        prompt and parameters, including optional coercion prompts if applicable. It
        initializes a Mistral client and streams the response from the API, handling
        token counts and logging relevant information throughout the process. In case
        of errors, it logs the issue and manages specific SDK errors related to API
        permissions.
        
        Args:
            prompt (str): The input text prompt to generate a response for.
            parameters (dict): Additional parameters for the API request.
            kind (str): The type of message or context for the generation.
        
        Returns:
            str: The generated text response from the Mistral API.
        
        Raises:
            Exception: If no Mistral API key is set or for other unexpected errors.
        """
        if not self.mistral_api_key:
            raise Exception("No mistral.ai API key set")

        client = Mistral(api_key=self.api_key, server_url=self.base_url)

        if self.can_be_coerced:
            prompt, coercion_prompt = self.split_prompt_for_coercion(prompt)
        else:
            coercion_prompt = None

        system_message = self.get_system_message(kind)

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt.strip()},
        ]

        if coercion_prompt:
            log.debug("Adding coercion pre-fill", coercion_prompt=coercion_prompt)
            messages.append(
                {
                    "role": "assistant",
                    "content": coercion_prompt.strip(),
                    "prefix": True,
                }
            )

        self.log.debug(
            "generate",
            base_url=self.base_url,
            prompt=prompt[:128] + " ...",
            parameters=parameters,
            system_message=system_message,
        )

        try:
            event_stream = await client.chat.stream_async(
                model=self.model_name,
                messages=messages,
                **parameters,
            )

            response = ""

            completion_tokens = 0
            prompt_tokens = 0

            async for event in event_stream:
                if event.data.choices:
                    response += event.data.choices[0].delta.content
                    self.update_request_tokens(
                        self.count_tokens(event.data.choices[0].delta.content)
                    )
                if event.data.usage:
                    completion_tokens += event.data.usage.completion_tokens
                    prompt_tokens += event.data.usage.prompt_tokens

            self._returned_prompt_tokens = prompt_tokens
            self._returned_response_tokens = completion_tokens

            return response
        except SDKError as e:
            self.log.error("generate error", e=e)
            if hasattr(e, "status_code") and e.status_code in [403, 401]:
                emit(
                    "status",
                    message="mistral.ai API: Permission Denied",
                    status="error",
                )
            return ""
        except Exception:
            raise
