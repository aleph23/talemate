import pydantic
import structlog
from cohere import AsyncClientV2

from talemate.client.base import (
    ClientBase,
    ErrorAction,
    ParameterReroute,
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
from talemate.util import count_tokens

__all__ = [
    "CohereClient",
]
log = structlog.get_logger("talemate")

# Edit this to add new models / remove old models
SUPPORTED_MODELS = [
    "command",
    "command-light",
    "command-r",
    "command-r-plus",
    "command-r-plus-08-2024",
    "command-r7b-12-2024",
    "command-a-03-2025",
]


class Defaults(EndpointOverride, CommonDefaults, pydantic.BaseModel):
    max_token_length: int = 16384
    model: str = "command-r-plus"


class ClientConfig(EndpointOverride, BaseClientConfig):
    pass


@register()
class CohereClient(EndpointOverrideMixin, ClientBase):
    """
    Cohere client for generating text.
    """

    client_type = "cohere"
    conversation_retries = 0
    decensor_enabled = True
    config_cls = ClientConfig

    class Meta(ClientBase.Meta):
        name_prefix: str = "Cohere"
        title: str = "Cohere"
        manual_model: bool = True
        manual_model_choices: list[str] = SUPPORTED_MODELS
        requires_prompt_template: bool = False
        extra_fields: dict[str, ExtraField] = endpoint_override_extra_fields()
        defaults: Defaults = Defaults()

    @property
    def cohere_api_key(self):
        return self.config.cohere.api_key

    @property
    def supported_parameters(self):
        """Return a list of supported parameters."""
        return [
            "temperature",
            ParameterReroute(talemate_parameter="top_p", client_parameter="p"),
            ParameterReroute(talemate_parameter="top_k", client_parameter="k"),
            ParameterReroute(
                talemate_parameter="stopping_strings", client_parameter="stop_sequences"
            ),
            "frequency_penalty",
            "presence_penalty",
            "max_tokens",
        ]

    def emit_status(self, processing: bool = None):
        """Emit the current status of the processing state.
        
        This function updates the processing state based on the provided  argument and
        emits the current status to the client. It checks  for the presence of a valid
        API key and model name, setting the  status accordingly. If there are errors,
        it prepares an error  action and message to be included in the emitted data.
        
        Args:
            processing (bool?): Indicates whether processing is
                currently happening. If None, the current processing state
                is not changed.
        """
        """Emit the current status of the processing state.
        
        This function updates the processing state based on the provided  argument and
        emits the current status to the client. It checks  for the presence of a valid
        API key and model name, setting the  status accordingly. If there are errors,
        it prepares an error  action and message to be included in the emitted data.
        
        Args:
            processing (bool?): Indicates whether processing is
                currently happening. If None, the current processing state
                is retained.
        """
        error_action = None
        error_message = None
        if processing is not None:
            self.processing = processing

        if self.cohere_api_key:
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
                    "cohere_api",
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
        """Count the tokens in the given response string."""
        return count_tokens(response)

    def prompt_tokens(self, prompt: str):
        """Returns the token count for the given prompt."""
        return count_tokens(prompt)

    async def status(self):
        """Emit the current status."""
        self.emit_status()

    def clean_prompt_parameters(self, parameters: dict):
        """Cleans and validates prompt parameters.
        
        This method first calls the superclass's clean_prompt_parameters method to
        handle any initial cleaning. It then enforces specific constraints on the
        parameters:  the "temperature" value is clamped between 0.0 and 1.0, the
        "stop_sequences"  list is limited to a maximum of 5 items, and if both
        "frequency_penalty"  and "presence_penalty" are provided, the
        "frequency_penalty" is removed.
        """
        """Cleans and validates prompt parameters.
        
        This method first calls the superclass's clean_prompt_parameters method to
        handle any initial cleaning. It then enforces specific constraints on the
        parameters:  the "temperature" value is clamped between 0.0 and 1.0, the
        "stop_sequences"  list is limited to a maximum of 5 items, and if both
        "frequency_penalty"  and "presence_penalty" are provided, the
        "frequency_penalty" is removed.
        """
        super().clean_prompt_parameters(parameters)

        # if temperature is set, it needs to be clamped between 0 and 1.0
        if "temperature" in parameters:
            parameters["temperature"] = max(0.0, min(1.0, parameters["temperature"]))

        # if stop_sequences is set, max 5 items
        if "stop_sequences" in parameters:
            parameters["stop_sequences"] = parameters["stop_sequences"][:5]

        # if both frequency_penalty and presence_penalty are set, drop frequency_penalty
        if "presence_penalty" in parameters and "frequency_penalty" in parameters:
            del parameters["frequency_penalty"]

    async def generate(self, prompt: str, parameters: dict, kind: str):

        """Generates text from the given prompt and parameters.
        
        This asynchronous function utilizes the Cohere API to generate a response based
        on the provided prompt and parameters. It first checks for the necessary API
        key and initializes an asynchronous client. The function constructs messages
        """
        """Generates text from the given prompt and parameters.
        
        This asynchronous function utilizes the Cohere API to generate a response based
        on the provided prompt and parameters. It first checks for the necessary API
        key and initializes an asynchronous client. The function constructs messages
        for the system and user roles, then streams the response from the API. As the
        response is received, it accumulates the generated text and tracks token usage
        for the request. Finally, it returns the complete generated response.
        """
        if not self.cohere_api_key and not self.endpoint_override_base_url_configured:
            raise Exception("No cohere API key set")

        client = AsyncClientV2(self.api_key, base_url=self.base_url)

        human_message = prompt.strip()
        system_message = self.get_system_message(kind)

        self.log.debug(
            "generate",
            prompt=prompt[:128] + " ...",
            parameters=parameters,
            system_message=system_message,
        )

        messages = [
            {
                "role": "system",
                "content": system_message,
            },
            {
                "role": "user",
                "content": human_message,
            },
        ]

        try:
            # Cohere's `chat_stream` returns an **asynchronous generator** that can be
            # consumed directly with `async for`. It is not an asynchronous context
            # manager, so attempting to use `async with` raises a `TypeError` as seen
            # in issue logs. We therefore iterate over the generator directly.

            stream = client.chat_stream(
                model=self.model_name,
                messages=messages,
                **parameters,
            )

            response = ""

            async for event in stream:
                if event and event.type == "content-delta":
                    chunk = event.delta.message.content.text
                    response += chunk
                    # Track token usage incrementally
                    self.update_request_tokens(self.count_tokens(chunk))

            self._returned_prompt_tokens = self.prompt_tokens(prompt)
            self._returned_response_tokens = self.response_tokens(response)

            log.debug("generated response", response=response)

            return response
        # except PermissionDeniedError as e:
        #    self.log.error("generate error", e=e)
        #    emit("status", message="cohere API: Permission Denied", status="error")
        #    return ""
        except Exception:
            raise
