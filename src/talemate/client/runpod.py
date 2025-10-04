"""
Retrieve pod information from the server which can then be used to bootstrap talemate client
connection for the pod.  This is a simple wrapper around the runpod module.
"""

import asyncio

import dotenv
import runpod
import structlog

from talemate.config import get_config

from .bootstrap import ClientBootstrap, ClientType, register_list

log = structlog.get_logger("talemate.client.runpod")

dotenv.load_dotenv()


TEXTGEN_IDENTIFIERS = ["textgen", "thebloke llms", "text-generation-webui"]


def is_textgen_pod(pod):
    """Check if the given pod is a text generation pod."""
    name = pod["name"].lower()

    if any(identifier in name for identifier in TEXTGEN_IDENTIFIERS):
        return True

    return False


async def _async_get_pods():
    """Asynchronously retrieves pods using runpod API."""
    runpod.api_key = get_config().runpod.api_key

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, runpod.get_pods)


async def get_textgen_pods():
    """Return a list of text generation pods.
    
    This asynchronous function retrieves text generation pods from the  runpod
    service. It first sets the API key using the configuration  obtained from
    `get_config()`. If the API key is not available, the  function exits early. It
    then iterates through the pods obtained  from `_async_get_pods()`, yielding
    only those pods that are in  the "RUNNING" state and identified as text
    generation pods by  the `is_textgen_pod()` function.
    """
    runpod.api_key = get_config().runpod.api_key

    if not runpod.api_key:
        return

    for pod in await _async_get_pods():
        if not pod["desiredStatus"] == "RUNNING":
            continue
        if is_textgen_pod(pod):
            yield pod


async def get_automatic1111_pods():

    """Return a list of automatic1111 pods.
    
    This asynchronous function retrieves pods from the runpod service.  It first
    checks for a valid API key from the configuration. If the  API key is not
    present, the function exits early. The function then  iterates through the
    retrieved pods, yielding only those that are  in the "RUNNING" state and
    contain "automatic1111" in their name.
    """
    runpod.api_key = get_config().runpod.api_key

    if not runpod.api_key:
        return

    for pod in await _async_get_pods():
        if not pod["desiredStatus"] == "RUNNING":
            continue
        if "automatic1111" in pod["name"].lower():
            yield pod


def _client_bootstrap(client_type: ClientType, pod):
    """
    Return a client bootstrap object for the given client type and pod.
    """

    id = pod["id"]

    if client_type == ClientType.textgen:
        api_url = f"https://{id}-5000.proxy.runpod.net"
    elif client_type == ClientType.automatic1111:
        api_url = f"https://{id}-5000.proxy.runpod.net"

    return ClientBootstrap(
        client_type=client_type,
        uid=pod["id"],
        name=pod["name"],
        api_url=api_url,
        service_name="runpod",
    )


@register_list("runpod")
async def client_bootstrap_list():
    """Return a list of client bootstrap options.
    
    This function asynchronously retrieves client bootstrap options from two
    different sources: text generation pods and automatic1111 pods. It first
    collects the pods from each source using the `get_textgen_pods` and
    `get_automatic1111_pods` functions. Then, it yields the bootstrap options  for
    each pod by calling the `_client_bootstrap` function with the appropriate
    `ClientType`.
    """
    textgen_pods = []
    async for pod in get_textgen_pods():
        textgen_pods.append(pod)

    automatic1111_pods = []
    async for pod in get_automatic1111_pods():
        automatic1111_pods.append(pod)

    for pod in textgen_pods:
        yield _client_bootstrap(ClientType.textgen, pod)

    for pod in automatic1111_pods:
        yield _client_bootstrap(ClientType.automatic1111, pod)
