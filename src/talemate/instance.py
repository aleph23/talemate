"""
Keep track of clients and agents
"""

import asyncio

import structlog

import talemate.agents as agents
import talemate.client as clients
import talemate.client.bootstrap as bootstrap
from talemate.client.base import ClientStatus
from talemate.emit import emit
from talemate.emit.signals import handlers
import talemate.emit.async_signals as async_signals
from talemate.config import get_config, Config

log = structlog.get_logger("talemate")

AGENTS = {}
CLIENTS = {}


def get_agent(typ: str):
    agent = AGENTS.get(typ)

    if not agent:
        raise KeyError(f"Agent {typ} has not been instantiated")

    return agent


async def destroy_client(name: str):
    """Destroys the client associated with the given name."""
    client = CLIENTS.get(name)
    if client:
        await client.destroy()
        del CLIENTS[name]


def get_client(name: str):
    client = CLIENTS.get(name)

    if not client:
        raise KeyError(f"Client {name} has not been instantiated")

    return client


def agent_types():
    """Return the keys of AGENT_CLASSES from agents."""
    return agents.AGENT_CLASSES.keys()


def client_types():
    """Return the keys of CLIENT_CLASSES from clients."""
    return clients.CLIENT_CLASSES.keys()


def client_instances():
    """Return the items of the CLIENTS dictionary."""
    return CLIENTS.items()


def agent_instances():
    """Return items from the AGENTS dictionary."""
    return AGENTS.items()


def agent_instances_with_client(client):

    """Return agents associated with the specified client."""
    for typ, agent in agent_instances():
        if getattr(agent, "client", None) == client:
            yield agent


def emit_agent_status_by_client(client):

    """Emit status of all agents associated with the specified client."""
    for agent in agent_instances_with_client(client):
        emit_agent_status(agent.__class__, agent)


async def emit_clients_status():
    # log.debug("emit", type="client status")
    """Emit the status of all clients."""
    for client in list(CLIENTS.values()):
        if client:
            await client.status()


def _sync_emit_clients_status(*args, **kwargs):
    """Emit the status of all clients in synchronous mode."""
    loop = asyncio.get_event_loop()
    loop.run_until_complete(emit_clients_status())


handlers["request_client_status"].connect(_sync_emit_clients_status)


async def emit_client_bootstraps():
    """Emit client bootstraps data."""
    emit("client_bootstraps", data=list(await bootstrap.list_all()))


def sync_emit_clients_status():
    """Emit the status of all clients in synchronous mode."""
    loop = asyncio.get_event_loop()
    loop.run_until_complete(emit_clients_status())


async def sync_client_bootstraps():

    """Synchronizes client instances from registered bootstrap lists."""
    for service_name, func in bootstrap.LISTS.items():
        async for client_bootstrap in func():
            log.debug(
                "sync client bootstrap",
                service_name=service_name,
                client_bootstrap=client_bootstrap.dict(),
            )
            client = get_client(
                client_bootstrap.name,
                type=client_bootstrap.client_type.value,
                api_url=client_bootstrap.api_url,
                enabled=True,
            )
            await client.status()


def emit_agent_status(cls, agent=None):
    if not agent:
        emit(
            "agent_status",
            message="",
            id=cls.agent_type,
            status="uninitialized",
            data=cls.config_options(),
        )
    else:
        asyncio.create_task(agent.emit_status())
        # loop = asyncio.get_event_loop()
        # loop.run_until_complete(agent.emit_status())


def emit_agents_status(*args, **kwargs):
    # log.debug("emit", type="agent status")
    """Emit the status of all agents."""
    for typ, cls in sorted(
        agents.AGENT_CLASSES.items(), key=lambda x: x[1].verbose_name
    ):
        agent = AGENTS.get(typ)
        emit_agent_status(cls, agent)


handlers["request_agent_status"].connect(emit_agents_status)


async def agent_ready_checks():
    for agent in AGENTS.values():
        if agent and agent.enabled:
            await agent.ready_check()
        elif agent and not agent.enabled:
            await agent.setup_check()


def get_active_client():
    """Return the first enabled client from CLIENTS."""
    for client in CLIENTS.values():
        if client.enabled:
            return client
    return None


async def instantiate_agents():
    config: Config = get_config()

    for typ, cls in agents.AGENT_CLASSES.items():
        if typ in AGENTS:
            continue

        agent_config = config.agents.get(typ)
        if agent_config:
            _agent_config = agent_config.model_dump()

            client_name = _agent_config.pop("client", None)
            if client_name:
                _agent_config["client"] = CLIENTS.get(client_name)

            _agent_config.pop("name", None)
            actions = _agent_config.pop("actions", None)
            enabled = _agent_config.pop("enabled", True)

            agent = cls(**_agent_config)

            if actions:
                await agent.apply_config(actions=actions)

            if not enabled and agent.has_toggle:
                agent.is_enabled = False
            elif enabled is True and agent.has_toggle:
                agent.is_enabled = True

            AGENTS[typ] = agent
            await agent.emit_status()
        else:
            agent = cls()
            AGENTS[typ] = agent
            await agent.emit_status()

    await ensure_agent_llm_client()


async def instantiate_clients():
    config: Config = get_config()
    for name, client_config in config.clients.items():
        if name in CLIENTS:
            continue

        client = clients.get_client_class(client_config.type)(
            **client_config.model_dump()
        )
        CLIENTS[name] = client

    await emit_clients_status()


async def configure_agents():
    """Configures agents based on the provided configuration."""
    config: Config = get_config()
    for name, agent_config in config.agents.items():
        agent = AGENTS.get(name)
        if not agent:
            log.warn("agent not found", name=name)
            continue

        await agent.apply_config(**agent_config.model_dump())
        await agent.emit_status()

    await ensure_agent_llm_client()


async def ensure_agent_llm_client():
    config: Config = get_config()
    for name, agent in AGENTS.items():
        agent_config = config.agents.get(name)

        if not agent:
            log.warn("agent not found", name=name)
            continue

        if not agent.requires_llm_client:
            continue

        client_name = agent_config.client if agent_config else None

        if not client_name:
            client = get_active_client()

        elif not CLIENTS.get(client_name):
            client = get_active_client()

        else:
            client = CLIENTS.get(client_name)
            if client and not client.enabled:
                client = get_active_client()

        log.debug(
            "ensure_agent_llm_client",
            agent=agent.agent_type,
            client=client.client_type if client else None,
        )

        if agent.client != client:
            agent.client = client
            await agent.emit_status()


async def purge_clients():
    """Checks for clients in CLIENTS that are not longer in the config
    and removes them
    """
    config: Config = get_config()
    for name, _ in list(CLIENTS.items()):
        if name in config.clients:
            continue
        await destroy_client(name)


async def on_config_changed(config: Config):
    """Handle changes in the configuration."""
    await emit_clients_status()
    emit_agents_status()


async def on_client_disabled(client_status: ClientStatus):
    await ensure_agent_llm_client()


async def on_client_enabled(client_status: ClientStatus):
    """Handles the event when the client is enabled."""
    await ensure_agent_llm_client()


async_signals.get("config.changed").connect(on_config_changed)
async_signals.get("client.disabled").connect(on_client_disabled)
async_signals.get("client.enabled").connect(on_client_enabled)
