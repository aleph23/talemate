import pydantic
import uuid
from typing import Any, Callable, ClassVar, Annotated
import networkx as nx
import contextvars
import asyncio
import structlog
import traceback
import json
import time
import reprlib
import re
from enum import IntEnum

from talemate.game.engine.nodes.base_types import base_node_type, BASE_TYPES
from talemate.game.engine.nodes.registry import get_node, register
from talemate.exceptions import (
    ExitScene,
    ResetScene,
    RestartSceneLoop,
    ActedAsCharacter,
    GenerationCancelled,
)
import talemate.emit.async_signals as async_signals
from talemate.util.async_tools import shared_debounce
from talemate.context import active_scene

log = structlog.get_logger("talemate.game.engine.nodes.core")

graph_state = contextvars.ContextVar("graph_state")
save_state = contextvars.ContextVar("save_state")

PYTHON_TYPE_TO_STRING = {
    "<class 'str'>": "str",
    "<class 'int'>": "int",
    "<class 'float'>": "float",
    "<class 'bool'>": "bool",
    "<class 'list'>": "list",
    "<class 'dict'>": "dict",
    "<class 'NoneType'>": "None",
}

TYPE_CHOICES = sorted(
    [
        "str",
        "int",
        "float",
        "bool",
        "list",
        "dict",
        "any",
        "character",
        "interaction_state",
        "actor",
        "event",
        "client",
        "agent",
        "function",
    ]
)

TYPE_TO_CLASS = {
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "list": list,
    "dict": dict,
    "any": Any,
}


def get_type_class(type_str: str) -> Any:
    if TYPE_TO_CLASS.get(type_str):
        return TYPE_TO_CLASS[type_str]
    raise ValueError(f"Could not find class for type {type_str}")


class LoopContinue(Exception):
    pass


class LoopBreak(Exception):
    pass


class LoopExit(Exception):
    pass


class StopModule(Exception):
    pass


class StopGraphExecution(Exception):
    pass


class ModuleError(Exception):
    pass


PASSTHROUGH_ERRORS = (
    ExitScene,
    ResetScene,
    RestartSceneLoop,
    ActedAsCharacter,
    LoopContinue,
    LoopBreak,
    LoopExit,
    GenerationCancelled,
)


class UNRESOLVED:
    def __bool__(self):
        return False

    def __str__(self):
        return "<UNRESOLVED>"

    def __repr__(self):
        return "<UNRESOLVED>"


class NodeVerbosity(IntEnum):
    SILENT = 0
    NORMAL = 1
    VERBOSE = 2


async_signals.register("nodes_node_state")


class InputValueError(ValueError):
    def __init__(self, node: "Node", input_name: str, message: str):
        self.node = node
        self.input_name = input_name
        super().__init__(f"Error in node {node.title} input {input_name}: {message}")


def load_extended_components(file_path: str, node_data: dict):

    """Loads all extended components from a file.
    
    This function reads a JSON file specified by file_path and loads its  contents
    into the node_data dictionary. It supports inheritance by  recursively loading
    components from any specified "extends" key. The  function populates nodes,
    edges, groups, and comments, marking them  as inherited where applicable. Debug
    logs are generated before and  after the loading process to track the
    operation.
    
    Args:
        file_path (str): The path to the JSON file containing extended components.
        node_data (dict): The dictionary to which the loaded components will be added.
    """
    with open(file_path, "r") as f:
        data = json.load(f)

    log.debug("loading extended components", file_path=file_path)

    if data.get("extends"):
        load_extended_components(data["extends"], node_data)

    for node_id, node in data.get("nodes", {}).items():
        if node_id not in node_data.get("nodes"):
            node_data["nodes"][node_id] = node
            node_data["nodes"][node_id]["inherited"] = True

    for edge in data.get("edges", []):
        node_data["edges"][edge] = data["edges"][edge]

    for group in data.get("groups", []):
        group["inherited"] = True
        node_data["groups"].append(group)

    for comment in data.get("comments", []):
        comment["inherited"] = True
        node_data["comments"].append(comment)

    log.debug("loaded extended components", file_path=file_path)


def dynamic_node_import(
    node_data: dict, registry_name: str, registry_container: dict | None = None
) -> "Graph | Loop":

    """Dynamically import or create a node class from the given data."""
    base_type = node_data.get("base_type")
    node_cls = BASE_TYPES.get(base_type)

    if not node_cls:
        raise ValueError(
            f"Cannont import node data with base type {node_data.get('base_type')}"
        )

    if node_data.get("extends"):
        log.debug("loading extended components", extends=node_data["extends"])
        load_extended_components(node_data["extends"], node_data)

    @register(registry_name, container=registry_container)
    class DynamicNode(node_cls):
        def __init__(self, *args, **kwargs):
            node_data_copy = node_data.copy()
            node_data_copy.update(kwargs)
            super().__init__(*args, **node_data_copy)

    DynamicNode.__name__ = registry_name.split("/")[-1]
    DynamicNode.__dynamic_imported__ = True
    DynamicNode._base_type = base_type

    return DynamicNode


def get_ancestors_with_forks(graph: nx.DiGraph, node_id: str) -> set[str]:
    """
    Returns a set of node IDs that are ancestors of the given node,
    plus any nodes in forked branches that don't lead to the target.

    A drop-in replacement for nx.ancestors() with extended functionality.

    Parameters:
    - graph: A NetworkX directed graph
    - node_id: The target node ID

    Returns:
    - A set of node IDs including ancestors and forked paths
    """
    # Get direct ancestors (standard behavior)
    ancestors = nx.ancestors(graph, node_id)

    # Find all forks from ancestors
    forked_nodes = set()
    for ancestor_id in ancestors:
        # For each ancestor, find its descendants that aren't already in ancestors
        descendants = nx.descendants(graph, ancestor_id)
        # Add descendants that aren't ancestors of our target node and aren't the target
        forked_nodes.update(
            desc for desc in descendants if desc not in ancestors and desc != node_id
        )

    # Combine direct ancestors with forked nodes
    return ancestors.union(forked_nodes)


class NodeStyle(pydantic.BaseModel):
    title_color: str | None = None
    node_color: str | None = None
    icon: str | None = None
    auto_title: str | None = None


class NodeState(pydantic.BaseModel):
    node_id: str
    start_time: float | None = pydantic.Field(default_factory=time.time)
    end_time: float | None = None
    deactivated: bool = False
    error: str | None = None

    input_values: dict[str, Any] = pydantic.Field(default_factory=dict)
    output_values: dict[str, Any] = pydantic.Field(default_factory=dict)
    properties: dict[str, Any] = pydantic.Field(default_factory=dict)

    def __init__(self, node: "NodeBase", state: "GraphState", **kwargs):
        super().__init__(node_id=node.id, **kwargs)

        self.input_values = {socket.name: socket.value for socket in node.inputs}
        self.output_values = {socket.name: socket.value for socket in node.outputs}
        self.properties = node.properties.copy()

    def __eq__(self, value) -> bool:
        try:
            return self.node_id == value.node_id
        except AttributeError:
            return False

    def __hash__(self):
        return hash(self.node_id)

    def __lt__(self, value) -> bool:
        if not isinstance(value, NodeState):
            return NotImplemented
        return self.node_id < value.node_id

    def __gt__(self, value) -> bool:
        if not isinstance(value, NodeState):
            return NotImplemented
        return self.node_id > value.node_id

    def __str__(self):
        return f"NodeState {self.node_id}"

    def __repr__(self):
        return f"NodeState {self.node_id}"

    @property
    def flattened(self) -> dict:
        """
        Creates a flattened representation of the node state.
        with repr directly applied to input and output values to avoid circular references.
        """
        r = reprlib.Repr()
        r.maxlevel = 1
        r.maxlist = 10
        r.maxstring = 255

        return {
            "node_id": self.node_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "deactivated": self.deactivated,
            "error": self.error,
            "input_values": {k: r.repr(v) for k, v in self.input_values.items()},
            "output_values": {k: r.repr(v) for k, v in self.output_values.items()},
            "properties": {k: r.repr(v) for k, v in self.properties.items()},
        }


class GraphState(pydantic.BaseModel):
    data: dict[str, Any] = pydantic.Field(default_factory=dict)
    outer: "GraphState | None" = None

    shared: dict[str, Any] = pydantic.Field(default_factory=dict)

    graph: "Graph | None" = None

    stack: list[NodeState] = pydantic.Field(default_factory=list)

    verbosity: NodeVerbosity = NodeVerbosity.NORMAL

    @property
    def flattened(self) -> dict:
        """Return a dictionary representation of the flattened stack."""
        try:
            return {
                "stack": [node_state.flattened for node_state in self.stack],
            }
        except Exception as e:
            log.error(
                "error dumping stack (stack probably contains circular references)",
                error=e,
            )
            self.stack = []
            return {"stack": []}

    def node_property_key(self, node: "NodeBase", name: str) -> str:
        """Generate a property key for a node.
        
        Args:
            node (NodeBase): The node object.
            name (str): The property name.
        
        Returns:
            str: The formatted property key.
        """
        return f"{node.id}.{name}"

    def set_node_property(self, node: "NodeBase", name: str, value: Any):
        """Sets a property value for a given node."""
        self.data[self.node_property_key(node, name)] = value

    def get_node_property(self, node: "NodeBase", name: str) -> Any:
        """Retrieve a property value for a given node."""
        return self.data.get(
            self.node_property_key(node, name), node.properties.get(name, UNRESOLVED)
        )

    def node_socket_value_key(self, node: "NodeBase", socket_name: str) -> str:
        """Generate a unique key for a socket in a node."""
        return f"{node.id}__socket.{socket_name}"

    def set_node_socket_value(self, node: "NodeBase", socket_name: str, value: Any):
        self.data[self.node_socket_value_key(node, socket_name)] = value

    def get_node_socket_value(self, node: "NodeBase", socket_name: str) -> Any:
        return self.data.get(self.node_socket_value_key(node, socket_name), UNRESOLVED)

    def node_socket_state_key(self, node: "NodeBase", socket_name: str) -> str:
        """Generate a key for the deactivated socket state of a node."""
        return f"{node.id}__socket_deactivated.{socket_name}"

    def set_node_socket_state(self, node: "NodeBase", socket_name: str, value: bool):
        """Sets the state of a socket for a given node."""
        self.data[self.node_socket_state_key(node, socket_name)] = value

    def get_node_socket_state(self, node: "NodeBase", socket_name: str) -> bool:
        return self.data.get(self.node_socket_state_key(node, socket_name), False)


class GraphContext:
    def __init__(self, outer_state: GraphState = None, graph: "Graph" = None):
        self.outer_state = outer_state
        self.graph = graph
        self.token = None

    def __enter__(self) -> GraphState:
        state = GraphState(outer=self.outer_state, graph=self.graph)
        state.shared = self.outer_state.shared if self.outer_state else {}
        state.stack = self.outer_state.stack if self.outer_state else []
        self.token = graph_state.set(state)
        return state

    def __exit__(self, exc_type, exc_value, traceback):
        graph_state.reset(self.token)


class SaveContext:
    def __enter__(self):
        self.token = save_state.set(True)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        save_state.reset(self.token)


class Socket(pydantic.BaseModel):
    id: str = pydantic.Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    node: "NodeBase | None" = pydantic.Field(exclude=True, default=None)

    source: "Socket" = pydantic.Field(default=None, exclude=True)
    optional: bool = False
    group: str | None = None

    socket_type: str | list = "any"

    @classmethod
    def as_bool(cls, value: Any) -> bool:
        if value is UNRESOLVED:
            return False
        return bool(value)

    @property
    def value(self):
        """Gets the value from the graph state or returns UNRESOLVED if unavailable."""
        try:
            state: GraphState = graph_state.get()
        except LookupError:
            # we dont have a state, so we can't get the value
            return UNRESOLVED
        if not self.source:
            return state.get_node_socket_value(self.node, self.name)
        return state.get_node_socket_value(self.source.node, self.source.name)

    @value.setter
    def value(self, value):
        try:
            state: GraphState = graph_state.get()
        except LookupError:
            # we dont have a state, so we can't set the value
            return

        state.set_node_socket_value(self.node, self.name, value)

    @property
    def deactivated(self) -> bool:
        """Indicates if the node is deactivated based on its socket state."""
        try:
            state: GraphState = graph_state.get()
        except LookupError:
            # we dont have a state, so we can't get the socket activation state
            return True

        return state.get_node_socket_state(self.node, self.name)

    @deactivated.setter
    def deactivated(self, value: bool):
        """Sets the activation state of a node's socket."""
        try:
            state: GraphState = graph_state.get()
        except LookupError:
            # we dont have a state, so we can't set the socket activation state
            return

        state.set_node_socket_state(self.node, self.name, value)

    @property
    def full_id(self) -> str:
        """Return the full identifier as a string."""
        return f"{self.node.id}.{self.name}"

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return self.id == other.id

    def __str__(self):
        return f"{self.node.title}.{self.name}" if self.node else self.name

    def __repr__(self):
        return str(self)


class PropertyField(pydantic.BaseModel):
    """
    Describe a property field for a node
    """

    name: str
    description: str
    type: str
    default: Any = None
    choices: list[Any] = None
    readonly: bool = False

    step: float | int | None = None
    min: float | int | None = None
    max: float | int | None = None

    # if true value will not be saved in the graph, past the initial value
    ephemeral: bool = False

    generate_choices: Callable | None = pydantic.Field(default=None, exclude=True)

    def model_dump(self, **kwargs):
        """Return a dictionary representation of the model."""
        data = super().model_dump(**kwargs)
        # if generate_choices is set, use it to override choices
        if self.generate_choices:
            data["choices"] = self.generate_choices()
        return data


class NodeBase(pydantic.BaseModel):
    title: str = "Node"
    id: str = pydantic.Field(default_factory=lambda: str(uuid.uuid4()))
    properties: dict[str, Any] = pydantic.Field(default_factory=dict)
    x: int = 0
    y: int = 0
    width: int = 200
    height: int = 100
    collapsed: bool = False
    inherited: bool = False

    registry: str | None = None
    _registry: ClassVar[str | None] = None

    _export_definition: ClassVar[bool] = True

    _isolated: ClassVar[bool] = False

    _base_type: ClassVar[str] = ""

    _module_path: ClassVar[str] = ""

    @pydantic.computed_field(description="Base type")
    @property
    def base_type(self) -> str:
        """Return the base type as a string."""
        return self._base_type

    @property
    def field_definitions(self) -> dict[str, PropertyField]:
        """
        Returns a dictionary of property field definitions.
        
        This method iterates through the properties of the instance and retrieves the
        corresponding PropertyField for each property using the  get_property_field
        method. Additionally, if the class has a Fields  object, it checks for any
        remaining fields that are instances of  PropertyField and adds them to the
        dictionary, ensuring no duplicates  are included.
        """
        fields = {}

        for name, value in self.properties.items():
            fields[name] = self.get_property_field(name)

        # if the class has a Fields object, add any remaining fields from that
        if hasattr(self.__class__, "Fields"):
            for name, value in self.__class__.Fields.__dict__.items():
                # if the field is a PropertyField and it's not already in the fields dictionary, add it
                if isinstance(value, PropertyField) and name not in fields:
                    fields[name] = value

        return fields

    def __init__(self, *args, **kwargs):
        if kwargs.get("title", "Node") == "Node":
            title = self.__class__.__name__
            # replace camel case with spaces
            title = re.sub(r"(?<!^)(?=[A-Z])", " ", title)
            kwargs["title"] = title

        # never override the registry through kwargs
        kwargs.pop("registry", None)
        properties = kwargs.pop("properties", {})

        super().__init__(registry=self._registry, *args, **kwargs)

        if not self.inputs and not self.outputs:
            self.setup()

        self.properties.update(properties)

    def setup(self):
        """Initializes the setup process."""
        pass

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        try:
            return self.id == other.id
        except AttributeError:
            return False

    def __str__(self):
        return f"{self.title} ({self.id})"

    def __repr__(self):
        return f"{self.title} ({self.id})"

    def model_dump(self, **kwargs):
        """Dump the model data, handling UNRESOLVED values in properties."""
        data = super().model_dump(**kwargs)
        # Handle UNRESOLVED values in properties
        if "properties" in data:
            data["properties"] = {
                k: None if v is UNRESOLVED else v for k, v in data["properties"].items()
            }
        return data

    @pydantic.model_validator(mode="before")
    @classmethod
    def handle_unresolved_properties(cls, data: Any) -> Any:
        """
        Handle unresolved properties in the given data.
        
        This class method checks if the input `data` is a dictionary and contains the
        key "properties". If so, it iterates over the properties and updates their
        values to UNRESOLVED if they are currently set to "UNRESOLVED" or None. The
        modified data is then returned, ensuring that unresolved properties are
        consistently marked.
        """
        if isinstance(data, dict) and "properties" in data:
            properties = data["properties"]
            data["properties"] = {
                k: UNRESOLVED if v in ("UNRESOLVED", None) else v
                for k, v in properties.items()
            }
        return data

    def get_output_socket(self, name: str) -> Socket:
        """Retrieve the output socket by its name."""
        for socket in self.outputs:
            if socket.name == name:
                return socket
        return None

    def get_input_socket(self, name: str) -> Socket:
        """Retrieve an input socket by its name."""
        for socket in self.inputs:
            if socket.name == name:
                return socket
        return None

    def add_input(self, name: str, **kwargs) -> Socket:
        """Add an input socket with the given name and additional parameters."""
        socket = Socket(name=name, node=self, **kwargs)
        self.inputs.append(socket)
        return socket

    def remove_input(self, name: str):
        """Remove the input socket with the given name if it exists."""
        socket = self.get_input_socket(name)
        if socket:
            self.inputs.remove(socket)

    def add_output(self, name: str, **kwargs) -> Socket:
        """Add a new output socket to the node."""
        socket = Socket(name=name, node=self, **kwargs)
        self.outputs.append(socket)
        return socket

    def remove_output(self, name: str):
        socket = self.get_output_socket(name)
        if socket:
            self.outputs.remove(socket)

    def get_property_field(self, name: str) -> PropertyField:
        """
        checks self.Fields for a field with the given name

        it will be defined as a meta class in the derived class

        returns the field if it exists, otherwise return a generic field

        class Node:
            class Fields:
                name = PropertyField(name="number", description="A number", type=int)
        """

        if not hasattr(self.__class__, "Fields"):
            type_str = (
                PYTHON_TYPE_TO_STRING.get(str(type(self.properties.get(name, ""))))
                or "str"
            )
            return PropertyField(name=name, description=name, type=type_str)

        FieldMeta = self.__class__.Fields

        if not hasattr(FieldMeta, name):
            type_str = (
                PYTHON_TYPE_TO_STRING.get(str(type(self.properties.get(name, ""))))
                or "str"
            )
            return PropertyField(name=name, description=name, type=type_str)

        return getattr(FieldMeta, name)

    def set_property(self, name: str, value: Any, state: GraphState | None = None):
        """Set a property value."""
        if state is None:
            self.properties[name] = value
        else:
            state.set_node_property(self, name, value)

    def get_property(self, name: str, state: GraphState | None = None) -> Any:
        """Get a property value"""

        if state is None:
            try:
                state: GraphState = graph_state.get()
            except LookupError:
                return self.properties.get(name, UNRESOLVED)

        return state.get_node_property(self, name)

    def get_input_value(self, name: str) -> Any:
        # Find matching input socket
        """Get the value for a specific input or fallback to property."""
        for socket in self.inputs:
            if socket.name == name:
                # If socket is connected use it
                if socket.source:
                    return socket.value
                # Otherwise fall back to property
                break

        return self.get_property(name)

    def get_input_values(self) -> dict[str, Any]:
        """Get all input values as a dictionary mapped by socket name,
        falling back to properties for unconnected inputs"""
        values = {}

        names = set(
            [socket.name for socket in self.inputs] + list(self.properties.keys())
        )

        for name in names:
            values[name] = self.get_input_value(name)

        return values

    def set_output_values(self, values: dict[str, Any]):
        """Set output values based on the provided socket names."""
        for socket in self.outputs:
            if socket.name in values:
                socket.value = values[socket.name]

    async def run(self, state: GraphState):
        """Run the node behavior."""
        pass

    def check_is_available(self, state: GraphState) -> bool:
        """
        Check if the node is available based on its inputs and outputs.
        
        This function determines the availability of a node by checking its input
        sockets  and their connections. It groups the input sockets and ensures that
        all ungrouped  sockets are available. For grouped sockets, at least one must be
        active. If the  node has no outputs, it is considered available if all inputs
        are satisfied.  Additionally, it checks for active paths in the output sockets
        to ensure they  connect to non-deactivated nodes.
        
        Args:
            state (GraphState): The current state of the graph to evaluate node availability.
        """
        if self._isolated:
            return False

        # Group sockets by their group
        grouped_sockets = {}
        ungrouped_sockets = []

        for socket in self.inputs:
            if socket.optional:
                continue

            if socket.group:
                if socket.group not in grouped_sockets:
                    grouped_sockets[socket.group] = []
                grouped_sockets[socket.group].append(socket)
            else:
                ungrouped_sockets.append(socket)

        # Check ungrouped sockets - all must be available
        for socket in ungrouped_sockets:
            if (
                socket.source is None
                or socket.source.deactivated
                or socket.value is UNRESOLVED
            ):
                if self.get_property(socket.name) is UNRESOLVED:
                    if state.verbosity >= NodeVerbosity.VERBOSE:
                        log.warning(
                            f"Node {self.title} input {socket.name} is not available, missing socket {socket.name}"
                        )

                    for out_socket in self.outputs:
                        out_socket.deactivated = True
                    return False

        # Check grouped sockets - at least one socket in each group must be available
        for group_sockets in grouped_sockets.values():
            group_has_active = False

            for socket in group_sockets:
                # Check if socket has an active source or a property value
                if (
                    socket.source
                    and not socket.source.deactivated
                    and socket.value is not UNRESOLVED
                ) or self.get_property(socket.name) is not UNRESOLVED:
                    group_has_active = True
                    break

            if not group_has_active:
                if state.verbosity >= NodeVerbosity.VERBOSE:
                    log.warning(
                        f"Node {self.title} group {group_sockets[0].group} is not available"
                    )
                # If no socket in the group is active, deactivate outputs and return False
                for out_socket in self.outputs:
                    out_socket.deactivated = True
                return False

        # If we have no outputs, we're an endpoint node - run if we have our inputs
        if not self.outputs:
            return True

        # Keep track of visited nodes to handle cycles
        visited = set()

        def has_active_path(current_socket: Socket, visited_nodes: set) -> bool:
            # If we've seen this node already, skip it to avoid cycles
            if current_socket.node in visited_nodes:
                return False

            visited_nodes.add(current_socket.node)

            # If this output socket is already deactivated, path is dead
            if current_socket.deactivated:
                return False

            # If any input socket uses this as a source and isn't deactivated,
            # this is a valid path
            for node in current_socket.node.outputs:
                if not node.deactivated:
                    return True

                # For each output, look for nodes that use it as input
                # and check their outputs recursively
                if node.source and not node.source.deactivated:
                    if has_active_path(node.source, visited_nodes.copy()):
                        return True

            return False

        # Check if any output path leads somewhere active
        is_available = any(
            has_active_path(socket, visited.copy()) for socket in self.outputs
        )

        # If not available, mark all our outputs as deactivated
        if not is_available:
            for socket in self.outputs:
                socket.deactivated = True

        return is_available

    def is_set(self, value: Any, none_is_set: bool = False) -> bool:

        """Check if a value is set."""
        if none_is_set:
            return value is not UNRESOLVED
        return value is not UNRESOLVED and value is not None

    def require_input(self, input_name: str, none_is_set: bool = False) -> Any:
        """
        Require an input to be set and return it

        If the input is not set, raise an InputValueError

        If none_is_set is True, None is considered a set value
        """

        value = self.get_input_value(input_name)

        if not self.is_set(value, none_is_set):
            raise InputValueError(self, input_name, f"Value is not set: {value}")

        return value

    def normalized_input_value(self, input_name: str, none_is_set: bool = False) -> Any:
        """
        Helper function to check if a value is set

        UNRESOLVED values are returned as None
        """
        value = self.get_input_value(input_name)

        if not self.is_set(value, none_is_set):
            return None

        return value

    def require_number_input(self, name: str, types: tuple = (int, float)):
        value = self.require_input(name)

        if isinstance(value, str):
            try:
                if float in types:
                    value = float(value)
                else:
                    value = int(value)
            except ValueError:
                raise InputValueError(self, name, "Invalid number")

        if not isinstance(value, types):
            raise InputValueError(self, name, "Value must be a number")

        return value


@base_node_type("core/Node")
class Node(NodeBase):
    inputs: list[Socket] = pydantic.Field(default_factory=list)
    outputs: list[Socket] = pydantic.Field(default_factory=list)


class Entry(Node):
    def __init__(self, title="Entry", **kwargs):
        super().__init__(title=title, **kwargs)
        self.add_output("state")

    async def run(self, state: GraphState):
        """Sets the output values with the given state."""
        self.set_output_values({"state": state})


class Router(Node):
    selector: Callable = pydantic.Field(default_factory=lambda state: 0)

    num_outputs: int = 2

    def __init__(self, num_outputs: int, title="Router", **kwargs):
        super().__init__(num_outputs=num_outputs, title=title, **kwargs)

    def setup(self):
        """Set up outputs and an input for the instance."""
        for i in range(self.num_outputs):
            self.add_output(f"output_{i}")
        self.add_input("input")

    async def run(self, state: GraphState):
        """Processes the outputs based on the selected route."""
        route_to = self.selector(state)

        for idx, socket in enumerate(self.outputs):
            if idx != route_to:
                socket.deactivated = True
            else:
                print(
                    f"Setting output {socket.name} to {self.get_input_value('input')}"
                )
                self.set_output_values({socket.name: self.get_input_value("input")})


@register("core/Input")
class Input(Node):
    class Fields:
        input_type = PropertyField(
            name="input_type",
            description="Input Type",
            type="str",
            default="any",
            choices=TYPE_CHOICES,
            generate_choices=lambda: TYPE_CHOICES,
        )

        input_name = PropertyField(
            name="input_name", description="Input Name", type="str", default="state"
        )

        input_optional = PropertyField(
            name="input_optional",
            description="Input Optional",
            type="bool",
            default=False,
        )

        input_group = PropertyField(
            name="input_group", description="Input Group", type="str", default=""
        )

        num = PropertyField(name="num", description="Number", type="int", default=0)

    @pydantic.computed_field(description="Node style")
    @property
    def style(self) -> NodeStyle:
        """Return the style of the node."""
        return NodeStyle(
            node_color="#2d2c39",
            title_color="#312e57",
            icon="F02FA",  # import
            auto_title="IN {input_name}",
        )

    def __init__(self, title="Input Socket", **kwargs):
        super().__init__(title=title, **kwargs)

    def setup(self):
        """Initializes properties and output for the instance."""
        self.set_property("input_type", "any")
        self.set_property("input_name", "state")
        self.set_property("input_optional", False)
        self.set_property("input_group", "")
        self.set_property("num", 0)

        self.add_output("value")


@register("core/Output")
class Output(Node):
    class Fields:
        output_type = PropertyField(
            name="output_type",
            description="Output Type",
            type="str",
            default="",
            choices=TYPE_CHOICES,
            generate_choices=lambda: TYPE_CHOICES,
        )

        output_name = PropertyField(
            name="output_name", description="Output Name", type="str", default="state"
        )

        num = PropertyField(name="num", description="Number", type="int", default=0)

    @pydantic.computed_field(description="Node style")
    @property
    def style(self) -> NodeStyle:
        """Return the style of the node."""
        return NodeStyle(
            node_color="#2d392c",
            title_color="#30572e",
            icon="F0207",  # export
            auto_title="OUT {output_name}",
        )

    def __init__(self, title="Output Socket", **kwargs):
        super().__init__(title=title, **kwargs)

    def setup(self):
        self.set_property("output_type", "any")
        self.set_property("output_name", "state")
        self.set_property("num", 0)

        self.add_input("value", optional=True)


@register("core/ModuleProperty")
class ModuleProperty(Node):
    """
    A node that can be placed to define a property of a Graph

    Properties:
    - property_name: The name of the property
    - property_type: The type of the property
    - default: The default value of the property
    - choices: The choices of the property
    - readonly: Whether the property is readonly
    - ephemeral: Whether the property is ephemeral
    - required: Whether the property is required

    Outputs:
    - name: The name of the property
    - value: The value of the property
    """

    class Fields:
        property_name = PropertyField(
            name="property_name",
            description="Property Name",
            type="str",
            default="",
        )
        property_type = PropertyField(
            name="property_type",
            description="Property Type",
            type="str",
            default="",
            choices=["str", "bool", "int", "float", "text"],
        )
        default = PropertyField(
            name="default",
            description="Default Value",
            type="any",
            default=UNRESOLVED,
        )
        choices = PropertyField(
            name="choices",
            description="Choices",
            type="list",
            default=[],
        )
        description = PropertyField(
            name="description",
            description="Description",
            type="str",
            default="",
        )
        num = PropertyField(
            name="num",
            description="Number",
            type="int",
            default=0,
            min=0,
        )

    @pydantic.computed_field(description="Node style")
    @property
    def style(self) -> NodeStyle:
        """Return the style of the node."""
        return NodeStyle(
            node_color="#2c3339",
            title_color="#2e4657",
            icon="F0AE7",  # variable
            auto_title="PROP {property_name}",
        )

    @property
    def to_property_field(self) -> PropertyField:
        """Returns a PropertyField constructed from instance properties."""
        return PropertyField(
            name=self.get_property("property_name"),
            description=self.get_property("description"),
            type=self.get_property("property_type"),
            default=self.cast_value(self.get_property("default")),
            choices=self.get_property("choices"),
        )

    def __init__(self, title="Module Property", **kwargs):
        super().__init__(title=title, **kwargs)

    def setup(self):
        """Initializes properties and outputs for the object."""
        self.set_property("property_name", "")
        self.set_property("property_type", "")
        self.set_property("default", UNRESOLVED)
        self.set_property("choices", UNRESOLVED)
        self.set_property("description", "")
        self.set_property("num", 0)
        self.add_output("name")
        self.add_output("value")

    def cast_value(self, value: Any) -> Any:
        # if UNRESOLVED return default
        """
        Cast a value to a specified type based on property settings.
        
        This method checks if the provided value is UNRESOLVED and retrieves a default
        value if necessary. It then determines the target type  from the property
        settings and converts the value accordingly.  Supported types include string,
        boolean, integer, and float.  The conversion logic for boolean values also
        handles string  representations of truthy and falsy values.
        
        Args:
            value (Any): The value to be cast to the specified type.
        """
        if value is UNRESOLVED:
            value = self.get_property("default")

        if self.get_property("property_type") in ["str", "text"]:
            return str(value)
        elif self.get_property("property_type") == "bool":
            if isinstance(value, str):
                if value.lower() in ["true", "yes", "1"]:
                    return True
                elif value.lower() in ["false", "no", "0"]:
                    return False
                else:
                    return bool(value)
            return bool(value)
        elif self.get_property("property_type") == "int":
            return int(value)
        elif self.get_property("property_type") == "float":
            return float(value)
        return str(value)


@register("core/Route")
class Route(Node):
    """Simply passes the value of the input to the output"""

    def __init__(self, title="Route", **kwargs):
        super().__init__(title=title, **kwargs)

    def setup(self):
        """Initialize input and output for the value."""
        self.add_input("value")
        self.add_output("value")

    async def run(self, state: GraphState):
        """Executes the run process by getting input and setting output values."""
        value = self.get_input_value("value")
        self.set_output_values({"value": value})


@register("core/Watch")
class Watch(Node):
    """
    Outputs the value of the input socket
    """

    @pydantic.computed_field(description="Node style")
    @property
    def style(self) -> NodeStyle:
        """Return the style of the node."""
        return NodeStyle(
            node_color="#2c3339",
            title_color="#2e4657",
            icon="F06D0",  # eye-outline
        )

    def __init__(self, title="Watch", **kwargs):
        super().__init__(title=title, **kwargs)

    def setup(self):
        """Initialize input and output for the setup."""
        self.add_input("value")
        self.add_output("value")

    async def run(self, state: GraphState):
        value = self.get_input_value("value")

        if state.shared.get("creative_mode"):
            log.debug(f"Watch:{self.title}", value=value)

        self.set_output_values({"value": value})


@register("core/Stage")
class Stage(Node):
    """
    A node that can be connected in or out and
    defines a stage level for the nodes connected to it

    This stage level can be used to control the order
    of execution of nodes in the graph, the lowest stage
    will be executed first.

    Inputs:

    - state: Any value to pass through. If not connected, defaults to True
    - state_b: Any value to pass through.
    - state_c: Any value to pass through.
    - state_d: Any value to pass through.

    Outputs:

    - state: The value of the input state or True if corresponding input is not connected
    - state_b: The value of the input state_b
    - state_c: The value of the input state_c
    - state_d: The value of the input state_d
    """

    class Fields:
        stage = PropertyField(
            name="stage", description="Stage", type="int", default=0, min=0, step=1
        )

    @pydantic.computed_field(description="Node style")
    @property
    def style(self) -> NodeStyle:
        """Return the style configuration for a node."""
        return NodeStyle(
            node_color="#2c2c38",
            title_color="#343055",
            icon="F0603",  # priority-high
            auto_title="Stage {stage}",
        )

    def __init__(self, title="Stage", **kwargs):
        super().__init__(title=title, **kwargs)

    def setup(self):
        """Initializes inputs and outputs for the setup."""
        self.add_input("state", optional=True)
        self.add_input("state_b", optional=True)
        self.add_input("state_c", optional=True)
        self.add_input("state_d", optional=True)
        self.set_property("stage", 0)
        self.add_output("state")
        self.add_output("state_b")
        self.add_output("state_c")
        self.add_output("state_d")

    async def run(self, state: GraphState):
        """Processes input values and sets output values for the given state."""
        state_value = self.get_input_value("state")
        state_b_value = self.get_input_value("state_b")
        state_c_value = self.get_input_value("state_c")
        state_d_value = self.get_input_value("state_d")

        # unconnected inputs default to True in the output

        if state_value is UNRESOLVED and not self.get_input_socket("state").source:
            state_value = True

        self.set_output_values(
            {
                "state": state_value,
                "state_b": state_b_value,
                "state_c": state_c_value,
                "state_d": state_d_value,
            }
        )


def validate_node(
    v: Any,
    handler: pydantic.ValidatorFunctionWrapHandler,
    info: pydantic.ValidationInfo,
) -> Node:
    # If it's already a Node instance, return it
    """def validate_node(
    v: Any,     handler: pydantic.ValidatorFunctionWrapHandler,     info:
    pydantic.ValidationInfo, ) -> Node:     Validate and return a Node instance
    from the given input.      This function checks if the input `v` is already a
    Node instance.      If not, it verifies if `v` is a dictionary and attempts to
    retrieve      the corresponding node class from the registry. If a valid class
    is found, it instantiates and returns the Node. If the input cannot      be
    validated, a ValueError is raised.      Args:         v (Any): The input value
    to validate.         handler (pydantic.ValidatorFunctionWrapHandler): The
    handler for validation.         info (pydantic.ValidationInfo): Information
    about the validation context."""
    if isinstance(v, NodeBase):
        return v

    # If it's a dict, check registry and instantiate appropriate class
    if isinstance(v, dict):
        registry_name = v.get("registry")
        # print(f"Validating node with registry: {registry_name}")
        if registry_name:
            node_cls = get_node(registry_name)
            # print(f"Found node class: {node_cls}")
            if node_cls:
                return node_cls(**v)

    raise ValueError(f"Could not validate node: {v}")


# Create annotated type for nodes with registry validation
RegistryNode = Annotated[NodeBase, pydantic.WrapValidator(validate_node)]


class Group(pydantic.BaseModel):
    title: str = "Group"
    x: int = 0
    y: int = 0
    width: int = 200
    height: int = 100
    color: str | None = None
    font_size: int = 24
    inherited: bool = False


class Comment(pydantic.BaseModel):
    text: str = "Comment"
    x: int = 0
    y: int = 0
    width: int = 200
    inherited: bool = False


@register("util/ModuleStyle")
class ModuleStyle(Node):
    """
    An isolated node that will define the Graph's style
    """

    _isolated: ClassVar[bool] = True

    class Fields:
        node_color = PropertyField(
            name="node_color", description="Node Color", type="str", default=UNRESOLVED
        )
        title_color = PropertyField(
            name="title_color",
            description="Title Color",
            type="str",
            default=UNRESOLVED,
        )
        auto_title = PropertyField(
            name="auto_title", description="Auto Title", type="str", default=UNRESOLVED
        )
        icon = PropertyField(
            name="icon",
            description="Icon (Material Icon Codepoint)",
            type="str",
            default=UNRESOLVED,
        )

    def __init__(self, title="Module Style", **kwargs):
        super().__init__(title=title, **kwargs)

    def setup(self):
        """Initialize properties for the node."""
        self.set_property("node_color", UNRESOLVED)
        self.set_property("title_color", UNRESOLVED)
        self.set_property("auto_title", UNRESOLVED)
        self.set_property("icon", UNRESOLVED)

    def get_style(self) -> NodeStyle:
        """Retrieve the style properties for a node."""
        return NodeStyle(
            node_color=self.get_property("node_color"),
            title_color=self.get_property("title_color"),
            auto_title=self.get_property("auto_title"),
            icon=self.get_property("icon"),
        )

    async def run(self, state: GraphState):
        """Runs the asynchronous process with the given state."""
        pass


@base_node_type("core/Graph")
class Graph(NodeBase):
    nodes: dict[str, RegistryNode] = pydantic.Field(default_factory=dict)
    edges: dict[str, list[str]] = pydantic.Field(default_factory=dict)
    sockets: dict[str, Socket] = pydantic.Field(default_factory=dict, exclude=True)
    groups: list[Group] = pydantic.Field(default_factory=list)
    comments: list[Comment] = pydantic.Field(default_factory=list)
    extends: str | None = None

    error_handlers: list[Callable] = pydantic.Field(default_factory=list, exclude=True)
    callbacks: list[Callable] = pydantic.Field(default_factory=list, exclude=True)

    _interrupt: bool = False

    @property
    def input_nodes(self) -> list[Input]:
        """Return a list of input nodes from the nodes dictionary."""
        return [node for node in self.nodes.values() if isinstance(node, Input)]

    @property
    def output_nodes(self) -> list[Output]:
        """Return a list of Output nodes from the nodes dictionary."""
        return [node for node in self.nodes.values() if isinstance(node, Output)]

    @property
    def module_property_nodes(self) -> list[ModuleProperty]:
        nodes = [
            node for node in self.nodes.values() if isinstance(node, ModuleProperty)
        ]

        # sort by num property
        nodes.sort(key=lambda x: x.get_property("num"))

        return nodes

    @pydantic.computed_field(description="Inputs")
    @property
    def inputs(self) -> list[Socket]:
        if hasattr(self, "_inputs"):
            return self._inputs

        # find sub nodes of Input type and dynamically output Socket types
        inputs = []

        nodes = []

        # collect nodes and sort by num property
        for node in self.input_nodes:
            nodes.append(node)

        nodes.sort(key=lambda x: x.get_property("num"))

        for node in nodes:
            inputs.append(
                Socket(
                    name=node.get_property("input_name"),
                    socket_type=node.get_property("input_type"),
                    optional=node.get_property("input_optional"),
                    group=node.get_property("input_group"),
                    node=self,
                )
            )

        self._inputs = inputs

        return inputs

    @pydantic.computed_field(description="Outputs")
    @property
    def outputs(self) -> list[Socket]:
        """Returns a list of Socket objects representing the outputs of the node."""
        if hasattr(self, "_outputs"):
            return self._outputs

        # find sub nodes of Output type and dynamically output Socket types
        outputs = []

        nodes = []

        for node in self.output_nodes:
            nodes.append(node)

        nodes.sort(key=lambda x: x.get_property("num"))

        for node in nodes:
            outputs.append(
                Socket(
                    name=node.get_property("output_name"),
                    socket_type=node.get_property("output_type"),
                    node=self,
                )
            )

        self._outputs = outputs

        return outputs

    @pydantic.computed_field(description="Module Fields")
    @property
    def module_properties(self) -> dict[str, PropertyField]:
        # Dynamically find all ModuleProperty nodes and return them
        # as a list of PropertyField objects

        """Return a dictionary of module properties as PropertyField objects."""
        if hasattr(self, "_module_properties"):
            return self._module_properties

        properties = {}
        for node in self.module_property_nodes:
            name = node.get_property("property_name")
            if name not in properties:
                properties[name] = node.to_property_field
            else:
                log.warning("Duplicate module property", name=name)

        self._module_properties = properties

        return properties

    @pydantic.computed_field(description="Node style")
    @property
    def style(self) -> NodeStyle | None:

        """Return the style of the ModuleStyle node in the graph."""
        for _node in self.nodes.values():
            if isinstance(_node, ModuleStyle):
                return _node.get_style()

        return None

    def model_dump(self, **kwargs) -> dict:
        """Serialize the model's data while filtering out inherited nodes, groups, and
        comments.
        
        This function overrides the model_dump method to customize the serialization
        process.  When save_state is active, it removes all inherited nodes, groups,
        and comments from the  output. Additionally, it updates the edges dictionary to
        exclude any edges that reference  the dropped nodes, ensuring that the
        serialized data accurately reflects the current state  of the model.
        
        Args:
            **kwargs: Additional keyword arguments passed to the superclass method.
        
        Returns:
            dict: The serialized model data, filtered according to the save_state condition.
        """
        data = super().model_dump(**kwargs)

        try:
            if save_state.get():
                # Filter out inherited nodes
                data["nodes"] = {
                    node_id: node
                    for node_id, node in data["nodes"].items()
                    if not node.get("inherited", False)
                }

                # Filter out inherited groups
                data["groups"] = [
                    group
                    for group in data["groups"]
                    if not group.get("inherited", False)
                ]

                # Filter out inherited comments
                data["comments"] = [
                    comment
                    for comment in data["comments"]
                    if not comment.get("inherited", False)
                ]

                # Remove edges that reference dropped nodes
                dropped_node_ids = {
                    node_id for node_id, node in self.nodes.items() if node.inherited
                }

                data["edges"] = {
                    output_id: input_ids
                    for output_id, input_ids in data["edges"].items()
                    if output_id.split(".")[0] not in dropped_node_ids
                    and all(
                        input_id.split(".")[0] not in dropped_node_ids
                        for input_id in input_ids
                    )
                }

        except LookupError:
            # save_state not set, return full data
            pass

        return data

    def reinitialize(self) -> "Graph":
        """Reinitialize the graph by resetting its properties and connections."""
        self.set_node_references()
        self.set_socket_source_references()
        self.reset_ephemeral_properties()
        self.ensure_connections()
        return self

    def reset(self):
        """
        Reset all _value properties in all sockets
        """
        for node in self.nodes.values():
            for socket in node.inputs + node.outputs:
                socket.value = UNRESOLVED
                socket.deactivated = False

        self.reinitialize()

    def reset_sockets(self):
        """Reset all deactivated properties in all sockets."""
        for node in self.nodes.values():
            for socket in node.inputs + node.outputs:
                socket.value = UNRESOLVED
                socket.deactivated = False

    def ensure_connections(self):
        """
        Ensure all sockets are connected.
        
        This function iterates through the edges of the graph, ensuring that  all
        output sockets are properly connected to their corresponding input  sockets.
        For each output socket, it checks if the associated input  socket exists and is
        not already connected. If the input socket is  found to be unconnected, it
        establishes a connection between the  output and input sockets.
        """
        for output_socket_id, input_socket_ids in self.edges.items():
            output_node_id, output_socket_name = output_socket_id.split(".", 1)
            output_node = self.nodes[output_node_id]
            output_socket = output_node.get_output_socket(output_socket_name)

            for input_socket_id in input_socket_ids:
                input_node_id, input_socket_name = input_socket_id.split(".", 1)
                input_node = self.nodes[input_node_id]
                input_socket = input_node.get_input_socket(input_socket_name)

                if not input_socket:
                    log.warning(
                        "Input socket not found",
                        input_socket_name=input_socket_name,
                        input_node_id=input_node_id,
                    )
                    continue

                if not input_socket.source:
                    self.connect(output_socket, input_socket)

    def reset_ephemeral_properties(self):

        """Reset all ephemeral properties in all nodes."""
        for node in self.nodes.values():
            for property_name in node.properties:
                field = node.get_property_field(property_name)
                if field.ephemeral:
                    node.set_property(property_name, field.default)

    def set_node_references(self) -> "Graph":

        """Loops through all nodes and sets their socket.node references"""
        for node in self.nodes.values():
            for socket in node.inputs + node.outputs:
                socket.node = node
                self.sockets[socket.full_id] = socket

        return self

    def set_socket_source_references(self) -> "Graph":
        """
        def set_socket_source_references(self) -> "Graph":
        Set source references for input sockets based on edge connections.  This method
        iterates through all nodes in the graph and their input sockets. For each input
        socket, it checks the edges to find the corresponding output socket. If a match
        is found, it sets the `source` attribute of the input socket to the identified
        output socket. This process ensures that each input socket correctly references
        its source based on the current graph structure.
        """
        for node in self.nodes.values():
            for socket in node.inputs:
                for output_socket_id, input_socket_ids in self.edges.items():
                    output_node_id, output_socket_name = output_socket_id.split(".", 1)
                    if socket.id in input_socket_ids:
                        output_socket = self.sockets[
                            f"{output_node_id}.{output_socket_name}"
                        ]
                        socket.source = output_socket
                        break
        return self

    def node(self, node_id: str) -> NodeBase:
        """Retrieve a node by its identifier."""
        return self.nodes[node_id]

    def add_node(self, node: NodeBase):
        self.nodes[node.id] = node

        for socket in node.inputs + node.outputs:
            self.sockets[socket.id] = socket

    def connect(self, output_socket: Socket | str, input_socket: Socket | str):
        """
        Connect an output socket to an input socket.
        
        This method establishes a connection between an output socket and an input
        socket.  It first checks if the provided sockets are strings and resolves them
        to their  corresponding Socket objects. If either socket is invalid, a warning
        is logged.  The function also manages the internal edges structure to maintain
        the connections  and sets the source of the input socket to the output socket.
        """
        if isinstance(output_socket, str):
            output_socket = self.sockets[output_socket]

        if isinstance(input_socket, str):
            input_socket = self.sockets[input_socket]

        if not output_socket or not input_socket:
            log.warning(
                "Could not connect sockets",
                output_socket=output_socket,
                input_socket=input_socket,
            )
            return

        if output_socket.full_id not in self.edges:
            self.edges[output_socket.full_id] = []

        if input_socket.full_id not in self.edges[output_socket.full_id]:
            self.edges[output_socket.full_id].append(input_socket.full_id)
        input_socket.source = output_socket

    def build(self) -> nx.DiGraph:
        """Build a directed acyclic graph from socket connections."""
        graph = nx.DiGraph()

        # Add edges between nodes based on socket connections
        for output_socket_id, input_socket_ids in self.edges.items():
            output_node_id, _ = output_socket_id.split(".", 1)

            for input_socket_id in input_socket_ids:
                input_node_id, _ = input_socket_id.split(".", 1)

                graph.add_edge(output_node_id, input_node_id)

        return graph

    def assign_priority(self, node_chain: nx.DiGraph) -> int:

        """Assigns a priority based on the minimum stage value in the node chain."""
        min_stage = float("inf")
        for node_id in node_chain:
            node = self.nodes[node_id]
            if isinstance(node, Stage):
                stage_value = node.get_property("stage")
                min_stage = min(min_stage, stage_value)
        # If no Stage nodes found, use default priority
        return min_stage

    async def run(self, state: GraphState):
        await self.execute(state)

    async def handle_error(self, state: GraphState, exc: Exception):
        for handler in self.error_handlers:
            try:
                await handler(state, exc)
            except Exception as exc:
                log.error("Error in error handler", exc=traceback.format_exc())

    async def clone(self) -> "Graph":
        """Clone the graph."""
        data = json.loads(self.model_dump_json())
        return Graph(**data)

    @shared_debounce(1.0, "node_state")
    async def signal_note_state(self, state: GraphState):
        """Signals the node state if not in creative mode."""
        if not state.shared.get("creative_mode"):
            return

        await async_signals.get("nodes_node_state").send(state)
        state.stack.clear()

    async def node_state_push(
        self,
        node: NodeBase,
        state: GraphState,
        inactive: bool = False,
        reset: bool = False,
    ) -> NodeState:
        """Pushes the node state to the stack and signals the state change."""
        if not state.shared.get("creative_mode"):
            return
        node_exec = NodeState(node, state)

        if inactive:
            node_exec.deactivated = True

        if reset:
            node_exec.start_time = None
            node_exec.end_time = None

        # push to the end of the stack
        state.stack.append(node_exec)
        await self.signal_note_state(state)

        return node_exec

    async def node_state_pop(
        self,
        prev_state: NodeState,
        node: NodeBase,
        state: GraphState,
        error: str = None,
    ) -> NodeState:
        """Populates the node state and updates the graph state."""
        if not state.shared.get("creative_mode"):
            return

        node_exec = NodeState(
            node, state, start_time=prev_state.start_time, end_time=prev_state.end_time
        )
        node_exec.end_time = time.time()

        if error:
            node_exec.error = error

        state.stack.append(node_exec)
        await self.signal_note_state(state)

        return node_exec

    async def node_state_sync_all(self, state: GraphState):
        """Synchronizes the state of all nodes."""
        for node in self.nodes.values():
            await self.node_state_push(node, state, reset=True)

    async def get_nodes(self, fn_filter: Callable = None) -> list[NodeBase]:
        """
        Returns a list of nodes in the graph
        """
        if not fn_filter:
            return list(self.nodes.values())
        return [node for node in self.nodes.values() if fn_filter(node)]

    async def get_node(
        self, fn_filter: Callable = None, require_unique: bool = True
    ) -> NodeBase:
        """Returns a single node from the graph based on the provided filter."""
        nodes = await self.get_nodes(fn_filter)
        if require_unique and len(nodes) > 1:
            raise ValueError("Multiple nodes found")
        return nodes[0] if nodes else None

    async def get_nodes_connected_to(
        self, node: NodeBase, fn_filter: Callable = None
    ) -> list[NodeBase]:
        """
        Returns a list of nodes connected to the given node.
        
        This function builds a graph and retrieves the ancestors of the specified  node
        using the `get_ancestors_with_forks` function. If a filter function
        (`fn_filter`) is provided, it applies this filter to the ancestors before
        returning the list of connected nodes. If no filter is provided, it returns
        all ancestors directly.
        """
        graph = self.build()
        predecessors = get_ancestors_with_forks(graph, node.id)
        if not fn_filter:
            return [self.nodes[node_id] for node_id in predecessors]
        return [
            self.nodes[node_id]
            for node_id in predecessors
            if fn_filter(self.nodes[node_id])
        ]

    async def execute_to_node(
        self,
        stop_at_node: NodeBase,
        outer_state: GraphState | None = None,
        callbacks: list[Callable] = [],
        emit_state: bool = False,
        state_values: dict[str, Any] = None,
        execute_forks: bool = False,
        run_isolated: bool = True,
    ):
        """
        Execute the graph in topological order up to a specified node.
        
        This function builds a graph and checks for the existence of the  specified
        stop_at_node. It retrieves all predecessor nodes,  constructs a subgraph, and
        verifies that it is acyclic before  executing the graph. The execution is
        performed within a  GraphContext, allowing for state management and callback
        execution after the inner execution completes.
        """
        graph = self.build()

        # check that node exists
        if stop_at_node.id not in self.nodes:
            raise ValueError(f"Node {stop_at_node.id} not found in graph")

        # Get all predecessor nodes including the target node
        if not execute_forks:
            predecessors = nx.ancestors(graph, stop_at_node.id)
        else:
            predecessors = get_ancestors_with_forks(graph, stop_at_node.id)
        predecessors.add(stop_at_node.id)

        # Get subgraph of only the nodes we need to execute
        subgraph = graph.subgraph(predecessors)

        # Check for cycles
        if not nx.is_directed_acyclic_graph(subgraph):
            raise ValueError("Graph contains cycles")

        with GraphContext(outer_state, self) as state:
            if state_values:
                state.data.update(state_values)

            await self._execute_inner(
                subgraph, state, emit_state=emit_state, run_isolated=run_isolated
            )

            for callback in callbacks:
                await callback(state)

            return state

    async def execute(
        self,
        outer_state: GraphState | None = None,
        state_values: dict[str, Any] = None,
        callbacks: list[Callable] = [],
    ):
        """Execute the graph in topological order"""

        graph = self.build()

        # Check for cycles
        if not nx.is_directed_acyclic_graph(graph):
            raise ValueError("Graph contains cycles")

        with GraphContext(outer_state, self) as state:
            self.reset()

            if state_values:
                state.data.update(state_values)

            await self.node_state_sync_all(state)
            await self._execute_inner(graph, state)
            for callback in self.callbacks:
                await callback(state)
            for callback in callbacks:
                await callback(state)

    async def _execute_inner(
        self,
        graph: nx.DiGraph,
        state: GraphState,
        emit_state: bool = True,
        run_isolated: bool = False,
    ):
        verbosity: NodeVerbosity = state.verbosity

        try:
            # route input socket values to their corresponding Input nodes
            for node in self.input_nodes:
                socket = self.get_input_socket(node.get_property("input_name"))
                if not socket or not socket.source:
                    continue

                socket_value = state.outer.get_node_socket_value(
                    socket.source.node, socket.source.name
                )

                if state.verbosity == NodeVerbosity.VERBOSE:
                    log.debug(f"Setting input value for {node.title} to {socket_value}")
                node.set_output_values({"value": socket_value})

            # for module property nodes we need to set their output socket values
            # base on the property value
            for node in self.module_property_nodes:
                name = node.get_property("property_name")
                value = self.get_property(name)
                node.set_output_values({"name": name, "value": node.cast_value(value)})

            # Separate into weakly connected components (isolated chains)
            chains = list(nx.weakly_connected_components(graph))

            # sort chains by priority
            chains.sort(key=lambda chain: self.assign_priority(chain))

            for chain in chains:
                subgraph = graph.subgraph(chain)

                sorted_nodes = list(nx.topological_sort(subgraph))

                # check if the final in the chain is _isolated, and if so, skip the chain
                if self.nodes[sorted_nodes[-1]]._isolated and not run_isolated:
                    continue

                # Execute nodes in topological order
                for node_id in sorted_nodes:
                    if self._interrupt:
                        self._interrupt = False
                        break

                    node = self.nodes[node_id]
                    if verbosity == NodeVerbosity.VERBOSE:
                        log.debug(f"Running node {node.title} (pre check)")

                    if not node.check_is_available(state):
                        if emit_state:
                            await self.node_state_push(node, state, inactive=True)
                        continue

                    if verbosity == NodeVerbosity.VERBOSE:
                        log.debug(f"Running node {node.title}")

                    if emit_state:
                        node_state = await self.node_state_push(node, state)

                    # run node
                    try:
                        await node.run(state)
                        if emit_state:
                            await self.node_state_pop(node_state, node, state)
                    except StopGraphExecution:
                        if emit_state:
                            await self.node_state_pop(node_state, node, state)
                        raise
                    except PASSTHROUGH_ERRORS as exc:
                        if emit_state:
                            await self.node_state_pop(node_state, node, state)

                        await self.attempt_catch_with_node_error_handler(state, exc)
                        raise exc
                    except ModuleError:
                        if emit_state:
                            await self.node_state_pop(
                                node_state, node, state, error=traceback.format_exc()
                            )
                    except Exception as exc:
                        if emit_state:
                            await self.node_state_pop(
                                node_state, node, state, error=traceback.format_exc()
                            )

                        await self.attempt_catch_with_node_error_handler(state, exc)

                    # route Output node values to their corresponding output sockets
                    if isinstance(node, Output):
                        socket = self.get_output_socket(
                            node.get_property("output_name")
                        )
                        if socket:
                            socket.value = node.get_input_socket("value").value
                            state.outer.set_node_socket_value(
                                self, socket.name, socket.value
                            )

                            if verbosity == NodeVerbosity.VERBOSE:
                                log.debug(
                                    f"Setting output value for {socket.full_id} to {socket.value}"
                                )

        except StopGraphExecution:
            pass
        except PASSTHROUGH_ERRORS as exc:
            raise exc
        except Exception as exc:
            try:
                await self.handle_error(state, exc)
            except Exception as handle_exc:
                log.error("Error in error handler", error=handle_exc, graph=self.title)
            finally:
                raise exc

    async def attempt_catch_with_node_error_handler(
        self, state: GraphState, exc: Exception
    ):
        """Attempt to catch an exception with a node error handler

        Args:
            state (GraphState): The current state of the graph
            exc (Exception): The exception to catch

        Raises:
            Will re-raise the exception if no error handler is found
        """

        error_handlers = await self.get_nodes(lambda n: hasattr(n, "catch"))
        if not error_handlers:
            raise exc

        caught = False
        for error_handler in error_handlers:
            if await error_handler.catch(state, exc):
                caught = True
                break

        if not caught:
            raise exc


@base_node_type("core/Loop")
class Loop(Graph):
    exit_condition: Callable = pydantic.Field(
        default_factory=lambda: lambda state: False, exclude=True
    )

    sleep: float = 0.001

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def setup(self):
        """Initialize input and output for the state."""
        self.add_input("state")
        self.add_output("state")

    async def on_loop_start(self, state: GraphState):
        """Handles the start of the loop."""
        pass

    async def on_loop_end(self, state: GraphState):
        """Handles actions to be performed at the end of a loop."""
        pass

    async def on_loop_error(self, state: GraphState, exc: Exception):
        """Handles errors that occur during the event loop."""
        pass

    async def execute(
        self,
        outer_state: GraphState,
        state_values: dict = None,
        run_isolated: bool = False,
    ):
        """Execute the graph in topological order"""
        graph = self.build()

        if not nx.is_directed_acyclic_graph(graph):
            raise ValueError("Graph contains cycles")

        with GraphContext(outer_state, self) as state:
            self.reset()

            if state_values:
                state.data.update(state_values)

            try:
                # Separate into weakly connected components (isolated chains)
                chains = list(nx.weakly_connected_components(graph))

                # sort chains by priority
                chains.sort(key=lambda chain: self.assign_priority(chain))

                while True:
                    self.reset_sockets()

                    BREAK_LOOP = False

                    # LOOP START

                    try:
                        await self.on_loop_start(state)
                    except Exception as exc:
                        try:
                            await self.handle_error(state, exc)
                            await self.on_loop_error(state, exc)
                            log.error(
                                "Error in on_loop_start",
                                exc=exc,
                                traceback=traceback.format_exc(),
                            )
                        except LoopBreak:
                            BREAK_LOOP = True
                        except LoopContinue:
                            pass
                        except LoopExit:
                            return

                    # PROCESS NODE CHAINS

                    for chain in chains:
                        if BREAK_LOOP:
                            break

                        await self.node_state_sync_all(state)

                        subgraph = graph.subgraph(chain)

                        sorted_nodes = list(nx.topological_sort(subgraph))

                        # check if the final in the chain is _isolated, and if so, skip the chain
                        if self.nodes[sorted_nodes[-1]]._isolated and not run_isolated:
                            continue

                        # Execute nodes in topological order
                        for node_id in sorted_nodes:
                            try:
                                if self._interrupt:
                                    self._interrupt = False
                                    raise LoopExit()

                                node = self.nodes[node_id]
                                if state.verbosity == NodeVerbosity.VERBOSE:
                                    log.debug(f"Running node {node.title} (pre check)")

                                if not node.check_is_available(state):
                                    await self.node_state_push(
                                        node, state, inactive=True
                                    )
                                    continue

                                if state.verbosity == NodeVerbosity.VERBOSE:
                                    log.debug(f"Running node {node.title}")

                                node_state = await self.node_state_push(node, state)
                                try:
                                    await node.run(state)
                                    await self.node_state_pop(node_state, node, state)
                                except PASSTHROUGH_ERRORS:
                                    raise
                                except ModuleError:
                                    await self.node_state_pop(
                                        node_state,
                                        node,
                                        state,
                                        error=traceback.format_exc(),
                                    )
                                except Exception:
                                    await self.node_state_pop(
                                        node_state,
                                        node,
                                        state,
                                        error=traceback.format_exc(),
                                    )
                                    raise

                                # Check for loop exit condition
                                if self.exit_condition(state):
                                    return
                            except LoopContinue:
                                continue
                            except LoopBreak:
                                BREAK_LOOP = True
                                break
                            except LoopExit:
                                return
                            except StopGraphExecution:
                                BREAK_LOOP = True
                                break
                            except (
                                ExitScene,
                                ResetScene,
                                RestartSceneLoop,
                                StopModule,
                            ):
                                raise
                            except Exception as exc:
                                try:
                                    await self.handle_error(state, exc)
                                    await self.on_loop_error(state, exc)
                                    log.error(
                                        "Error in Loop",
                                        graph=self.title,
                                        graph_cls=self.__class__,
                                        exc=exc,
                                        traceback=traceback.format_exc(),
                                    )
                                    await asyncio.sleep(1.0)
                                except LoopBreak:
                                    BREAK_LOOP = True
                                    break
                                except LoopContinue:
                                    continue
                                except LoopExit:
                                    return
                                except (
                                    ExitScene,
                                    ResetScene,
                                    RestartSceneLoop,
                                    StopModule,
                                ) as exc:
                                    raise
                                except Exception:
                                    log.error(
                                        "Error in on_loop_error",
                                        exc=traceback.format_exc(),
                                    )
                                BREAK_LOOP = True
                                break

                        if BREAK_LOOP:
                            break

                        await asyncio.sleep(self.sleep)

                    # LOOP END

                    await self.on_loop_end(state)
            # except Exception as e:
            #    log.error("Error in loop", exc=e, traceback=traceback.format_exc())
            #    raise
            finally:
                for callback in self.callbacks:
                    await callback(state)


@base_node_type("core/Event")
class Listen(Graph):
    """
    Listens for an event
    """

    _isolated: ClassVar[bool] = True

    class Fields:
        event_name = PropertyField(
            name="event_name",
            description="Event to listen for",
            type="str",
            default="",
        )

    @pydantic.computed_field(description="Node style")
    @property
    def style(self) -> NodeStyle:
        # If a style is defined in the graph it overrides the default
        """Get the node style, overriding the default if defined."""
        defined_style = super().style
        if defined_style:
            return defined_style

        return NodeStyle(
            node_color="#39382c",
            title_color="#57532e",
            icon="F0BF8",  # alpha-e-circle
        )

    def __init__(self, title="Listen", **kwargs):
        super().__init__(title=title, **kwargs)

    def setup(self):
        """Sets the event name property to UNRESOLVED."""
        """Initialize the event name property to UNRESOLVED."""
        self.set_property("event_name", UNRESOLVED)

    async def run(self, state: GraphState):
        """Execute the run method for the listen node."""
        """Run the listen node directly."""
        log.warning("Listen node run directly", node=self)
        return await super().run(state)

    async def execute_from_event(self, event: object):
        """
        async def execute_from_event(self, event: object):
        Execute a process based on the provided event.  This function attempts to
        retrieve the current GraphState, either from the global context or from the
        active scene if the global context is unavailable. It then pushes the node
        state, executes a process with the given event, and handles any exceptions that
        may arise by popping the node state accordingly. Proper error logging is
        performed if the event is executed outside of an active graph state.
        
        Args:
            event (object): The event object that triggers the execution.
        """
        try:
            state: GraphState = graph_state.get()
        except LookupError:
            # we are outside of the graph state, however we can still attempt
            # to get the scene's graph state
            scene = active_scene.get()
            if scene and getattr(scene, "nodegraph_state", None):
                state = scene.nodegraph_state
            else:
                log.error("Event node executed outside of active graph state")
                return

        node_state = await self.node_state_push(self, state)
        try:
            await self.execute(state, state_values={"event": event})
        except Exception as exc:
            await self.node_state_pop(
                node_state, self, state, error=traceback.format_exc()
            )
            raise exc
        await self.node_state_pop(node_state, self, state)


@base_node_type("core/EventTrigger")
class Trigger(Node):
    """
    Triggers an event
    """

    class Fields:
        event_name = PropertyField(
            name="event_name",
            description="Event to trigger",
            type="str",
            default="",
        )

    @property
    def signals(self):
        """Get the async_signals property."""
        return async_signals

    @property
    def signal_name(self) -> str | UNRESOLVED:
        """Get the event name input value."""
        return self.get_input_value("event_name")

    def __init__(self, title="Trigger Event", **kwargs):
        super().__init__(title=title, **kwargs)

    def setup_properties(self):
        """Initialize properties for the event."""
        self.set_property("event_name", "")

    def setup_required_inputs(self):
        """Sets up the required input for the instance."""
        self.add_input("trigger")

    def setup_optional_inputs(self):
        """Set up optional input for event name."""
        self.add_input("event_name", socket_type="str", optional=True)

    def setup_outputs(self):
        """Set up the output for the event socket."""
        self.add_output("event", socket_type="event")

    def setup(self):
        """Initializes required and optional inputs, properties, and outputs."""
        self.setup_required_inputs()
        self.setup_optional_inputs()
        self.setup_properties()
        self.setup_outputs()

    def make_event_object(self, state: GraphState) -> object:
        """Creates an event object based on the given state."""
        raise NotImplementedError("Event object not defined")

    async def after(self, state: GraphState, event: object):
        """Handles an event after a state change."""
        pass

    async def run(self, state: GraphState):
        """
        Triggers an event based on the current state.
        
        This asynchronous function checks if a valid event name is set and retrieves
        the corresponding signal.  If the signal is found, it creates an event object
        using the provided state and sends the event.  Additionally, it logs the event
        triggering if verbosity is set to verbose and updates the output values.
        Finally, it calls the after method to perform any follow-up actions with the
        state and event.
        """
        event_name = self.signal_name

        if not event_name or event_name == UNRESOLVED:
            log.error("Event name not set")
            return

        signal = async_signals.get(event_name)
        if not signal:
            log.error("Signal not found", event_name=event_name)
            return

        event = self.make_event_object(state)

        await signal.send(event)

        if state.verbosity >= NodeVerbosity.VERBOSE:
            log.debug(f"Triggered event {event_name}")

        self.set_output_values({"event": event})

        await self.after(state, event)
