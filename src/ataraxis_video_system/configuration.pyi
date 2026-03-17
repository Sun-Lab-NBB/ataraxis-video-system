from enum import IntEnum
from dataclasses import field, dataclass

from genicam.genapi import NodeMap as NodeMap
from ataraxis_data_structures import YamlConfig

class _NodeType(IntEnum):
    INTEGER = 2
    BOOLEAN = 3
    COMMAND = 4
    FLOAT = 5
    STRING = 6
    REGISTER = 7
    CATEGORY = 8
    ENUMERATION = 9
    ENUM_ENTRY = 10
    PORT = 11

class _AccessMode(IntEnum):
    NOT_IMPLEMENTED = 0
    NOT_AVAILABLE = 1
    WRITE_ONLY = 2
    READ_ONLY = 3
    READ_WRITE = 4

_VALUE_NODE_TYPES: frozenset[int]
DEFAULT_BLACKLISTED_NODES: frozenset[str]
_APPLY_PHASE_ORDER: tuple[tuple[str, ...], ...]
_PHASE_RESET_VALUES: dict[str, int | float | bool | str]

@dataclass(frozen=True, slots=True)
class GenicamNodeInfo:
    name: str
    value: int | float | str | bool

@dataclass
class GenicamConfiguration(YamlConfig):
    camera_model: str = ...
    camera_serial_number: str = ...
    nodes: list[GenicamNodeInfo] = field(default_factory=list)

def enumerate_genicam_nodes(node_map: NodeMap, blacklisted_nodes: frozenset[str] = ...) -> list[str]: ...
def read_genicam_node(node_map: NodeMap, name: str) -> GenicamNodeInfo: ...
def format_genicam_node(node_map: NodeMap, name: str) -> str: ...
def write_genicam_node(node_map: NodeMap, name: str, value: str) -> None: ...
def apply_genicam_configuration(
    node_map: NodeMap,
    config: GenicamConfiguration,
    current_model: str,
    current_serial: str,
    *,
    strict: bool = False,
    blacklisted_nodes: frozenset[str] = ...,
) -> None: ...
