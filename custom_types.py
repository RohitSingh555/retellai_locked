
from typing import Any, List, Optional, Literal, Union
from pydantic import BaseModel
from typing import Literal, Dict, Optional


# Retell -> Your Server Events
class Utterance(BaseModel):
    role: Literal["agent", "user", "system"]
    content: str


class PingPongRequest(BaseModel):
    interaction_type: Literal["ping_pong"]
    timestamp: int


class CallDetailsRequest(BaseModel):
    interaction_type: Literal["call_details"]
    call: dict


class UpdateOnlyRequest(BaseModel):
    interaction_type: Literal["update_only"]
    transcript: List[Utterance]


class ResponseRequiredRequest(BaseModel):
    interaction_type: Literal["reminder_required", "response_required"]
    response_id: int
    transcript: List[Utterance]


CustomLlmRequest = Union[
    ResponseRequiredRequest | UpdateOnlyRequest | CallDetailsRequest | PingPongRequest
]


# Your Server -> Retell Events
class ConfigResponse(BaseModel):
    response_type: Literal["config"] = "config"
    config: Dict[str, bool] = {
        "auto_reconnect": bool,
        "call_details": bool,
    }


class PingPongResponse(BaseModel):
    response_type: Literal["ping_pong"] = "ping_pong"
    timestamp: int


class ResponseResponse(BaseModel):
    response_type: Literal["response"] = "response"
    response_id: int
    content: str
    content_complete: bool
    end_call: Optional[bool] = False
    transfer_number: Optional[str] = None


CustomLlmResponse = Union[ConfigResponse | PingPongResponse | ResponseResponse]

class Utterance(BaseModel):
    role: Literal["agent", "user", "system"]
    content: str

class CustomLlmRequest(BaseModel):
    interaction_type: Literal["update_only", "response_required", "reminder_required", "ping_pong", "call_details"]
    response_id: Optional[int] = 0 # Used by response_required and reminder_required
    transcript: Optional[List[Any]] = [] # Used by response_required and reminder_required
    call: Optional[dict] = None # Used by call_details
    timestamp: Optional[int] = None # Used by ping_pong

class CustomLlmResponse(BaseModel):
    response_type: Literal["response", "config", "ping_pong"] = "response"
    response_id: Optional[int] = None # Used by response
    content: Any = None # Used by response
    content_complete: Optional[bool] = False # Used by response
    end_call: Optional[bool] = False # Used by response
    config: Optional[dict] = None # Used by config
    timestamp: Optional[int] = None # Used by ping_pong
