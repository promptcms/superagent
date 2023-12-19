from datetime import datetime
from typing import Any, Dict, List, Optional

from langfuse import Langfuse
from langfuse.model import CreateTrace, CreateEvent, CreateGeneration, UpdateGeneration
from llama_index.callbacks.base_handler import BaseCallbackHandler
from llama_index.callbacks.schema import (
    CBEvent,
    CBEventType,
    BASE_TRACE_EVENT, EventPayload,
)

langfuse = Langfuse()

class LangfuseHandler(BaseCallbackHandler):
    """Callback handler that sends LLM events to Langfuse.

    Args:
        event_starts_to_ignore (Optional[List[CBEventType]]): list of event types to
            ignore when tracking event starts.
        event_ends_to_ignore (Optional[List[CBEventType]]): list of event types to
            ignore when tracking event ends.

    """

    def __init__(
        self,
        event_starts_to_ignore: Optional[List[CBEventType]] = None,
        event_ends_to_ignore: Optional[List[CBEventType]] = None,
        agent_id: Optional[str] = None,
        api_user_id: Optional[str] = None,
        debug: bool = True,
    ) -> None:
        """Initialize the llama debug handler."""
        self.langfuse = langfuse
        self._lf_object_map: Dict[str, any] = {}
        self.agent_id = agent_id
        self.api_user_id = api_user_id
        self.debug = debug
        event_starts_to_ignore = (
            event_starts_to_ignore if event_starts_to_ignore else []
        )
        event_ends_to_ignore = event_ends_to_ignore if event_ends_to_ignore else []
        super().__init__(
            event_starts_to_ignore=event_starts_to_ignore,
            event_ends_to_ignore=event_ends_to_ignore,
        )

    def start_trace(self, trace_id: Optional[str] = None) -> None:
        """Launch a trace."""
        if self.debug:
            print("starting trace {}".format(trace_id))
        trace = self.langfuse.trace(CreateTrace(name="rag-invoke",
                                                user_id=self.api_user_id,
                                                metadata={"agent_id": self.agent_id}))
        self._lf_object_map[BASE_TRACE_EVENT] = trace

    def end_trace(
        self,
        trace_id: Optional[str] = None,
        trace_map: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        """Shutdown the current trace."""
        if self.debug:
            print("ending trace {}".format(trace_id))
        self._lf_object_map.pop(BASE_TRACE_EVENT)
        self.langfuse.flush()


    def on_event_start(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs: Any,
    ) -> str:
        """On Event Start

        Args:
            event_type (CBEventType): event type to store.
            payload (Optional[Dict[str, Any]]): payload to store.
            event_id (str): event id to store.
            parent_id (str): parent event id.

        """
        if self.debug:
            print("on_event_start {}".format(event_type))
            print("event: {}".format(str(payload)))
        event = CBEvent(event_type, payload=payload, id_=event_id)
        parent = self._lf_object_map[parent_id]
        if event_type == CBEventType.LLM:
            messages = payload.get(EventPayload.MESSAGES, [])
            prompt = payload.get(EventPayload.PROMPT, "") or messages[:-1]
            response_text = payload.get(EventPayload.RESPONSE, "") or payload.get(EventPayload.COMPLETION, "") or messages[-1] or ""
            model_parameters = payload.get(EventPayload.SERIALIZED, {})
            # Sanitize any nested parameters
            for key, value in model_parameters.items():
                if not isinstance(value, (bool, int, str, type(None))):
                    model_parameters[key] = str(value)
            self._lf_object_map[event_id] = parent.generation(CreateGeneration(name="OpenAI Chat", prompt=prompt, completion=response_text, model=str(model_parameters.get("model", "")), model_parameters=model_parameters))
        elif event_type == CBEventType.EXCEPTION:
            exception: Exception = payload.get(EventPayload.EXCEPTION)
            exception_name = str(exception.__class__.__name__)
            self._lf_object_map[event_id] = parent.event(CreateEvent(name=exception_name, metadata=str(exception)))
        return event_id

    def on_event_end(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        """On event end.

        Args:
            event_type (CBEventType): event type to store.
            payload (Optional[Dict[str, Any]]): payload to store.
            event_id (str): event id to store.

        """
        if self.debug:
            print("on_event_end {}".format(event_type))
        event = self._lf_object_map.pop(event_id)
        if event_type == CBEventType.LLM:
            event.update(UpdateGeneration(end_time=datetime.utcnow()))
