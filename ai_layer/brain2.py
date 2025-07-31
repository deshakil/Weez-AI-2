"""
brain.py — Enterprise-grade ReAct-style reasoning agent for Weezy MCP AI Agent (Azure OpenAI).

This module coordinates:
  • Intent parsing (`intent_parser.parse_intent`)
  • Embedding generation (`embedder.get_query_embedding`)
  • Tool execution: search, summarize, rag
  • Per-user conversational + semantic memory (`CosmosMemoryManager`)
  • Azure OpenAI Chat Completions with *function calling* (aka tools) in a ReAct loop

Design goals
============
• **Deterministic orchestration**: Brain decides when to delegate vs directly tool-call.
• **Model-led tool routing**: Uses ChatCompletions w/ tools schemas so the model can choose.
• **Multi-step ReAct**: Model may call multiple tools sequentially; loop limited by `max_reasoning_steps`.
• **Graceful clarification**: If `parse_intent` returns `needs_clarification`, we immediately ask user.
• **Memory augmentation**: Recent queries + tool results are injected as conversation context to improve grounding.
• **Robust error handling**: Captures tool errors, surfaces helpful fallback messages to the model and user.

Azure OpenAI Notes
==================
You must configure the `AzureOpenAI` client outside this module and pass it to `initialize_brain(...)`.
For Azure, the *model* argument to `chat.completions.create` is the **deployment name** you configured in Azure (e.g., "gpt-4o").

Message Protocol
================
We use the newer *tools* API structure:
  tools=[{"type":"function","function":{...}}]
Model responses may include `message.tool_calls` (a list). For each tool call we execute the mapped python function,
append a `role="tool"` message with the JSON result, then re-call the model.
We stop when model returns a message **without** tool calls or when `max_reasoning_steps` reached.

Usage
=====
>>> from openai import AzureOpenAI
>>> client = AzureOpenAI(api_key=..., api_version=..., azure_endpoint=...)
>>> initialize_brain(client, chat_deployment="gpt-4o")
>>> reply = reason_and_act(user_id="user123", user_input="Summarize yesterday's design meeting notes from Google Drive", conversation_id="conv-123")
print(reply)

"""

from __future__ import annotations

import json
import logging
import traceback
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Callable

from openai import AzureOpenAI

# --- Module Imports ---------------------------------------------------------
from .intent_parser import parse_user_intent
from .embedder import get_query_embedding

# tools.py is expected to expose TOOL_FUNCTIONS, a dict keyed by function name:
# {
#   "search": {"function": <callable>, "spec": {"name": "search", "description": "...", "parameters": {...}}},
#   "summarize": {"function": <callable>, "spec": {...}},
#   "rag": {"function": <callable>, "spec": {...}},
# }
from .tools2 import TOOL_FUNCTIONS, dispatch_tool

# Import the Cosmos DB memory manager
from .memory import CosmosMemoryManager

# ---------------------------------------------------------------------------
# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utility: extract tool schemas + callables from TOOL_FUNCTIONS

def _extract_tool_schemas() -> List[Dict[str, Any]]:
    """Return a list of tool schema dicts in the format expected by Chat Completions."""
    schemas: List[Dict[str, Any]] = []
    for name, meta in TOOL_FUNCTIONS.items():
        # tools.py uses 'spec' key, not 'schema'
        spec = meta.get("spec")
        if not spec:
            logger.warning("Tool %s missing spec; skipping.", name)
            continue
        
        # Convert the spec to the format expected by Azure OpenAI
        schema = {
            "type": "function",
            "function": spec
        }
        schemas.append(schema)
    return schemas


def _extract_tool_callables() -> Dict[str, Callable[..., Any]]:
    mapping: Dict[str, Callable[..., Any]] = {}
    for name, meta in TOOL_FUNCTIONS.items():
        fn = meta.get("function")
        if callable(fn):
            mapping[name] = fn
        else:
            logger.warning("Tool %s has non-callable function entry.", name)
    return mapping


# ---------------------------------------------------------------------------
# ReAct Brain

class ReActBrain:
    """Reason + Act orchestration layer using Azure OpenAI function calling."""

    def __init__(
        self,
        azure_openai_client: AzureOpenAI,
        chat_deployment: str,
        memory_manager: Optional[CosmosMemoryManager] = None,
        max_reasoning_steps: int = 5,
        temperature: float = 0.1,
        conversation_history_limit: int = 10,
    ) -> None:
        self.client = azure_openai_client
        self.model = chat_deployment  # Azure deployment name
        self.memory_manager = memory_manager or CosmosMemoryManager()
        self.max_reasoning_steps = max_reasoning_steps
        self.temperature = temperature
        self.conversation_history_limit = conversation_history_limit

        self.tool_mapping = _extract_tool_callables()
        self.tool_schemas = _extract_tool_schemas()

    # ------------------------------------------------------------------
    # Public API
    def reason_and_act(self, user_id: str, user_input: str, conversation_id: Optional[str] = None) -> str:
        """Main entry point. Returns final user-facing response string."""
        logger.info("ReActBrain.start user=%s input=%s conversation_id=%s", user_id, user_input, conversation_id)

        # Generate conversation_id if not provided
        if not conversation_id:
            conversation_id = str(uuid.uuid4())
            logger.info("Generated new conversation_id: %s", conversation_id)

        # Parse intent --------------------------------------------------
        intent = self._safe_parse_intent(user_input)

        # Clarification gate -------------------------------------------
        if intent.get("needs_clarification"):
            clarification_msg = self._clarification_message(intent)
            # Store clarification request
            self._store_conversation(user_id, conversation_id, user_input, clarification_msg)
            return clarification_msg

        # Generate embedding -----------------------------------
        query_text = intent.get("query_text") or user_input
        embedding = self._safe_get_embedding(query_text)

        # Build context -------------------------------------------------
        messages = self._build_conversation_messages(
            user_id=user_id, 
            conversation_id=conversation_id,
            user_input=user_input, 
            intent=intent, 
            embedding=embedding
        )

        # ReAct loop ----------------------------------------------------
        reply = self._react_loop(user_id=user_id, conversation_id=conversation_id, messages=messages, intent=intent)
        
        # Store the complete conversation in memory
        self._store_conversation(user_id, conversation_id, user_input, reply)
        
        return reply

    # ------------------------------------------------------------------
    # Internal helpers
    def _safe_parse_intent(self, user_input: str) -> Dict[str, Any]:
        try:
            return parse_user_intent(user_input) or {}
        except Exception as e:  # fallback default minimal intent
            logger.error("Intent parsing failed: %s", e)
            logger.debug(traceback.format_exc())
            return {
                "action": "search",
                "query_text": user_input,
                "needs_clarification": False,
            }

    def _clarification_message(self, intent: Dict[str, Any]) -> str:
        reason = intent.get("clarification_reason")
        base = "I need a bit more information to help you effectively."
        if reason:
            base += f" {reason}"
        base += " Could you please clarify what you need (topic, file, platform, or format)?"
        return base

    def _safe_get_embedding(self, text: str) -> List[float]:
        try:
            return get_query_embedding(text)
        except Exception as e:
            logger.error("Embedding generation failed: %s", e)
            logger.debug(traceback.format_exc())
            return []

    def _build_conversation_messages(
        self, 
        user_id: str, 
        conversation_id: str,
        user_input: str, 
        intent: Dict[str, Any], 
        embedding: List[float]
    ) -> List[Dict[str, Any]]:
        """Build conversation messages including system prompt and conversation history."""
        messages: List[Dict[str, Any]] = []
        
        # Add system prompt
        messages.append({
            "role": "system", 
            "content": self._system_prompt(intent=intent, embedding=embedding)
        })
        
        # Get conversation history from Cosmos DB for this specific conversation
        try:
            history = self.memory_manager.get_conversation_history(
                user_id=user_id, 
                conversation_id=conversation_id,
                limit=self.conversation_history_limit
            )
            
            # Convert Cosmos DB history to chat format
            # History comes back in chronological order (ascending)
            for conversation in history:
                # Add user message
                messages.append({
                    "role": "user",
                    "content": conversation.get("user_query", "")
                })
                
                # Add assistant response
                messages.append({
                    "role": "assistant",
                    "content": conversation.get("agent_response", "")
                })
                
        except Exception as e:
            logger.error("Failed to retrieve conversation history: %s", e)
            logger.debug(traceback.format_exc())
            # Continue without history if retrieval fails
        
        # Add current user message
        messages.append({"role": "user", "content": user_input})
        
        return messages

    def _system_prompt(self, intent: Dict[str, Any], embedding: List[float]) -> str:
        """Generate system prompt with context."""
        # Keep short; models perform better w/ concise instructions.
        prompt = (
            "You are Weezy MCP's enterprise AI reasoning agent. "
            "Use the available function tools to gather info (search, summarize, rag) before answering. "
            "Think step-by-step: decide if you need to call a tool; if so, return a tool call. "
            "After tools return, synthesize a clear answer citing the tool results (do not hallucinate). "
            "Be helpful, accurate, and concise in your responses."
        )
        
        # Add lightweight context injection
        if intent:
            prompt += f"\n\nIntent context: action={intent.get('action')}, query={intent.get('query_text')}"
            if intent.get('platform'):
                prompt += f", platform={intent.get('platform')}"
            if intent.get('mime_type'):
                prompt += f", mime_type={intent.get('mime_type')}"
        
        if embedding:
            prompt += f"\n\nEmbedding available (length: {len(embedding)}) for semantic search."
        
        prompt += "\n\nReturn responses in markdown format when appropriate."
        return prompt

    def _store_conversation(self, user_id: str, conversation_id: str, user_query: str, agent_response: str) -> None:
        """Store conversation in Cosmos DB with proper conversation threading."""
        try:
            self.memory_manager.store_conversation(
                user_id=user_id,
                conversation_id=conversation_id,
                user_query=user_query,
                agent_response=agent_response,
                timestamp=datetime.now(timezone.utc).isoformat()
            )
            logger.info("Stored conversation for user: %s, conversation: %s", user_id, conversation_id)
        except Exception as e:
            logger.error("Failed to store conversation: %s", e)
            logger.debug(traceback.format_exc())
            # Don't fail the entire response if storage fails

    # ------------------------------------------------------------------
    # Core ReAct loop w/ tool calling
    def _react_loop(self, user_id: str, conversation_id: str, messages: List[Dict[str, Any]], intent: Dict[str, Any]) -> str:
        """Iteratively call the model; execute tool calls until done or step limit reached."""
        steps = 0
        while steps < self.max_reasoning_steps:
            steps += 1
            logger.debug("ReAct step %s messages_len=%s", steps, len(messages))
            
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=self.tool_schemas,
                    tool_choice="auto",
                    temperature=self.temperature,
                )
            except Exception as e:
                logger.error("Azure OpenAI chat call failed: %s", e)
                logger.debug(traceback.format_exc())
                return "I ran into an error contacting the language model. Please try again."

            msg = resp.choices[0].message

            # If the model returned tool calls, execute them
            tool_calls = getattr(msg, "tool_calls", None) or []
            if tool_calls:
                # Add the assistant message that triggered the tool calls
                messages.append({
                    "role": "assistant",
                    "content": msg.content or None,
                    "tool_calls": [
                        tc.model_dump() if hasattr(tc, "model_dump") else _tool_call_to_dict(tc) 
                        for tc in tool_calls
                    ],
                })

                # Execute each tool call sequentially
                for tc in tool_calls:
                    name = (getattr(tc.function, "name", None) if hasattr(tc, "function") 
                           else tc.get("function", {}).get("name"))
                    arg_str = (getattr(tc.function, "arguments", "{}") if hasattr(tc, "function") 
                              else tc.get("function", {}).get("arguments", "{}"))
                    args = self._safe_json_loads(arg_str)

                    logger.info("Executing tool %s with args: %s", name, args)
                    result = self._dispatch_tool(
                        user_id=user_id, 
                        conversation_id=conversation_id,
                        tool_name=name, 
                        args=args, 
                        intent=intent
                    )

                    # Append tool result message
                    tool_call_id = (getattr(tc, "id", None) if hasattr(tc, "id") 
                                   else tc.get("id"))
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "name": name,
                        "content": json.dumps(result, ensure_ascii=False),
                    })
                
                # Continue loop to let model observe tool outputs
                continue

            # No tool calls => final answer
            content = msg.content or "I don't have further information."
            logger.info("ReActBrain.final steps=%s", steps)
            return content.strip()

        # Step limit hit; ask model to summarize
        logger.warning("ReActBrain reached max_reasoning_steps=%s; forcing finalization.", self.max_reasoning_steps)
        try:
            messages.append({
                "role": "user", 
                "content": "Please provide your best final answer based on all tool results so far."
            })
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
            )
            return (resp.choices[0].message.content or "(No response)").strip()
        except Exception as e:
            logger.error("Finalization call failed: %s", e)
            logger.debug(traceback.format_exc())
            return "I've gathered information but couldn't generate a final response. Please retry."

    # ------------------------------------------------------------------
    # Tool dispatch + error safety (UPDATED to use new tools.py)
    def _dispatch_tool(
        self, 
        user_id: str, 
        conversation_id: str,
        tool_name: Optional[str], 
        args: Dict[str, Any], 
        intent: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a tool function with error handling using the new dispatch_tool function."""
        if not tool_name:
            return {"success": False, "error": "Missing tool name."}

        # Ensure user_id is always provided to tools
        if "user_id" not in args:
            args["user_id"] = user_id

        # Augment args with intent defaults if missing
        if "platform" not in args and intent.get("platform") is not None:
            args["platform"] = intent["platform"]
        if "mime_type" not in args and intent.get("mime_type") is not None:
            args["mime_type"] = intent["mime_type"]
        if "file_type" not in args and intent.get("file_type") is not None:
            args["file_type"] = intent["file_type"]

        # Tool-specific argument handling
        if tool_name == "summarize" and "summary_type" not in args and intent.get("summary_type"):
            args["summary_type"] = intent["summary_type"]

        # Ensure query_text is provided for tools that need it
        if "query_text" not in args:
            args["query_text"] = intent.get("query_text") or ""

        # Add conversation_id for RAG tools that might need it
        if tool_name in ["rag", "conversation_history"] and "conversation_id" not in args:
            args["conversation_id"] = conversation_id

        try:
            # Use the centralized dispatch_tool function from tools.py
            result = dispatch_tool(tool_name, args)
            
            # Check if the result indicates an error
            if isinstance(result, dict) and result.get("error"):
                logger.warning("Tool %s returned error: %s", tool_name, result.get("error"))
                return {
                    "success": False, 
                    "function": tool_name, 
                    "error": result.get("error"),
                    "details": result
                }
            
            wrapped_result = {
                "success": True, 
                "function": tool_name, 
                "result": result
            }
            
            # Store tool result as a separate conversation entry for context
            self._store_tool_result(user_id, conversation_id, tool_name, result)
            
            return wrapped_result
            
        except Exception as e:
            logger.error("Tool %s execution error: %s", tool_name, e)
            logger.debug(traceback.format_exc())
            return {"success": False, "function": tool_name, "error": str(e)}

    def _store_tool_result(self, user_id: str, conversation_id: str, tool_name: str, result: Any) -> None:
        """Store tool execution result for context in future conversations."""
        try:
            # Create a summary of tool result for storage
            tool_summary = f"Tool '{tool_name}' executed"
            if isinstance(result, dict):
                if "message" in result:
                    tool_summary += f": {result['message']}"
                elif "summary" in result:
                    tool_summary += f": {result['summary']}"
                elif "answer" in result:
                    tool_summary += f": {result['answer'][:200]}..."
                elif "total_results" in result:
                    tool_summary += f" with {result['total_results']} results"
                else:
                    tool_summary += f" with {len(result)} result items"
            else:
                tool_summary += f": {str(result)[:200]}..."

            # Store as a system message for context using the same conversation_id
            # Generate a unique conversation_id for tool results to avoid confusion
            tool_conversation_id = f"{conversation_id}_tool_{tool_name}_{uuid.uuid4().hex[:8]}"
            
            self.memory_manager.store_conversation(
                user_id=user_id,
                conversation_id=tool_conversation_id,
                user_query=f"[TOOL_EXECUTION] {tool_name}",
                agent_response=tool_summary,
                timestamp=datetime.now(timezone.utc).isoformat()
            )
        except Exception as e:
            logger.error("Failed to store tool result: %s", e)
            # Don't fail the main operation if tool result storage fails

    # ------------------------------------------------------------------
    @staticmethod
    def _safe_json_loads(data: Any) -> Dict[str, Any]:
        """Safely parse JSON data with fallbacks."""
        if isinstance(data, dict):
            return data
        if not data:
            return {}
        try:
            return json.loads(data)
        except Exception:
            try:
                # Attempt to coerce python-style dict string
                cleaned = data.replace("'", '"')
                return json.loads(cleaned)
            except Exception:
                logger.warning("Failed to parse tool args: %s", data)
                return {}


# ---------------------------------------------------------------------------
# Global singleton wiring (optional convenience)
_brain_instance: Optional[ReActBrain] = None


def initialize_brain(
    azure_openai_client: AzureOpenAI,
    chat_deployment: str = "gpt-4o",
    memory_manager: Optional[CosmosMemoryManager] = None,
    max_reasoning_steps: int = 5,
    temperature: float = 0.1,
    conversation_history_limit: int = 10,
) -> None:
    """Initialize the global ReAct brain instance."""
    global _brain_instance
    _brain_instance = ReActBrain(
        azure_openai_client=azure_openai_client,
        chat_deployment=chat_deployment,
        memory_manager=memory_manager,
        max_reasoning_steps=max_reasoning_steps,
        temperature=temperature,
        conversation_history_limit=conversation_history_limit,
    )


def reason_and_act(user_id: str, user_input: str, conversation_id: Optional[str] = None) -> str:
    """Global convenience wrapper."""
    if _brain_instance is None:
        raise RuntimeError("Brain not initialized. Call initialize_brain() first.")
    return _brain_instance.reason_and_act(user_id=user_id, user_input=user_input, conversation_id=conversation_id)


# ---------------------------------------------------------------------------
# Back-compat CLI smoke test
if __name__ == "__main__":  # pragma: no cover
    import os

    # Minimal environment-driven client setup
    _api_key = os.getenv("AZURE_OPENAI_API_KEY", "YOUR-KEY")
    _endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "https://YOUR-RESOURCE.openai.azure.com/")
    _api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
    client = AzureOpenAI(api_key=_api_key, azure_endpoint=_endpoint, api_version=_api_version)

    initialize_brain(client, chat_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o"))

    print("Type a query (Ctrl+C to quit)...")
    try:
        while True:
            q = input("> ").strip()
            if not q:
                continue
            try:
                # Generate a conversation ID for this session
                conv_id = str(uuid.uuid4())
                print(reason_and_act("demo", q, conv_id))
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print("Error:", e)
    except KeyboardInterrupt:
        print("\nBye!")


# ---------------------------------------------------------------------------
# Local helper for tool call dict fallback

def _tool_call_to_dict(tc: Any) -> Dict[str, Any]:
    """Convert tool call object to dict format."""
    try:
        return {
            "id": getattr(tc, "id", None),
            "type": getattr(tc, "type", "function"),
            "function": {
                "name": getattr(getattr(tc, "function", None), "name", None),
                "arguments": getattr(getattr(tc, "function", None), "arguments", "{}"),
            },
        }
    except Exception:
        return {"id": None, "type": "function", "function": {"name": None, "arguments": "{}"}}
