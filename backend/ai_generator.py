import anthropic
from typing import List, Optional, Dict, Any


class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""

    MAX_TOOL_ROUNDS = 2

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to a comprehensive search tool for course information.

Tool Usage:
- Use the **search_course_content** tool for questions about specific course content or detailed educational materials
- Use the **get_course_outline** tool for questions about a course's structure, outline, syllabus, or lesson list. This returns the course title, course link, and each lesson's number and title — include all of these in your response.
- **Up to 2 tool calls per query** — use a second call only when the first result is insufficient or when you need information from a different source
- Synthesize tool results into accurate, fact-based responses
- If a tool yields no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without searching
- **Course-specific questions**: Search first, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
 - Do not mention "based on the search results"


All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""

    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

        # Pre-build base API parameters
        self.base_params = {"model": self.model, "temperature": 0, "max_tokens": 800}

    def generate_response(
        self,
        query: str,
        conversation_history: Optional[str] = None,
        tools: Optional[List] = None,
        tool_manager=None,
    ) -> str:
        """
        Generate AI response with optional tool usage and conversation context.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools

        Returns:
            Generated response as string
        """

        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        # Prepare API call parameters efficiently
        api_params = {
            **self.base_params,
            "messages": [{"role": "user", "content": query}],
            "system": system_content,
        }

        # Add tools if available
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}

        # Get response from Claude
        response = self.client.messages.create(**api_params)

        # Handle tool execution if needed
        if response.stop_reason == "tool_use" and tool_manager:
            return self._handle_tool_execution(response, api_params, tool_manager)

        # Return direct response
        return self._extract_text(response)

    @staticmethod
    def _extract_text(response) -> str:
        """Extract text from a response that may contain mixed content blocks."""
        for block in response.content:
            if block.type == "text":
                return block.text
        return ""

    def _handle_tool_execution(
        self, initial_response, base_params: Dict[str, Any], tool_manager
    ):
        """
        Handle up to MAX_TOOL_ROUNDS of sequential tool calls.

        Each round: execute tools from the current response, send results back.
        If the model requests more tools and rounds remain, continue looping.
        On the last round or after a tool error, omit tools to force a text response.
        """
        messages = base_params["messages"].copy()
        current_response = initial_response

        for round_num in range(self.MAX_TOOL_ROUNDS):
            # Append assistant's tool_use response
            messages.append({"role": "assistant", "content": current_response.content})

            # Execute tools, collect results
            tool_results = []
            tool_failed = False
            for block in current_response.content:
                if block.type == "tool_use":
                    try:
                        result = tool_manager.execute_tool(block.name, **block.input)
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": result,
                            }
                        )
                    except Exception as e:
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": str(e),
                                "is_error": True,
                            }
                        )
                        tool_failed = True

            messages.append({"role": "user", "content": tool_results})

            # Build follow-up params
            is_last_round = round_num == self.MAX_TOOL_ROUNDS - 1
            follow_up_params = {
                **self.base_params,
                "messages": messages,
                "system": base_params["system"],
            }
            if not is_last_round and not tool_failed:
                follow_up_params["tools"] = base_params["tools"]
                follow_up_params["tool_choice"] = {"type": "auto"}

            current_response = self.client.messages.create(**follow_up_params)

            # Stop if no more tool calls or a tool failed
            if current_response.stop_reason != "tool_use" or tool_failed:
                break

        return self._extract_text(current_response)
