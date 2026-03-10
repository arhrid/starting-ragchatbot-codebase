"""Tests for AIGenerator — all Anthropic API calls are mocked."""

import pytest
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Helpers ──────────────────────────────────────────────────────────


def _text_block(text="Hello"):
    return SimpleNamespace(type="text", text=text)


def _tool_use_block(name="search_course_content", input_data=None, tool_id="t1"):
    return SimpleNamespace(
        type="tool_use",
        name=name,
        input=input_data or {"query": "q"},
        id=tool_id,
    )


def _api_response(content, stop_reason="end_turn"):
    return SimpleNamespace(content=content, stop_reason=stop_reason)


# ── Tests ────────────────────────────────────────────────────────────


class TestAIGeneratorResponse:

    @patch("anthropic.Anthropic")
    def test_non_tool_response(self, MockAnthropic):
        """stop_reason='end_turn' → returns content text directly."""
        from ai_generator import AIGenerator

        mock_client = MockAnthropic.return_value
        mock_client.messages.create.return_value = _api_response(
            [_text_block("Hi there")], stop_reason="end_turn"
        )

        gen = AIGenerator(api_key="fake", model="test-model")
        result = gen.generate_response("hello")

        assert result == "Hi there"

    @patch("anthropic.Anthropic")
    def test_tool_use_flow(self, MockAnthropic):
        """tool_use → execute tool → second API call → final text."""
        from ai_generator import AIGenerator

        mock_client = MockAnthropic.return_value
        # First call returns tool_use
        mock_client.messages.create.side_effect = [
            _api_response([_tool_use_block()], stop_reason="tool_use"),
            _api_response([_text_block("Final answer")], stop_reason="end_turn"),
        ]

        tool_manager = MagicMock()
        tool_manager.execute_tool.return_value = "search results"

        gen = AIGenerator(api_key="fake", model="test-model")
        result = gen.generate_response(
            "question", tools=[{"name": "search_course_content"}], tool_manager=tool_manager
        )

        assert result == "Final answer"
        tool_manager.execute_tool.assert_called_once_with("search_course_content", query="q")

    @patch("anthropic.Anthropic")
    def test_tool_use_without_tool_manager(self, MockAnthropic):
        """stop_reason='tool_use' but tool_manager=None → tries content[0].text on ToolUseBlock → AttributeError."""
        from ai_generator import AIGenerator

        mock_client = MockAnthropic.return_value
        mock_client.messages.create.return_value = _api_response(
            [_tool_use_block()], stop_reason="tool_use"
        )

        gen = AIGenerator(api_key="fake", model="test-model")
        # _extract_text finds no text block → returns ""
        result = gen.generate_response("q", tools=[{"name": "t"}], tool_manager=None)
        assert result == ""

    @patch("anthropic.Anthropic")
    def test_tools_passed_in_api_params(self, MockAnthropic):
        """When tools are provided, they appear in API call along with tool_choice=auto."""
        from ai_generator import AIGenerator

        mock_client = MockAnthropic.return_value
        mock_client.messages.create.return_value = _api_response(
            [_text_block("ok")], stop_reason="end_turn"
        )

        tools_list = [{"name": "search_course_content"}]
        gen = AIGenerator(api_key="fake", model="test-model")
        gen.generate_response("q", tools=tools_list)

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["tools"] == tools_list
        assert call_kwargs["tool_choice"] == {"type": "auto"}

    @patch("anthropic.Anthropic")
    def test_conversation_history_in_system(self, MockAnthropic):
        """When history is provided, it's appended to the system prompt."""
        from ai_generator import AIGenerator

        mock_client = MockAnthropic.return_value
        mock_client.messages.create.return_value = _api_response(
            [_text_block("ok")], stop_reason="end_turn"
        )

        gen = AIGenerator(api_key="fake", model="test-model")
        gen.generate_response("q", conversation_history="User: hi\nAssistant: hello")

        call_kwargs = mock_client.messages.create.call_args[1]
        assert "Previous conversation:" in call_kwargs["system"]
        assert "User: hi" in call_kwargs["system"]

    @patch("anthropic.Anthropic")
    def test_api_error_propagates(self, MockAnthropic):
        """If client.messages.create raises, exception propagates."""
        from ai_generator import AIGenerator

        mock_client = MockAnthropic.return_value
        mock_client.messages.create.side_effect = Exception("API down")

        gen = AIGenerator(api_key="fake", model="test-model")

        with pytest.raises(Exception, match="API down"):
            gen.generate_response("q")


class TestSequentialToolCalls:

    @patch("anthropic.Anthropic")
    def test_two_sequential_tool_rounds(self, MockAnthropic):
        """tool_use → tool_use → text: 3 API calls, 2 tool executions."""
        from ai_generator import AIGenerator

        mock_client = MockAnthropic.return_value
        mock_client.messages.create.side_effect = [
            _api_response([_tool_use_block("get_course_outline", {"course": "X"}, "t1")], stop_reason="tool_use"),
            _api_response([_tool_use_block("search_course_content", {"query": "topic"}, "t2")], stop_reason="tool_use"),
            _api_response([_text_block("Combined answer")], stop_reason="end_turn"),
        ]

        tool_manager = MagicMock()
        tool_manager.execute_tool.side_effect = ["outline data", "search data"]

        tools_list = [{"name": "get_course_outline"}, {"name": "search_course_content"}]
        gen = AIGenerator(api_key="fake", model="test-model")
        result = gen.generate_response("complex question", tools=tools_list, tool_manager=tool_manager)

        assert result == "Combined answer"
        assert mock_client.messages.create.call_count == 3
        assert tool_manager.execute_tool.call_count == 2

    @patch("anthropic.Anthropic")
    def test_tool_error_sends_is_error_and_returns_text(self, MockAnthropic):
        """Tool exception → is_error result sent to Claude → final text returned."""
        from ai_generator import AIGenerator

        mock_client = MockAnthropic.return_value
        mock_client.messages.create.side_effect = [
            _api_response([_tool_use_block()], stop_reason="tool_use"),
            _api_response([_text_block("Sorry, I couldn't search")], stop_reason="end_turn"),
        ]

        tool_manager = MagicMock()
        tool_manager.execute_tool.side_effect = RuntimeError("tool broke")

        gen = AIGenerator(api_key="fake", model="test-model")
        result = gen.generate_response("q", tools=[{"name": "t"}], tool_manager=tool_manager)

        assert result == "Sorry, I couldn't search"

        # Verify the is_error result was sent
        second_call_kwargs = mock_client.messages.create.call_args_list[1][1]
        tool_result_msg = second_call_kwargs["messages"][-1]
        assert tool_result_msg["content"][0]["is_error"] is True
        assert "tool broke" in tool_result_msg["content"][0]["content"]

        # Error round should omit tools (force text)
        assert "tools" not in second_call_kwargs

    @patch("anthropic.Anthropic")
    def test_max_rounds_exhausted(self, MockAnthropic):
        """If model keeps requesting tools past MAX_TOOL_ROUNDS, loop stops and extracts text."""
        from ai_generator import AIGenerator

        mock_client = MockAnthropic.return_value
        # All responses are tool_use — the loop should stop at MAX_TOOL_ROUNDS
        mock_client.messages.create.side_effect = [
            _api_response([_tool_use_block(tool_id="t1")], stop_reason="tool_use"),
            _api_response([_tool_use_block(tool_id="t2")], stop_reason="tool_use"),
            # Last round omits tools, so model gives text
            _api_response([_text_block("gave up")], stop_reason="end_turn"),
        ]

        tool_manager = MagicMock()
        tool_manager.execute_tool.return_value = "data"

        gen = AIGenerator(api_key="fake", model="test-model")
        result = gen.generate_response("q", tools=[{"name": "t"}], tool_manager=tool_manager)

        assert result == "gave up"
        # 1 initial + 2 follow-ups = 3 total API calls
        assert mock_client.messages.create.call_count == 3
        assert tool_manager.execute_tool.call_count == 2

    @patch("anthropic.Anthropic")
    def test_follow_up_calls_include_tools(self, MockAnthropic):
        """Non-final follow-up API calls include tools param to allow another round."""
        from ai_generator import AIGenerator

        mock_client = MockAnthropic.return_value
        mock_client.messages.create.side_effect = [
            _api_response([_tool_use_block(tool_id="t1")], stop_reason="tool_use"),
            _api_response([_text_block("done after one round")], stop_reason="end_turn"),
        ]

        tool_manager = MagicMock()
        tool_manager.execute_tool.return_value = "result"

        tools_list = [{"name": "search_course_content"}]
        gen = AIGenerator(api_key="fake", model="test-model")
        gen.generate_response("q", tools=tools_list, tool_manager=tool_manager)

        # The second API call (first follow-up, round 0, not last round) should include tools
        second_call_kwargs = mock_client.messages.create.call_args_list[1][1]
        assert second_call_kwargs["tools"] == tools_list
        assert second_call_kwargs["tool_choice"] == {"type": "auto"}

    @patch("anthropic.Anthropic")
    def test_last_round_omits_tools(self, MockAnthropic):
        """The final follow-up API call (last round) omits tools to force text."""
        from ai_generator import AIGenerator

        mock_client = MockAnthropic.return_value
        mock_client.messages.create.side_effect = [
            _api_response([_tool_use_block(tool_id="t1")], stop_reason="tool_use"),
            _api_response([_tool_use_block(tool_id="t2")], stop_reason="tool_use"),
            _api_response([_text_block("final")], stop_reason="end_turn"),
        ]

        tool_manager = MagicMock()
        tool_manager.execute_tool.return_value = "data"

        gen = AIGenerator(api_key="fake", model="test-model")
        gen.generate_response("q", tools=[{"name": "t"}], tool_manager=tool_manager)

        # Third API call (round 1 = last round) should NOT have tools
        third_call_kwargs = mock_client.messages.create.call_args_list[2][1]
        assert "tools" not in third_call_kwargs

    @patch("anthropic.Anthropic")
    def test_follow_up_message_structure(self, MockAnthropic):
        """Verify message structure across two tool rounds."""
        from ai_generator import AIGenerator

        mock_client = MockAnthropic.return_value
        mock_client.messages.create.side_effect = [
            _api_response([_tool_use_block(tool_id="t1")], stop_reason="tool_use"),
            _api_response([_tool_use_block(tool_id="t2")], stop_reason="tool_use"),
            _api_response([_text_block("done")], stop_reason="end_turn"),
        ]

        tool_manager = MagicMock()
        tool_manager.execute_tool.return_value = "data"

        gen = AIGenerator(api_key="fake", model="test-model")
        gen.generate_response("q", tools=[{"name": "t"}], tool_manager=tool_manager)

        # Third call should have: user, assistant(t1), user(result1), assistant(t2), user(result2)
        third_call_kwargs = mock_client.messages.create.call_args_list[2][1]
        messages = third_call_kwargs["messages"]
        assert len(messages) == 5
        assert [m["role"] for m in messages] == ["user", "assistant", "user", "assistant", "user"]
