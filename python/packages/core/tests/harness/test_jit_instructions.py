# Copyright (c) Microsoft. All rights reserved.

"""Tests for the JIT instructions system (Phase 7)."""

from agent_framework._harness._jit_instructions import (
    DEFAULT_JIT_INSTRUCTIONS,
    JitContext,
    JitInstruction,
    JitInstructionProcessor,
)

# ---------------------------------------------------------------------------
# JitContext tests
# ---------------------------------------------------------------------------


class TestJitContext:
    """Tests for JitContext dataclass."""

    def test_basic_construction(self) -> None:
        ctx = JitContext(
            turn=5,
            max_turns=20,
            tool_usage={"read_file": 3},
            work_items_complete=1,
            work_items_total=5,
        )
        assert ctx.turn == 5
        assert ctx.max_turns == 20
        assert ctx.tool_usage == {"read_file": 3}
        assert ctx.work_items_complete == 1
        assert ctx.work_items_total == 5


# ---------------------------------------------------------------------------
# JitInstructionProcessor tests
# ---------------------------------------------------------------------------


class TestJitInstructionProcessor:
    """Tests for the JIT instruction processor."""

    def test_no_instructions_fires_nothing(self) -> None:
        processor = JitInstructionProcessor(instructions=[])
        ctx = JitContext(turn=10, max_turns=20, tool_usage={}, work_items_complete=0, work_items_total=0)
        assert processor.evaluate(ctx) == []

    def test_condition_true_fires(self) -> None:
        instr = JitInstruction(
            id="test",
            instruction="Do something",
            condition=lambda ctx: True,
        )
        processor = JitInstructionProcessor(instructions=[instr])
        ctx = JitContext(turn=1, max_turns=20, tool_usage={}, work_items_complete=0, work_items_total=0)
        result = processor.evaluate(ctx)
        assert result == ["Do something"]

    def test_condition_false_does_not_fire(self) -> None:
        instr = JitInstruction(
            id="test",
            instruction="Do something",
            condition=lambda ctx: False,
        )
        processor = JitInstructionProcessor(instructions=[instr])
        ctx = JitContext(turn=1, max_turns=20, tool_usage={}, work_items_complete=0, work_items_total=0)
        assert processor.evaluate(ctx) == []

    def test_once_true_fires_only_once(self) -> None:
        instr = JitInstruction(
            id="test_once",
            instruction="Only once",
            condition=lambda ctx: True,
            once=True,
        )
        processor = JitInstructionProcessor(instructions=[instr])
        ctx = JitContext(turn=1, max_turns=20, tool_usage={}, work_items_complete=0, work_items_total=0)

        result1 = processor.evaluate(ctx)
        assert result1 == ["Only once"]

        result2 = processor.evaluate(ctx)
        assert result2 == []

    def test_once_false_fires_every_time(self) -> None:
        instr = JitInstruction(
            id="test_repeat",
            instruction="Every time",
            condition=lambda ctx: True,
            once=False,
        )
        processor = JitInstructionProcessor(instructions=[instr])
        ctx = JitContext(turn=1, max_turns=20, tool_usage={}, work_items_complete=0, work_items_total=0)

        result1 = processor.evaluate(ctx)
        assert result1 == ["Every time"]

        result2 = processor.evaluate(ctx)
        assert result2 == ["Every time"]

    def test_callable_instruction(self) -> None:
        instr = JitInstruction(
            id="dynamic",
            instruction=lambda ctx: f"Turn {ctx.turn} of {ctx.max_turns}",
            condition=lambda ctx: True,
        )
        processor = JitInstructionProcessor(instructions=[instr])
        ctx = JitContext(turn=5, max_turns=20, tool_usage={}, work_items_complete=0, work_items_total=0)
        result = processor.evaluate(ctx)
        assert result == ["Turn 5 of 20"]

    def test_multiple_instructions_fire_together(self) -> None:
        instr1 = JitInstruction(id="a", instruction="A", condition=lambda ctx: True)
        instr2 = JitInstruction(id="b", instruction="B", condition=lambda ctx: True)
        instr3 = JitInstruction(id="c", instruction="C", condition=lambda ctx: False)
        processor = JitInstructionProcessor(instructions=[instr1, instr2, instr3])
        ctx = JitContext(turn=1, max_turns=20, tool_usage={}, work_items_complete=0, work_items_total=0)
        result = processor.evaluate(ctx)
        assert result == ["A", "B"]

    def test_condition_error_is_swallowed(self) -> None:
        def bad_condition(ctx: JitContext) -> bool:
            raise ValueError("boom")

        instr = JitInstruction(id="bad", instruction="Never", condition=bad_condition)
        processor = JitInstructionProcessor(instructions=[instr])
        ctx = JitContext(turn=1, max_turns=20, tool_usage={}, work_items_complete=0, work_items_total=0)
        result = processor.evaluate(ctx)
        assert result == []

    def test_default_instructions_loaded(self) -> None:
        processor = JitInstructionProcessor()
        assert len(processor.instructions) == len(DEFAULT_JIT_INSTRUCTIONS)


# ---------------------------------------------------------------------------
# Default JIT instruction condition tests
# ---------------------------------------------------------------------------


class TestDefaultInstructions:
    """Tests for the built-in default JIT instructions."""

    def _make_ctx(self, **kwargs: object) -> JitContext:
        defaults = {
            "turn": 1,
            "max_turns": 20,
            "tool_usage": {},
            "work_items_complete": 0,
            "work_items_total": 0,
        }
        defaults.update(kwargs)
        return JitContext(**defaults)  # type: ignore[arg-type]

    def _find_instruction(self, instruction_id: str) -> JitInstruction:
        for instr in DEFAULT_JIT_INSTRUCTIONS:
            if instr.id == instruction_id:
                return instr
        raise AssertionError(f"No instruction with id '{instruction_id}'")

    # --- no_reads_after_5_turns ---

    def test_no_reads_fires_at_turn_5(self) -> None:
        instr = self._find_instruction("no_reads_after_5_turns")
        ctx = self._make_ctx(turn=5, tool_usage={})
        assert instr.condition(ctx) is True

    def test_no_reads_does_not_fire_before_turn_5(self) -> None:
        instr = self._find_instruction("no_reads_after_5_turns")
        ctx = self._make_ctx(turn=4, tool_usage={})
        assert instr.condition(ctx) is False

    def test_no_reads_does_not_fire_if_reads_exist(self) -> None:
        instr = self._find_instruction("no_reads_after_5_turns")
        ctx = self._make_ctx(turn=5, tool_usage={"read_file": 1})
        assert instr.condition(ctx) is False

    # --- no_writes_after_reads ---

    def test_no_writes_fires_when_many_reads_no_writes(self) -> None:
        instr = self._find_instruction("no_writes_after_reads")
        ctx = self._make_ctx(turn=10, tool_usage={"read_file": 5})
        assert instr.condition(ctx) is True

    def test_no_writes_does_not_fire_early(self) -> None:
        instr = self._find_instruction("no_writes_after_reads")
        ctx = self._make_ctx(turn=9, tool_usage={"read_file": 5})
        assert instr.condition(ctx) is False

    def test_no_writes_does_not_fire_if_writes_exist(self) -> None:
        instr = self._find_instruction("no_writes_after_reads")
        ctx = self._make_ctx(turn=10, tool_usage={"read_file": 5, "write_file": 1})
        assert instr.condition(ctx) is False

    def test_no_writes_does_not_fire_with_few_reads(self) -> None:
        instr = self._find_instruction("no_writes_after_reads")
        ctx = self._make_ctx(turn=10, tool_usage={"read_file": 4})
        assert instr.condition(ctx) is False

    # --- approaching_turn_limit ---

    def test_approaching_limit_fires_at_80_percent(self) -> None:
        instr = self._find_instruction("approaching_turn_limit")
        ctx = self._make_ctx(turn=16, max_turns=20)
        assert instr.condition(ctx) is True

    def test_approaching_limit_does_not_fire_before_80_percent(self) -> None:
        instr = self._find_instruction("approaching_turn_limit")
        ctx = self._make_ctx(turn=15, max_turns=20)
        assert instr.condition(ctx) is False

    # --- all_planning_no_execution ---

    def test_planning_no_execution_fires(self) -> None:
        instr = self._find_instruction("all_planning_no_execution")
        ctx = self._make_ctx(
            turn=3,
            tool_usage={"work_item_add": 3},
            work_items_total=3,
            work_items_complete=0,
        )
        assert instr.condition(ctx) is True

    def test_planning_no_execution_does_not_fire_before_turn_3(self) -> None:
        instr = self._find_instruction("all_planning_no_execution")
        ctx = self._make_ctx(
            turn=2,
            tool_usage={"work_item_add": 3},
            work_items_total=3,
            work_items_complete=0,
        )
        assert instr.condition(ctx) is False

    def test_planning_no_execution_does_not_fire_if_reads_exist(self) -> None:
        instr = self._find_instruction("all_planning_no_execution")
        ctx = self._make_ctx(
            turn=3,
            tool_usage={"work_item_add": 3, "read_file": 1},
            work_items_total=3,
            work_items_complete=0,
        )
        assert instr.condition(ctx) is False

    def test_planning_no_execution_does_not_fire_if_items_complete(self) -> None:
        instr = self._find_instruction("all_planning_no_execution")
        ctx = self._make_ctx(
            turn=3,
            tool_usage={"work_item_add": 3},
            work_items_total=3,
            work_items_complete=1,
        )
        assert instr.condition(ctx) is False


# ---------------------------------------------------------------------------
# Integration: full processor with defaults
# ---------------------------------------------------------------------------


class TestDefaultProcessorIntegration:
    """Integration tests running the processor with default instructions."""

    def test_approaching_turn_limit_fires_at_turn_16_of_20(self) -> None:
        processor = JitInstructionProcessor()
        ctx = JitContext(turn=16, max_turns=20, tool_usage={"read_file": 2}, work_items_complete=0, work_items_total=0)
        results = processor.evaluate(ctx)
        assert any("approaching the turn limit" in r for r in results)

    def test_once_prevents_duplicate_across_evaluations(self) -> None:
        processor = JitInstructionProcessor()
        ctx = JitContext(turn=16, max_turns=20, tool_usage={"read_file": 2}, work_items_complete=0, work_items_total=0)

        results1 = processor.evaluate(ctx)
        approaching = [r for r in results1 if "approaching the turn limit" in r]
        assert len(approaching) == 1

        results2 = processor.evaluate(ctx)
        approaching2 = [r for r in results2 if "approaching the turn limit" in r]
        assert len(approaching2) == 0

    def test_no_instructions_fire_on_turn_1(self) -> None:
        processor = JitInstructionProcessor()
        ctx = JitContext(turn=1, max_turns=50, tool_usage={}, work_items_complete=0, work_items_total=0)
        assert processor.evaluate(ctx) == []

    def test_custom_instructions_override_defaults(self) -> None:
        custom = JitInstruction(
            id="custom",
            instruction="Custom!",
            condition=lambda ctx: ctx.turn >= 1,
        )
        processor = JitInstructionProcessor(instructions=[custom])
        assert len(processor.instructions) == 1

        ctx = JitContext(turn=1, max_turns=20, tool_usage={}, work_items_complete=0, work_items_total=0)
        results = processor.evaluate(ctx)
        assert results == ["Custom!"]
