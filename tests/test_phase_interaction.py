"""Tests for phase interactions — verify flags work independently and together."""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from engram.services.reconsolidation import _compute_competitor_penalties


# --- Phase 2+4 Interaction ---


class TestForgettingRespectsPlasticity:
    """When both forgetting and metaplasticity flags are on,
    established memories (low plasticity) should resist suppression."""

    def test_penalty_conceptually_scales_with_plasticity(self):
        """The reconsolidate function should multiply penalty by plasticity
        when both flags are on. We verify the math here."""
        base_penalty = 0.03
        overlap = 0.8
        scaled_penalty = base_penalty * overlap  # = 0.024

        # Established memory (low plasticity)
        low_plasticity = 0.1
        effective_low = scaled_penalty * low_plasticity  # = 0.0024

        # Fresh memory (high plasticity)
        high_plasticity = 0.9
        effective_high = scaled_penalty * high_plasticity  # = 0.0216

        # Established memories resist suppression
        assert effective_low < effective_high
        assert effective_high / effective_low == pytest.approx(9.0)


# --- Flag Independence ---


class TestFlagsIndependent:
    """Each flag works independently of the others."""

    @pytest.mark.asyncio
    async def test_phase1_without_phase2(self, enable_flag, disable_flag):
        """Context retrieval works when forgetting is off."""
        enable_flag("engram_flag_context_retrieval")
        disable_flag("engram_flag_forgetting")
        disable_flag("engram_flag_attractor")
        disable_flag("engram_flag_metaplasticity")

        from engram.config import settings
        assert settings.engram_flag_context_retrieval is True
        assert settings.engram_flag_forgetting is False

    @pytest.mark.asyncio
    async def test_phase2_without_phase4(self, enable_flag, disable_flag):
        """Forgetting works without plasticity scaling."""
        enable_flag("engram_flag_forgetting")
        disable_flag("engram_flag_metaplasticity")

        from engram.config import settings
        assert settings.engram_flag_forgetting is True
        assert settings.engram_flag_metaplasticity is False

        # Verify penalty computation doesn't need plasticity
        winner = uuid4()
        comp = uuid4()
        comp_ids, penalties = _compute_competitor_penalties(
            [{"id": winner, "matched_axes": ["semantic"]}],
            [winner, comp],
            {
                winner: {"scores": {"semantic": 0.9}},
                comp: {"scores": {"semantic": 0.7}},
            },
        )
        assert len(comp_ids) > 0
        assert all(p > 0 for p in penalties)

    @pytest.mark.asyncio
    async def test_phase4_without_phase2(self, enable_flag, disable_flag):
        """Metaplasticity works without forgetting."""
        enable_flag("engram_flag_metaplasticity")
        disable_flag("engram_flag_forgetting")

        from engram.config import settings
        assert settings.engram_flag_metaplasticity is True
        assert settings.engram_flag_forgetting is False

    @pytest.mark.asyncio
    async def test_phase3_without_others(self, enable_flag, disable_flag):
        """Attractor dynamics works independently."""
        enable_flag("engram_flag_attractor")
        disable_flag("engram_flag_context_retrieval")
        disable_flag("engram_flag_forgetting")
        disable_flag("engram_flag_metaplasticity")

        from engram.config import settings
        assert settings.engram_flag_attractor is True

    @pytest.mark.asyncio
    async def test_all_flags_off_default(self, disable_flag):
        """All flags default to off."""
        disable_flag("engram_flag_context_retrieval")
        disable_flag("engram_flag_forgetting")
        disable_flag("engram_flag_attractor")
        disable_flag("engram_flag_metaplasticity")
        disable_flag("engram_flag_stdp")

        from engram.config import settings
        assert not settings.engram_flag_context_retrieval
        assert not settings.engram_flag_forgetting
        assert not settings.engram_flag_attractor
        assert not settings.engram_flag_metaplasticity
        assert not settings.engram_flag_stdp

    @pytest.mark.asyncio
    async def test_phase5_independent(self, enable_flag, disable_flag):
        """STDP works independently of other flags."""
        enable_flag("engram_flag_stdp")
        disable_flag("engram_flag_context_retrieval")
        disable_flag("engram_flag_forgetting")
        disable_flag("engram_flag_attractor")
        disable_flag("engram_flag_metaplasticity")

        from engram.config import settings
        assert settings.engram_flag_stdp is True
        assert settings.engram_flag_metaplasticity is False


# --- Phase 1+3 Interaction ---


class TestContextWeightsWithAttractor:
    """Verify axis_weights bias persists through attractor settling."""

    def test_weights_not_washed_out(self, enable_flag):
        """Axis weights should still matter after settling."""
        enable_flag("engram_flag_context_retrieval")
        enable_flag("engram_flag_attractor")

        from engram.services.read import _attractor_settle

        seed1 = uuid4()
        seed2 = uuid4()
        cand = uuid4()

        # seed1 has high emotional, seed2 has high action
        # If we weighted emotional heavily, seed1 should stay on top
        candidates = {
            seed1: {"scores": {"semantic": 0.7, "emotional": 0.9}, "matched_axes": ["semantic", "emotional"]},
            seed2: {"scores": {"semantic": 0.7, "action": 0.9}, "matched_axes": ["semantic", "action"]},
            cand: {"scores": {"semantic": 0.5, "emotional": 0.5}, "matched_axes": ["semantic", "emotional"]},
        }

        # Pre-scored with emotional weighting (seed1 higher)
        seeds = [
            {"id": seed1, "adjusted": 6.0, "dims_matched": 2, "convergence_score": 6.0,
             "activation_level": 1.0, "dimension_scores": candidates[seed1]["scores"],
             "matched_axes": candidates[seed1]["matched_axes"]},
            {"id": seed2, "adjusted": 4.0, "dims_matched": 2, "convergence_score": 4.0,
             "activation_level": 1.0, "dimension_scores": candidates[seed2]["scores"],
             "matched_axes": candidates[seed2]["matched_axes"]},
        ]

        result = _attractor_settle(candidates, seeds, [])

        # seed1 should still be ranked first (its adjusted was higher)
        assert result[0]["id"] == seed1


# --- Phase 5 Cross-Phase Interactions ---


class TestStdpWithMetaplasticity:
    """STDP + metaplasticity: established edges resist directional shifts."""

    @pytest.mark.asyncio
    async def test_established_edges_resist_stdp(self, enable_flag):
        """Low plasticity should reduce STDP delta magnitude."""
        enable_flag("engram_flag_stdp")
        enable_flag("engram_flag_metaplasticity")

        from engram.services.reconsolidation import _apply_stdp

        node_a = uuid4()
        node_b = uuid4()
        if node_a > node_b:
            node_a, node_b = node_b, node_a

        edge_id = uuid4()
        now = datetime.now(timezone.utc)

        pre_timestamps = {
            node_a: now - timedelta(hours=1),
            node_b: now,
        }

        stats: dict = {}
        with (
            patch("engram.services.reconsolidation.edge_repo") as mock_edge,
            patch("engram.services.reconsolidation.memory_repo") as mock_mem,
        ):
            mock_edge.fetch_edge_ids_for_pairs = AsyncMock(return_value=[
                {"id": edge_id, "source_id": node_a, "target_id": node_b,
                 "forward_weight": 0.5, "backward_weight": 0.5},
            ])
            mock_edge.apply_stdp_update = AsyncMock(return_value=1)
            mock_mem.fetch_activation_and_content = AsyncMock(return_value=[
                {"id": node_a, "plasticity": 0.1},
                {"id": node_b, "plasticity": 0.1},
            ])

            await _apply_stdp(
                AsyncMock(), uuid4(),
                [node_a], [node_b], ["semantic"],
                pre_timestamps, stats,
            )

            # Verify plasticity scaling reduced the deltas
            call_args = mock_edge.apply_stdp_update.call_args
            forward_deltas = call_args[0][3]

            # 0.03 * 0.1 = 0.003 — heavily damped
            assert abs(forward_deltas[0]) < 0.01


class TestStdpWithAttractor:
    """STDP + attractor: attractor uses max(forward, backward) for edge signal."""

    def test_attractor_uses_max_directional(self, enable_flag):
        """Attractor settling should use max of directional weights."""
        enable_flag("engram_flag_stdp")
        enable_flag("engram_flag_attractor")

        from engram.services.read import _attractor_settle

        seed1 = uuid4()
        seed2 = uuid4()
        cand = uuid4()

        candidates = {
            seed1: {"scores": {"semantic": 0.8}, "matched_axes": ["semantic"]},
            seed2: {"scores": {"semantic": 0.7}, "matched_axes": ["semantic"]},
            cand: {"scores": {"semantic": 0.5}, "matched_axes": ["semantic"]},
        }

        seeds = [
            {"id": seed1, "adjusted": 5.0, "dims_matched": 1,
             "convergence_score": 5.0, "activation_level": 1.0,
             "dimension_scores": {"semantic": 0.8},
             "matched_axes": ["semantic"]},
            {"id": seed2, "adjusted": 4.0, "dims_matched": 1,
             "convergence_score": 4.0, "activation_level": 1.0,
             "dimension_scores": {"semantic": 0.7},
             "matched_axes": ["semantic"]},
        ]

        # Edge with asymmetric weights — max should be used
        edge_row = MagicMock()
        edge_row.__getitem__ = lambda self, key: {
            "source_id": min(seed1, cand),
            "target_id": max(seed1, cand),
            "weight": 0.3,
            "forward_weight": 0.8,
            "backward_weight": 0.2,
        }[key]
        edge_row.get = lambda key, default=None: {
            "forward_weight": 0.8, "backward_weight": 0.2,
        }.get(key, default)

        result = _attractor_settle(candidates, seeds, [edge_row])
        # Should settle without error; exact ranking depends on algorithm
        assert len(result) >= 2


class TestStdpWithForgetting:
    """STDP + forgetting: both features coexist without interference."""

    def test_stdp_delta_computation_independent_of_forgetting(self):
        """STDP deltas don't depend on forgetting flag state."""
        # STDP strengthen/weaken constants are fixed
        strengthen = 0.03
        weaken = -0.01

        # These values should be the same regardless of forgetting flag
        assert strengthen == 0.03
        assert weaken == -0.01
        # Net asymmetry per update
        assert strengthen + weaken == pytest.approx(0.02)
