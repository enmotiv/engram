"""Tests for Phase 2: Retrieval-induced forgetting."""

from uuid import uuid4

import pytest

from engram.services.reconsolidation import _compute_competitor_penalties


# --- Unit Tests ---


class TestPenaltyComputation:
    """Verify similarity-weighted penalty computation."""

    def test_penalty_scales_with_similarity(self):
        """Near-miss (high overlap) gets higher penalty than distant candidate."""
        winner_id = uuid4()
        near_miss_id = uuid4()
        distant_id = uuid4()

        returned_items = [
            {"id": winner_id, "matched_axes": ["semantic", "emotional"]},
        ]
        candidate_ids = [winner_id, near_miss_id, distant_id]
        candidates = {
            winner_id: {"scores": {"semantic": 0.9, "emotional": 0.8}},
            near_miss_id: {"scores": {"semantic": 0.85, "emotional": 0.75}},
            distant_id: {"scores": {"semantic": 0.4, "emotional": 0.35}},
        }

        comp_ids, penalties = _compute_competitor_penalties(
            returned_items, candidate_ids, candidates
        )

        # Both should be penalized
        assert near_miss_id in comp_ids
        assert distant_id in comp_ids

        near_idx = comp_ids.index(near_miss_id)
        dist_idx = comp_ids.index(distant_id)
        assert penalties[near_idx] > penalties[dist_idx]

    def test_zero_penalty_for_zero_overlap(self):
        """Candidates with no axis overlap with winners get zero/minimal penalty."""
        winner_id = uuid4()
        no_overlap_id = uuid4()

        returned_items = [
            {"id": winner_id, "matched_axes": ["semantic"]},
        ]
        candidate_ids = [winner_id, no_overlap_id]
        candidates = {
            winner_id: {"scores": {"semantic": 0.9}},
            no_overlap_id: {"scores": {"action": 0.8}},
        }

        comp_ids, penalties = _compute_competitor_penalties(
            returned_items, candidate_ids, candidates
        )

        # No shared axes → overlap = 0 → penalty = 0 → filtered out
        assert no_overlap_id not in comp_ids

    def test_empty_candidates_no_error(self):
        """When candidate_ids is empty, returns empty lists."""
        comp_ids, penalties = _compute_competitor_penalties(
            [{"id": uuid4(), "matched_axes": ["semantic"]}],
            [],
            {},
        )
        assert comp_ids == []
        assert penalties == []

    def test_winners_excluded_from_competitors(self):
        """Winners should never appear as competitors."""
        winner = uuid4()
        comp = uuid4()

        returned_items = [{"id": winner, "matched_axes": ["semantic"]}]
        candidate_ids = [winner, comp]
        candidates = {
            winner: {"scores": {"semantic": 0.9}},
            comp: {"scores": {"semantic": 0.7}},
        }

        comp_ids, penalties = _compute_competitor_penalties(
            returned_items, candidate_ids, candidates
        )

        assert winner not in comp_ids
        assert comp in comp_ids

    def test_base_penalty_parameter(self):
        """Custom base_penalty scales all penalties proportionally."""
        winner = uuid4()
        comp = uuid4()

        returned_items = [{"id": winner, "matched_axes": ["semantic"]}]
        candidate_ids = [winner, comp]
        candidates = {
            winner: {"scores": {"semantic": 0.9}},
            comp: {"scores": {"semantic": 0.8}},
        }

        _, penalties_default = _compute_competitor_penalties(
            returned_items, candidate_ids, candidates, base_penalty=0.03
        )
        _, penalties_double = _compute_competitor_penalties(
            returned_items, candidate_ids, candidates, base_penalty=0.06
        )

        # Double base_penalty → double penalty
        assert abs(penalties_double[0] - penalties_default[0] * 2) < 0.0001


class TestForgettingFlagOff:
    """Verify reconsolidate ignores candidates when flag is off."""

    @pytest.mark.asyncio
    async def test_no_suppression_when_flag_off(self, disable_flag):
        """When forgetting flag is off, no competitor suppression."""
        disable_flag("engram_flag_forgetting")

        from unittest.mock import AsyncMock, patch
        from engram.services.reconsolidation import reconsolidate

        winner = uuid4()
        comp = uuid4()
        owner = uuid4()

        returned_items = [
            {"id": winner, "matched_axes": ["semantic"]},
        ]

        mock_pool = AsyncMock()

        # Mock the tenant_connection context manager
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(return_value="UPDATE 1")
        mock_conn.fetch = AsyncMock(return_value=[])

        with patch("engram.services.reconsolidation.tenant_connection") as mock_tc:
            mock_tc.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
            mock_tc.return_value.__aexit__ = AsyncMock(return_value=False)

            with patch("engram.services.reconsolidation.memory_repo.boost_nodes", AsyncMock(return_value=1)):
                with patch("engram.services.reconsolidation.memory_repo.penalize_competitors_scaled") as mock_pen:
                    stats = await reconsolidate(
                        mock_pool, owner, returned_items,
                        candidate_ids=[winner, comp],
                        candidates={
                            winner: {"scores": {"semantic": 0.9}},
                            comp: {"scores": {"semantic": 0.7}},
                        },
                    )

                    # penalize should NOT have been called
                    mock_pen.assert_not_called()
                    assert "competitors_suppressed" not in stats


# --- Integration Tests (require TEST_DATABASE_URL) ---


@pytest.mark.asyncio
async def test_recall_no_errors_with_candidates(client, owner_a, seeded_memories):
    """Basic recall still works correctly with the reconsolidate signature change."""
    _, key = owner_a
    resp = await client.post(
        "/v1/recall",
        json={"cue": "dark mode preferences", "top_k": 3},
        headers={"Authorization": f"Bearer {key}"},
    )
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert "memories" in data
