"""Tests for Phase 5: STDP (Spike-Timing-Dependent Plasticity)."""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest


# --- Unit Tests: STDP Delta Computation ---


class TestStdpDeltaComputation:
    """Verify STDP directional weight updates are computed correctly."""

    @pytest.mark.asyncio
    async def test_forward_direction(self, enable_flag):
        """When canonical source was accessed first, forward_weight strengthens."""
        enable_flag("engram_flag_stdp")
        from engram.services.reconsolidation import _apply_stdp

        node_a = uuid4()  # canonical source (min)
        node_b = uuid4()  # canonical target (max)
        # Ensure a < b for canonical ordering
        if node_a > node_b:
            node_a, node_b = node_b, node_a

        edge_id = uuid4()
        now = datetime.now(timezone.utc)

        mock_conn = AsyncMock()
        # fetch_edge_ids_for_pairs returns edge rows
        mock_conn.fetch = AsyncMock(return_value=[
            {"id": edge_id, "source_id": node_a, "target_id": node_b,
             "forward_weight": 0.5, "backward_weight": 0.5},
        ])

        pre_timestamps = {
            node_a: now - timedelta(hours=1),  # accessed first
            node_b: now,
        }

        stats: dict = {}
        with patch("engram.services.reconsolidation.edge_repo") as mock_repo:
            mock_repo.fetch_edge_ids_for_pairs = AsyncMock(return_value=[
                {"id": edge_id, "source_id": node_a, "target_id": node_b,
                 "forward_weight": 0.5, "backward_weight": 0.5},
            ])
            mock_repo.apply_stdp_update = AsyncMock(return_value=1)

            await _apply_stdp(
                mock_conn, uuid4(),
                [node_a], [node_b], ["semantic"],
                pre_timestamps, stats,
            )

            # Verify apply_stdp_update was called
            mock_repo.apply_stdp_update.assert_called_once()
            call_args = mock_repo.apply_stdp_update.call_args
            forward_deltas = call_args[0][3]
            backward_deltas = call_args[0][4]

            # Forward should strengthen, backward should weaken
            assert forward_deltas[0] == pytest.approx(0.03)
            assert backward_deltas[0] == pytest.approx(-0.01)

    @pytest.mark.asyncio
    async def test_backward_direction(self, enable_flag):
        """When canonical target was accessed first, backward_weight strengthens."""
        enable_flag("engram_flag_stdp")
        from engram.services.reconsolidation import _apply_stdp

        node_a = uuid4()
        node_b = uuid4()
        if node_a > node_b:
            node_a, node_b = node_b, node_a

        edge_id = uuid4()
        now = datetime.now(timezone.utc)

        pre_timestamps = {
            node_a: now,
            node_b: now - timedelta(hours=1),  # target accessed first
        }

        stats: dict = {}
        with patch("engram.services.reconsolidation.edge_repo") as mock_repo:
            mock_repo.fetch_edge_ids_for_pairs = AsyncMock(return_value=[
                {"id": edge_id, "source_id": node_a, "target_id": node_b,
                 "forward_weight": 0.5, "backward_weight": 0.5},
            ])
            mock_repo.apply_stdp_update = AsyncMock(return_value=1)

            await _apply_stdp(
                AsyncMock(), uuid4(),
                [node_a], [node_b], ["semantic"],
                pre_timestamps, stats,
            )

            call_args = mock_repo.apply_stdp_update.call_args
            forward_deltas = call_args[0][3]
            backward_deltas = call_args[0][4]

            # Backward should strengthen, forward should weaken
            assert forward_deltas[0] == pytest.approx(-0.01)
            assert backward_deltas[0] == pytest.approx(0.03)

    @pytest.mark.asyncio
    async def test_simultaneous_skipped(self, enable_flag):
        """When both nodes have the same timestamp, no STDP update occurs."""
        enable_flag("engram_flag_stdp")
        from engram.services.reconsolidation import _apply_stdp

        node_a = uuid4()
        node_b = uuid4()
        if node_a > node_b:
            node_a, node_b = node_b, node_a

        edge_id = uuid4()
        now = datetime.now(timezone.utc)

        pre_timestamps = {
            node_a: now,
            node_b: now,  # same timestamp
        }

        stats: dict = {}
        with patch("engram.services.reconsolidation.edge_repo") as mock_repo:
            mock_repo.fetch_edge_ids_for_pairs = AsyncMock(return_value=[
                {"id": edge_id, "source_id": node_a, "target_id": node_b,
                 "forward_weight": 0.5, "backward_weight": 0.5},
            ])
            mock_repo.apply_stdp_update = AsyncMock(return_value=0)

            await _apply_stdp(
                AsyncMock(), uuid4(),
                [node_a], [node_b], ["semantic"],
                pre_timestamps, stats,
            )

            # apply_stdp_update should NOT be called (empty edge_ids list)
            mock_repo.apply_stdp_update.assert_not_called()

    @pytest.mark.asyncio
    async def test_none_timestamp_skipped(self, enable_flag):
        """When a node has no timestamp (first recall), skip STDP."""
        enable_flag("engram_flag_stdp")
        from engram.services.reconsolidation import _apply_stdp

        node_a = uuid4()
        node_b = uuid4()
        if node_a > node_b:
            node_a, node_b = node_b, node_a

        edge_id = uuid4()

        pre_timestamps = {
            node_a: datetime.now(timezone.utc),
            node_b: None,  # no timestamp
        }

        stats: dict = {}
        with patch("engram.services.reconsolidation.edge_repo") as mock_repo:
            mock_repo.fetch_edge_ids_for_pairs = AsyncMock(return_value=[
                {"id": edge_id, "source_id": node_a, "target_id": node_b,
                 "forward_weight": 0.5, "backward_weight": 0.5},
            ])
            mock_repo.apply_stdp_update = AsyncMock(return_value=0)

            await _apply_stdp(
                AsyncMock(), uuid4(),
                [node_a], [node_b], ["semantic"],
                pre_timestamps, stats,
            )

            # apply_stdp_update should NOT be called (skipped due to None timestamp)
            mock_repo.apply_stdp_update.assert_not_called()


# --- Unit Tests: Plasticity Scaling ---


class TestStdpPlasticityScaling:
    """When metaplasticity is on, STDP deltas are scaled by plasticity."""

    @pytest.mark.asyncio
    async def test_low_plasticity_reduces_delta(self, enable_flag):
        """Established edges (low plasticity) resist directional shifts."""
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
            patch("engram.services.reconsolidation.edge_repo") as mock_edge_repo,
            patch("engram.services.reconsolidation.memory_repo") as mock_mem_repo,
        ):
            mock_edge_repo.fetch_edge_ids_for_pairs = AsyncMock(return_value=[
                {"id": edge_id, "source_id": node_a, "target_id": node_b,
                 "forward_weight": 0.5, "backward_weight": 0.5},
            ])
            mock_edge_repo.apply_stdp_update = AsyncMock(return_value=1)

            # Low plasticity = 0.2 (established)
            mock_mem_repo.fetch_activation_and_content = AsyncMock(return_value=[
                {"id": node_a, "plasticity": 0.2},
                {"id": node_b, "plasticity": 0.2},
            ])

            await _apply_stdp(
                AsyncMock(), uuid4(),
                [node_a], [node_b], ["semantic"],
                pre_timestamps, stats,
            )

            call_args = mock_edge_repo.apply_stdp_update.call_args
            forward_deltas = call_args[0][3]
            backward_deltas = call_args[0][4]

            # Deltas scaled by min(0.2, 0.2) = 0.2
            assert forward_deltas[0] == pytest.approx(0.03 * 0.2)
            assert backward_deltas[0] == pytest.approx(-0.01 * 0.2)


# --- Unit Tests: Chain Following ---


class TestSequenceChainFollowing:
    """Verify _follow_sequence_chain logic."""

    @pytest.mark.asyncio
    async def test_follows_strongest_forward(self, enable_flag):
        """Chain follows strongest effective forward edge."""
        enable_flag("engram_flag_stdp")
        from engram.services.read import _follow_sequence_chain

        seed = uuid4()
        next_node = uuid4()
        third_node = uuid4()

        mock_conn = AsyncMock()
        # First call: seed → next_node (strong), seed → other (weak)
        # Second call: next_node → third_node
        # Third call: no more edges
        mock_conn.fetch = AsyncMock(side_effect=[
            [
                {"source_id": seed, "target_id": next_node,
                 "edge_type": "excitatory", "axis": "semantic",
                 "weight": 0.7, "forward_weight": 0.8, "backward_weight": 0.3,
                 "effective_forward": 0.8},
            ],
            [
                {"source_id": next_node, "target_id": third_node,
                 "edge_type": "excitatory", "axis": "semantic",
                 "weight": 0.6, "forward_weight": 0.6, "backward_weight": 0.3,
                 "effective_forward": 0.6},
            ],
            [],
        ])

        with patch("engram.services.read.edge_repo") as mock_repo:
            mock_repo.fetch_strongest_forward_edges = AsyncMock(side_effect=[
                [{"source_id": seed, "target_id": next_node,
                  "forward_weight": 0.8, "backward_weight": 0.3,
                  "effective_forward": 0.8}],
                [{"source_id": next_node, "target_id": third_node,
                  "forward_weight": 0.6, "backward_weight": 0.3,
                  "effective_forward": 0.6}],
                [],
            ])

            chain, confidence = await _follow_sequence_chain(
                mock_conn, uuid4(), seed, top_k=5
            )

        assert chain == [seed, next_node, third_node]
        assert confidence == pytest.approx(0.6)  # min forward weight

    @pytest.mark.asyncio
    async def test_stops_at_loop(self, enable_flag):
        """Chain stops when it would revisit a node."""
        enable_flag("engram_flag_stdp")
        from engram.services.read import _follow_sequence_chain

        seed = uuid4()
        next_node = uuid4()

        with patch("engram.services.read.edge_repo") as mock_repo:
            # After next_node, only edge leads back to seed (loop)
            mock_repo.fetch_strongest_forward_edges = AsyncMock(side_effect=[
                [{"source_id": seed, "target_id": next_node,
                  "forward_weight": 0.8, "backward_weight": 0.3,
                  "effective_forward": 0.8}],
                [{"source_id": next_node, "target_id": seed,
                  "forward_weight": 0.7, "backward_weight": 0.3,
                  "effective_forward": 0.7}],
            ])

            chain, confidence = await _follow_sequence_chain(
                AsyncMock(), uuid4(), seed, top_k=5
            )

        assert chain == [seed, next_node]
        assert confidence == pytest.approx(0.8)

    @pytest.mark.asyncio
    async def test_single_node_zero_confidence(self, enable_flag):
        """If no chain can be formed, confidence is 0.0."""
        enable_flag("engram_flag_stdp")
        from engram.services.read import _follow_sequence_chain

        seed = uuid4()

        with patch("engram.services.read.edge_repo") as mock_repo:
            mock_repo.fetch_strongest_forward_edges = AsyncMock(return_value=[])

            chain, confidence = await _follow_sequence_chain(
                AsyncMock(), uuid4(), seed, top_k=5
            )

        assert chain == [seed]
        assert confidence == 0.0


# --- Unit Tests: Directional Propagation ---


class TestDirectionalPropagation:
    """Verify _apply_edges uses directional weights when STDP is on."""

    def test_forward_weight_used_for_source(self, enable_flag):
        """When traversing from source, forward_weight is used."""
        enable_flag("engram_flag_stdp")
        from engram.services.read import _apply_edges

        src = uuid4()
        tgt = uuid4()

        row = MagicMock()
        row.__getitem__ = lambda self, key: {
            "source_id": src, "target_id": tgt,
            "edge_type": "excitatory", "weight": 0.5,
            "forward_weight": 0.9, "backward_weight": 0.1,
        }[key]
        row.get = lambda key, default=None: {
            "forward_weight": 0.9, "backward_weight": 0.1,
        }.get(key, default)

        activation_map = {src: 1.0}
        _apply_edges([row], {src}, activation_map)

        # Neighbor (tgt) should have been activated using forward_weight (0.9)
        # delta = 0.9 * 1.0 * 0.5 = 0.45
        assert tgt in activation_map
        assert activation_map[tgt] == pytest.approx(0.45)

    def test_backward_weight_used_for_target(self, enable_flag):
        """When traversing from target, backward_weight is used."""
        enable_flag("engram_flag_stdp")
        from engram.services.read import _apply_edges

        src = uuid4()
        tgt = uuid4()

        row = MagicMock()
        row.__getitem__ = lambda self, key: {
            "source_id": src, "target_id": tgt,
            "edge_type": "excitatory", "weight": 0.5,
            "forward_weight": 0.9, "backward_weight": 0.1,
        }[key]
        row.get = lambda key, default=None: {
            "forward_weight": 0.9, "backward_weight": 0.1,
        }.get(key, default)

        activation_map = {tgt: 1.0}
        _apply_edges([row], {tgt}, activation_map)

        # Neighbor (src) should have been activated using backward_weight (0.1)
        # delta = 0.1 * 1.0 * 0.5 = 0.05
        assert src in activation_map
        assert activation_map[src] == pytest.approx(0.05)

    def test_symmetric_weight_when_flag_off(self, disable_flag):
        """When STDP is off, regular weight is used."""
        disable_flag("engram_flag_stdp")
        from engram.services.read import _apply_edges

        src = uuid4()
        tgt = uuid4()

        row = MagicMock()
        row.__getitem__ = lambda self, key: {
            "source_id": src, "target_id": tgt,
            "edge_type": "excitatory", "weight": 0.5,
            "forward_weight": 0.9, "backward_weight": 0.1,
        }[key]
        row.get = lambda key, default=None: {
            "forward_weight": 0.9, "backward_weight": 0.1,
        }.get(key, default)

        activation_map = {src: 1.0}
        _apply_edges([row], {src}, activation_map)

        # Should use weight=0.5, not forward_weight=0.9
        # delta = 0.5 * 1.0 * 0.5 = 0.25
        assert activation_map[tgt] == pytest.approx(0.25)


# --- Unit Tests: Flag Off ---


class TestStdpFlagOff:
    """When STDP flag is off, all STDP behavior is skipped."""

    @pytest.mark.asyncio
    async def test_reconsolidate_skips_stdp(self, disable_flag):
        """No timestamp capture or STDP updates when flag is off."""
        disable_flag("engram_flag_stdp")
        disable_flag("engram_flag_forgetting")
        from engram.services.reconsolidation import reconsolidate

        node_a = uuid4()
        node_b = uuid4()

        mock_pool = AsyncMock()
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(return_value="UPDATE 1")
        mock_conn.fetch = AsyncMock(return_value=[])

        with (
            patch("engram.services.reconsolidation.tenant_connection") as mock_tc,
            patch("engram.services.reconsolidation.memory_repo") as mock_mem,
            patch("engram.services.reconsolidation.edge_repo") as mock_edge,
        ):
            mock_tc.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
            mock_tc.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_mem.boost_nodes = AsyncMock(return_value=2)
            mock_edge.strengthen_co_retrieval = AsyncMock(return_value=1)

            stats = await reconsolidate(
                mock_pool, uuid4(),
                [
                    {"id": node_a, "matched_axes": ["semantic"]},
                    {"id": node_b, "matched_axes": ["semantic"]},
                ],
            )

        # No STDP key in stats
        assert "stdp_pairs_updated" not in stats
        # fetch_edge_ids_for_pairs should NOT be called
        mock_edge.fetch_edge_ids_for_pairs.assert_not_called()


# --- Unit Tests: Edge Repo ---


class TestEdgeRepoStdpColumns:
    """Verify SQL in edge_repo includes forward/backward weight columns."""

    @pytest.mark.asyncio
    async def test_upsert_includes_directional_weights(self):
        """upsert_edge SQL should include forward_weight and backward_weight."""
        from engram.repositories.edge_repo import upsert_edge

        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(return_value="INSERT 1")

        await upsert_edge(
            mock_conn,
            owner_id=uuid4(),
            source_id=uuid4(),
            target_id=uuid4(),
            edge_type="excitatory",
            axis="semantic",
            weight=0.7,
        )

        sql = mock_conn.execute.call_args[0][0]
        assert "forward_weight" in sql
        assert "backward_weight" in sql

    @pytest.mark.asyncio
    async def test_fetch_edges_includes_directional_weights(self):
        """fetch_edges_for_nodes SQL should SELECT forward/backward weights."""
        from engram.repositories.edge_repo import fetch_edges_for_nodes

        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=[])

        await fetch_edges_for_nodes(mock_conn, uuid4(), [uuid4()], ["semantic"])

        sql = mock_conn.fetch.call_args[0][0]
        assert "forward_weight" in sql
        assert "backward_weight" in sql

    @pytest.mark.asyncio
    async def test_strengthen_updates_directional_weights(self):
        """strengthen_co_retrieval should also update forward/backward."""
        from engram.repositories.edge_repo import strengthen_co_retrieval

        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(return_value="UPDATE 1")

        await strengthen_co_retrieval(
            mock_conn, uuid4(), [uuid4()], [uuid4()], ["semantic"]
        )

        sql = mock_conn.execute.call_args[0][0]
        assert "forward_weight" in sql
        assert "backward_weight" in sql

    @pytest.mark.asyncio
    async def test_decay_includes_directional_weights(self):
        """decay_weights should also decay forward/backward weights."""
        from engram.repositories.edge_repo import decay_weights

        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(return_value="UPDATE 1")

        await decay_weights(mock_conn, uuid4())

        sql = mock_conn.execute.call_args[0][0]
        assert "forward_weight" in sql
        assert "backward_weight" in sql

    @pytest.mark.asyncio
    async def test_prune_uses_greatest_when_stdp_on(self, enable_flag):
        """prune_weak should use GREATEST(forward, backward) when STDP is on."""
        enable_flag("engram_flag_stdp")
        from engram.repositories.edge_repo import prune_weak

        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(return_value="DELETE 0")

        await prune_weak(mock_conn, uuid4())

        sql = mock_conn.execute.call_args[0][0]
        assert "GREATEST" in sql

    @pytest.mark.asyncio
    async def test_prune_uses_weight_when_stdp_off(self, disable_flag):
        """prune_weak should use weight column when STDP is off."""
        disable_flag("engram_flag_stdp")
        from engram.repositories.edge_repo import prune_weak

        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(return_value="DELETE 0")

        await prune_weak(mock_conn, uuid4())

        sql = mock_conn.execute.call_args[0][0]
        assert "GREATEST" not in sql

    @pytest.mark.asyncio
    async def test_apply_stdp_update_batch(self):
        """apply_stdp_update should batch UPDATE via unnest."""
        from engram.repositories.edge_repo import apply_stdp_update

        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(return_value="UPDATE 2")

        edge_ids = [uuid4(), uuid4()]
        result = await apply_stdp_update(
            mock_conn, uuid4(),
            edge_ids, [0.03, -0.01], [-0.01, 0.03],
        )

        assert result == 2
        sql = mock_conn.execute.call_args[0][0]
        assert "unnest" in sql

    @pytest.mark.asyncio
    async def test_fetch_transfer_includes_directional(self):
        """fetch_edges_for_transfer should include directional weights."""
        from engram.repositories.edge_repo import fetch_edges_for_transfer

        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=[])

        await fetch_edges_for_transfer(mock_conn, uuid4(), uuid4())

        sql = mock_conn.fetch.call_args[0][0]
        assert "forward_weight" in sql
        assert "backward_weight" in sql
