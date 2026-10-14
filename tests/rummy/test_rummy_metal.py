"""The fused Metal trunk on Rummy's shapes: head_dim 32, streams longer than one attention tile."""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from cardgym.games.rummy.game import RummyGame, RummyGameConfig
from cardgym.games.rummy.model import RummyModelConfig
from cardgym.games.rummy.tokens import FIELD_VOCAB, LEDGER_VOCAB
from cardgym.games.rummy.training import RummyLossConfig
from cardgym.nn.metal_trunk import DualTrunkPass, MetalTrunk, trunk_parameters, unsupported_reason
from cardgym.nn.packing import Packing
from cardgym.rl.continuous import ContinuousCollector
from cardgym.rl.ppo import masked_probabilities
from tests.conftest import randomize

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="the Metal trunk needs MPS")

DEVICE = torch.device("mps")


def rummy_game(**model) -> RummyGame:
    fields = dict(d_model=256, n_layers=2, n_heads=8, n_kv_heads=2, d_ff=512, per_layer_dim=64,
                  ledger_width=16, ledger_hidden=32, ledger_layers=1, score_dim=32)
    fields.update(model)
    settings = RummyGameConfig(player_counts=(2, 3, 4), hand_sizes=(13,), games_per_shape=4, max_turns=200)
    return RummyGame(RummyModelConfig(**fields), settings, RummyLossConfig())


def random_tokens(batch: int, length: int, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    columns = [torch.randint(0, vocab, (batch, length, 1), generator=generator) for vocab in FIELD_VOCAB]
    columns.append(torch.randint(0, LEDGER_VOCAB, (batch, length, 52), generator=generator))
    return torch.cat(columns, dim=-1).to(DEVICE)


def relative(a, b) -> float:
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def run(model, tokens, probe, *, bf16: bool, metal: bool):
    model.trunk.metal = MetalTrunk(model.trunk_config) if metal else None
    model.zero_grad(set_to_none=True)
    with torch.autocast("mps", dtype=torch.bfloat16, enabled=bf16):
        hidden = model.forward_hidden(tokens)
    model.pop_context() if hasattr(model, "pop_context") else None
    (hidden.float() * probe).sum().backward()
    grads = {n: p.grad.detach().float().clone() for n, p in model.named_parameters() if p.grad is not None}
    model.trunk.metal = None
    return hidden.detach().float(), grads


@pytest.mark.parametrize("length", [40, 200])
def test_long_streams_match_the_pytorch_trunk(length):
    game = rummy_game()
    assert unsupported_reason(game.model_config.trunk()) is None
    torch.manual_seed(0)
    model = randomize(game.build_critic(), seed=1, scale=0.02).to(DEVICE).train()
    tokens = random_tokens(32, length)
    probe = torch.randn(32, length, 256, device=DEVICE)
    exact, exact_grads = run(model, tokens, probe, bf16=False, metal=False)
    torch_bf16, torch_grads = run(model, tokens, probe, bf16=True, metal=False)
    fused, fused_grads = run(model, tokens, probe, bf16=True, metal=True)
    assert relative(fused, exact) < 2 * relative(torch_bf16, exact) + 1e-3
    ratios = []
    for name, grad in exact_grads.items():
        if grad.norm() == 0:
            continue
        error = relative(fused_grads[name], grad)
        assert error < 0.05, name
        ratios.append(error / max(relative(torch_grads[name], grad), 1e-4))
    assert np.median(ratios) < 1.5
    assert max(ratios) < 4


def test_dual_pass_with_policy_reading_card_features():
    """The policy loss reaches the grid both through the trunk and through the kept per-card features."""

    game = rummy_game()
    torch.manual_seed(0)
    model = randomize(game.build_actor(), seed=2, scale=0.02).to(DEVICE).train()
    tokens = random_tokens(32, 150, seed=3)
    params = [p for p in model.parameters() if p.requires_grad]
    kind = torch.randint(0, 7, (32 * 150, 6), device=DEVICE)
    cards = torch.randint(0, 53, (32 * 150, 6, 13), device=DEVICE)
    target = torch.randint(0, 53, (32 * 150, 6, 13), device=DEVICE)

    def losses(hidden, context):
        flat = hidden.reshape(-1, hidden.shape[-1])
        auxiliary = model.winner_head(flat).float().square().mean()
        features = {"kind": kind, "cards": cards, "target": target}
        logits = model.policy_logits(flat, "play", features, None, context.reshape(-1, *context.shape[2:]))
        return auxiliary, logits.float().logsumexp(-1).mean()

    with torch.autocast("mps", dtype=torch.bfloat16):
        hidden = model.forward_hidden(tokens)
        expected = [
            torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
            for loss in losses(hidden, model.pop_context())
        ]
    model.enable_metal_trunk(10_000)
    with torch.autocast("mps", dtype=torch.bfloat16):
        dual = DualTrunkPass(model, tokens)
        got = dual.gradients(list(losses(dual.hidden, dual.context)), params)
    model.trunk.metal = None
    trunk = {id(p) for p in trunk_parameters(model.trunk)}
    for stream in range(2):
        for p, want, have in zip(params, expected[stream], got[stream]):
            if want is None:
                assert have is None
                continue
            assert have is not None and have.shape == p.shape
            assert relative(have.float(), want.float()) < (0.06 if id(p) in trunk else 0.03)


def test_fused_gathered_decode_matches_full_replay():
    """Continuous rollout through the fused decode (gathered cache rows) against a full fp32 replay."""

    game = rummy_game()
    actor = randomize(game.build_actor().eval(), seed=3, scale=0.05).to(DEVICE)
    actor.enable_metal_trunk(game.max_length)
    collector = ContinuousCollector(
        actor, device=DEVICE, precision="bf16", kv_dtype="fp16", max_length=game.max_length,
    )
    trajectories = collector.collect(game.lanes(1.0), random.Random(0), version=0)
    actor.trunk.metal = None
    worst = []
    with torch.no_grad():
        for trajectory in trajectories[:6]:
            for seat, stream in enumerate(trajectory.streams):
                if not len(stream):
                    continue
                tokens = torch.from_numpy(stream.astype(np.int64))[None].to(DEVICE)
                hidden = actor.forward_hidden(tokens)[0]
                actor.pop_context()
                for decision in trajectory.decisions:
                    if decision.seat != seat:
                        continue
                    features = {k: torch.from_numpy(v)[None].to(DEVICE) for k, v in decision.features.items()}
                    logits = actor.policy_logits(
                        hidden[decision.position][None], decision.head, features, tokens[0, decision.position][None],
                    ).cpu().numpy()
                    probs = masked_probabilities(logits, np.ones_like(logits, dtype=bool))[0]
                    worst.append(float(np.abs(probs - decision.probs).max()))
    assert len(worst) > 100
    assert np.mean(worst) < 5e-3 and max(worst) < 0.05


@pytest.mark.parametrize("layers", [1, 2])
def test_fused_card_grid_matches_pytorch(layers):
    game = rummy_game(ledger_width=32, ledger_hidden=64, ledger_layers=layers)
    torch.manual_seed(0)
    grid = randomize(game.build_critic(), seed=4, scale=0.05).embedding.grid.to(DEVICE).train()
    ledger = random_tokens(64, 40)[..., -52:]
    probe = torch.randn(64, 40, 52, 32, device=DEVICE)

    def run(bf16: bool, fused: bool):
        grid.fused = fused
        grid.zero_grad(set_to_none=True)
        with torch.autocast("mps", dtype=torch.bfloat16, enabled=bf16):
            cards, summary = grid(ledger)
        ((cards.float() * probe).sum() + summary.float().square().sum()).backward()
        grid.fused = False
        return cards.detach().float(), {n: p.grad.float().clone() for n, p in grid.named_parameters()}

    exact, exact_grads = run(False, False)
    torch_bf16, torch_grads = run(True, False)
    fused, fused_grads = run(True, True)
    assert relative(fused, exact) < 2 * relative(torch_bf16, exact) + 1e-3
    assert set(fused_grads) == set(exact_grads)
    for name, grad in exact_grads.items():
        error = relative(fused_grads[name], grad)
        assert error < max(0.03, 2 * relative(torch_grads[name], grad)), name


def test_packed_rows_match_the_padded_batch():
    """Packed rows (real tokens only, with sequence bounds) give the padded batch's results."""

    game = rummy_game()
    model = randomize(game.build_critic(), seed=5, scale=0.02).to(DEVICE).train()
    model.enable_metal_trunk(game.max_length)
    tokens = random_tokens(48, 70, seed=6)
    lengths = np.random.default_rng(0).integers(1, 71, size=48)
    lengths[0] = 70
    packing = Packing(lengths, DEVICE)
    real = torch.from_numpy(np.arange(70)[None, :] < lengths[:, None]).to(DEVICE)
    probe = torch.randn(48, 70, 256, device=DEVICE) * real[..., None]
    params = [p for p in model.parameters() if p.requires_grad]
    with torch.autocast("mps", dtype=torch.bfloat16):
        padded = model.forward_hidden(tokens)
    padded_grads = torch.autograd.grad((padded.float() * probe).sum(), params, allow_unused=True)
    packed_tokens = torch.from_numpy(packing.pack(tokens.cpu().numpy())).to(DEVICE)
    with torch.autocast("mps", dtype=torch.bfloat16):
        packed = model.forward_hidden(packed_tokens, packing.bounds)
    assert packed.shape == (int(lengths.sum()), 256)
    packed_grads = torch.autograd.grad((packed.float() * probe[real]).sum(), params, allow_unused=True)
    assert relative(packed.detach().float(), padded.detach().float()[real]) < 1e-3
    for want, have in zip(padded_grads, packed_grads):
        if want is not None:
            assert relative(have.float(), want.float()) < 1e-2


def test_dual_pass_on_packed_rows():
    game = rummy_game()
    model = randomize(game.build_actor(), seed=7, scale=0.02).to(DEVICE).train()
    model.enable_metal_trunk(game.max_length)
    tokens = random_tokens(40, 60, seed=8)
    lengths = np.random.default_rng(1).integers(1, 61, size=40)
    packing = Packing(lengths, DEVICE)
    real = np.arange(60)[None, :] < lengths[:, None]
    params = [p for p in model.parameters() if p.requires_grad]
    rows = int(real.sum())
    kind = torch.randint(0, 7, (rows, 6), device=DEVICE)
    cards = torch.randint(0, 53, (rows, 6, 13), device=DEVICE)
    target = torch.randint(0, 53, (rows, 6, 13), device=DEVICE)
    flat_real = torch.from_numpy(np.flatnonzero(real)).to(DEVICE)

    def losses(hidden, context, select):
        flat = hidden.reshape(-1, hidden.shape[-1])[select]
        grid = context.reshape(-1, *context.shape[-2:])[select]
        auxiliary = model.winner_head(flat).float().square().sum(-1).mean()
        features = {"kind": kind, "cards": cards, "target": target}
        logits = model.policy_logits(flat, "play", features, None, grid)
        return auxiliary, logits.float().logsumexp(-1).mean()

    results = []
    with torch.autocast("mps", dtype=torch.bfloat16):
        dual = DualTrunkPass(model, tokens)
        results.append(dual.gradients(list(losses(dual.hidden, dual.context, flat_real)), params))
        packed_tokens = torch.from_numpy(packing.pack(tokens.cpu().numpy())).to(DEVICE)
        dual = DualTrunkPass(model, packed_tokens, packing.bounds)
        assert dual.hidden.shape == (rows, 256)
        everything = torch.arange(rows, device=DEVICE)
        results.append(dual.gradients(list(losses(dual.hidden, dual.context, everything)), params))
    for stream in range(2):
        for want, have in zip(results[0][stream], results[1][stream]):
            if want is None:
                assert have is None
                continue
            assert relative(have.float(), want.float()) < 1e-2


def test_fused_field_lookups_match_embedding_bag():
    from cardgym.nn.metal_bag import FieldBags

    generator = torch.Generator().manual_seed(9)
    tables = [torch.randn(809, width, generator=generator).to(DEVICE).requires_grad_() for width in (256, 768, 384)]
    ids = torch.stack([torch.randint(start, start + 30, (3000,), generator=generator) for start in range(0, 720, 30)], 1)
    ids = ids.to(DEVICE)
    probes = [torch.randn(3000, t.shape[1], device=DEVICE) for t in tables]
    fused = FieldBags.apply(ids, *tables)
    reference = [F.embedding_bag(ids, t, mode="sum") for t in tables]
    for got, want in zip(fused, reference):
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)
    fused_grads = torch.autograd.grad(sum((o * p).sum() for o, p in zip(fused, probes)), tables)
    want_grads = torch.autograd.grad(sum((o * p).sum() for o, p in zip(reference, probes)), tables)
    for got, want in zip(fused_grads, want_grads):
        assert relative(got, want) < 1e-2  # the output gradient is rounded to bf16
