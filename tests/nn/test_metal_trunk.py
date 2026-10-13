"""The fused Metal trunk against the PyTorch trunk it stands in for.

Both run bf16 autocast, so they cannot agree bit-for-bit: each is compared
with an fp32 PyTorch run, and the fused error must be the same size as the
PyTorch bf16 error.

Batches are at least 32 sequences on purpose: below a few hundred rows MPS
autograd is not itself reproducible on this model (identical runs differed
by up to 100% at 6 sequences and agreed to 3e-7 at 32), so a smaller test
would measure that rather than the kernels.
"""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from cardgym.games.plump.env import PlumpEnv
from cardgym.games.plump.model import PlumpActor, PlumpModelConfig
from cardgym.games.plump.rounds import RoundSpec, round_game_config
from cardgym.games.plump.tokens import Schema, seat_tokens
from cardgym.nn.metal_trunk import DualTrunkPass, MetalTrunk, trunk_parameters, unsupported_reason

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="the Metal trunk needs MPS")

DEVICE = torch.device("mps")
MAX_LENGTH = Schema().oracle_max_seq_len


def config(**overrides) -> PlumpModelConfig:
    # The rl-v6 trunk at two layers: GQA 10/2, PLE 64, alternating value embeddings.
    fields = dict(d_model=240, n_layers=2, n_heads=10, n_kv_heads=2, d_ff=720, per_layer_dim=64)
    fields.update(overrides)
    return PlumpModelConfig(**fields)


def token_batch(players: int, hand: int, seeds) -> torch.Tensor:
    rows = []
    for seed in seeds:
        start = random.Random(seed).randrange(players)
        env = PlumpEnv(round_game_config(RoundSpec(players, hand), bidding_start_player=start), seed=seed)
        env.reset(seed=seed)
        rng = random.Random(seed + 1)
        while not env.is_done():
            env.step(rng.choice(env.legal_actions()))
        observer = seed % players
        rows.append(
            seat_tokens(
                Schema(), env.state.event_log, observer, players, hand,
                env.state.current_round.initial_hands[observer], start,
            )
        )
    return torch.from_numpy(np.stack(rows)).to(DEVICE)


def make_model(model_config: PlumpModelConfig, seed: int = 0) -> PlumpActor:
    torch.manual_seed(seed)
    model = PlumpActor(model_config)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.02 * torch.randn_like(parameter))
    return model.to(DEVICE).train()


def run(model, tokens, probe, *, precision, metal):
    model.trunk.metal = MetalTrunk(model.trunk_config, max_length=MAX_LENGTH) if metal else None
    model.zero_grad(set_to_none=True)
    if precision == "bf16":
        with torch.autocast("mps", dtype=torch.bfloat16):
            hidden = model.forward_hidden(tokens)
    else:
        hidden = model.forward_hidden(tokens)
    (hidden.float() * probe).sum().backward()
    grads = {n: p.grad.detach().float().clone() for n, p in model.named_parameters() if p.grad is not None}
    model.trunk.metal = None
    return hidden.detach().float(), grads


def relative(a, b) -> float:
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        dict(d_model=96, n_heads=4, n_kv_heads=2, d_ff=192),
        dict(d_model=192, n_heads=8, n_kv_heads=1, d_ff=384, value_embeddings="all"),
    ],
)
def test_forward_and_backward_match_the_pytorch_trunk(overrides):
    model_config = config(**overrides)
    assert unsupported_reason(model_config.trunk(), MAX_LENGTH) is None
    model = make_model(model_config)
    tokens = token_batch(4, 7, range(32))
    probe = torch.randn(*tokens.shape[:2], model_config.d_model, device=DEVICE)
    exact, exact_grads = run(model, tokens, probe, precision="fp32", metal=False)
    torch_bf16, torch_grads = run(model, tokens, probe, precision="bf16", metal=False)
    fused, fused_grads = run(model, tokens, probe, precision="bf16", metal=True)
    assert relative(fused, exact) < 2 * relative(torch_bf16, exact) + 1e-3
    assert set(fused_grads) == set(torch_grads)
    ratios = []
    for name, grad in exact_grads.items():
        if grad.norm() == 0:
            continue
        fused_error = relative(fused_grads[name], grad)
        assert fused_error < 0.05, name
        ratios.append(fused_error / max(relative(torch_grads[name], grad), 1e-4))
    assert np.median(ratios) < 1.5
    assert max(ratios) < 4


def test_dual_pass_returns_each_losss_own_gradients():
    model = make_model(config(), seed=1)
    tokens = token_batch(5, 6, range(10, 42))
    params = [p for p in model.parameters() if p.requires_grad]

    def losses(hidden):
        auxiliary = model.suit_head(hidden).float().square().mean()
        policy = model.bid_head(hidden).float().logsumexp(-1).mean()
        return auxiliary, policy

    with torch.autocast("mps", dtype=torch.bfloat16):
        expected = [
            torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
            for loss in losses(model.forward_hidden(tokens))
        ]
    model.enable_metal_trunk(MAX_LENGTH)
    with torch.autocast("mps", dtype=torch.bfloat16):
        dual = DualTrunkPass(model, tokens)
        got = dual.gradients(list(losses(dual.hidden)), params)
    trunk = {id(p) for p in trunk_parameters(model.trunk)}
    for stream in range(2):
        for p, want, have in zip(params, expected[stream], got[stream]):
            if want is None:
                assert have is None
                continue
            assert have is not None and have.shape == p.shape
            assert relative(have.float(), want.float()) < (0.06 if id(p) in trunk else 0.03)


def test_trunk_only_replaces_bf16_autocast_on_mps():
    model = make_model(config())
    model.enable_metal_trunk(MAX_LENGTH)
    trunk = model.trunk
    assert not trunk.uses_metal(DEVICE, 81)
    with torch.autocast("mps", dtype=torch.float16):
        assert not trunk.uses_metal(DEVICE, 81)
    with torch.autocast("mps", dtype=torch.bfloat16):
        assert trunk.uses_metal(DEVICE, 81)
        assert not trunk.uses_metal(torch.device("cpu"), 81)
        assert trunk.uses_metal(DEVICE, 500)  # attention tiles keys: no length limit


def test_unsupported_configurations_are_named():
    assert unsupported_reason(config().trunk(), MAX_LENGTH) is None
    assert unsupported_reason(config(per_layer_dim=32).trunk(), MAX_LENGTH) is not None
    assert unsupported_reason(config(n_heads=2, n_kv_heads=2).trunk(), MAX_LENGTH) is not None  # head_dim 120
    assert unsupported_reason(config().trunk(), 500) is None


def test_merged_decode_matches_the_pytorch_cached_decode():
    """Ragged waves: rows at different positions, padded runs."""

    model = make_model(config(), seed=2).eval()
    tokens = token_batch(4, 7, range(40, 72))
    batch, length, width = tokens.shape
    rng = random.Random(0)
    caches = {name: model.new_cache(batch, Schema().max_seq_len + 1, torch.float16) for name in ("torch", "metal")}
    cursor = [0] * batch
    waves = 0
    with torch.inference_mode(), torch.autocast("mps", dtype=torch.bfloat16):
        hidden = model.forward_prefill(tokens[:, :9], caches["torch"], None)
        model.forward_prefill(tokens[:, :9], caches["metal"], None)
        cursor = [9] * batch
        while min(cursor) < length:
            counts = [min(rng.randint(1, 3), length - c) for c in cursor]
            run = max(counts)
            wave = torch.zeros(batch, run, width, dtype=tokens.dtype, device=DEVICE)
            for row in range(batch):
                if counts[row]:
                    block = tokens[row, cursor[row] : cursor[row] + counts[row]]
                    wave[row, : block.shape[0]] = block
                    wave[row, block.shape[0] :] = block[-1]
            positions = torch.tensor(cursor, device=DEVICE)
            offsets = torch.tensor([max(c, 1) - 1 for c in counts], device=DEVICE)
            hidden = {}
            for name, cache in caches.items():
                model.trunk.metal = MetalTrunk(model.trunk_config, max_length=MAX_LENGTH) if name == "metal" else None
                hidden[name] = model.forward_ragged(wave, positions, cache, offsets).float()
            live = torch.tensor([c > 0 for c in counts], device=DEVICE)
            assert relative(hidden["metal"][live], hidden["torch"][live]) < 1e-2
            cursor = [c + n for c, n in zip(cursor, counts)]
            waves += 1
    assert waves > 5
    for layer in range(model.trunk_config.n_layers):
        for store in ("k", "v"):
            got = getattr(caches["metal"], store)[layer][:, :, :length].float()
            want = getattr(caches["torch"], store)[layer][:, :, :length].float()
            assert relative(got, want) < 1e-2


def test_cached_bf16_weights_follow_a_fused_optimizer_step():
    """Fused MPS Adam does not bump Tensor._version; the weight cache must still refresh."""

    model = make_model(config())
    model.enable_metal_trunk(MAX_LENGTH)
    params = trunk_parameters(model.trunk)
    before = model.trunk.metal._bf16_weights(params)[0][0].clone()
    optimizer = torch.optim.Adam(params, lr=1e-2, fused=True)
    for p in params:
        p.grad = torch.ones_like(p)
    optimizer.step()
    after = model.trunk.metal._bf16_weights(params)[0][0]
    assert torch.equal(after, params[2].detach().to(torch.bfloat16))
    assert not torch.equal(after, before)
