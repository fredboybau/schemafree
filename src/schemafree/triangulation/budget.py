import math
from collections.abc import Sequence

_DEFAULT_ORDERS: tuple[float, ...] = tuple(
    [1.0 + x / 10.0 for x in range(1, 100)] + [float(x) for x in range(11, 64)]
)


def _log_add(a: float, b: float) -> float:
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    if a > b:
        return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))


def _log_binom(n: float, k: int) -> float:
    return math.lgamma(n + 1.0) - math.lgamma(k + 1.0) - math.lgamma(n - k + 1.0)


def _log_a_int(alpha: int, q: float, sigma: float) -> float:
    log_a = -math.inf
    for i in range(alpha + 1):
        term = (
            _log_binom(float(alpha), i)
            + i * math.log(q)
            + (alpha - i) * math.log1p(-q)
            + (i * i - i) / (2.0 * sigma * sigma)
        )
        log_a = _log_add(log_a, term)
    return log_a


def _rdp_at_order(q: float, sigma: float, order: float) -> float:
    if q == 0.0:
        return 0.0
    if sigma == 0.0:
        return math.inf
    if float(order).is_integer():
        log_a = _log_a_int(int(order), q, sigma)
    else:
        lower = _log_a_int(int(math.floor(order)), q, sigma)
        upper = _log_a_int(int(math.ceil(order)), q, sigma)
        frac = order - math.floor(order)
        log_a = (1.0 - frac) * lower + frac * upper
    return log_a / (order - 1.0)


def rdp_subsampled_gaussian(
    q: float,
    noise_multiplier: float,
    steps: int,
    orders: Sequence[float] = _DEFAULT_ORDERS,
) -> list[float]:
    return [steps * _rdp_at_order(q, noise_multiplier, o) for o in orders]


def rdp_to_epsilon(
    rdp: Sequence[float],
    delta: float,
    orders: Sequence[float] = _DEFAULT_ORDERS,
) -> tuple[float, float]:
    best_eps = math.inf
    best_order = orders[0]
    for value, order in zip(rdp, orders):
        if order <= 1.0:
            continue
        eps = value + math.log1p(-1.0 / order) - math.log(delta * order) / (order - 1.0)
        if eps < best_eps:
            best_eps = eps
            best_order = order
    return best_eps, best_order


def compose_epsilon(
    noise_multiplier: float,
    sample_rate: float,
    steps: int,
    delta: float,
) -> float:
    rdp = rdp_subsampled_gaussian(sample_rate, noise_multiplier, steps)
    eps, _ = rdp_to_epsilon(rdp, delta)
    return eps
