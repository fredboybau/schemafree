from collections import defaultdict
from collections.abc import Sequence

import numpy as np


def slide_grouped_split(
    slide_ids: Sequence[str],
    labels: Sequence[int],
    fractions: tuple[float, float, float] = (0.7, 0.1, 0.2),
    seed: int = 0,
) -> tuple[list[int], list[int], list[int]]:
    if abs(sum(fractions) - 1.0) > 1e-6:
        raise ValueError("fractions must sum to 1")
    rng = np.random.default_rng(seed)
    slide_label: dict[str, int] = {}
    slide_members: dict[str, list[int]] = defaultdict(list)
    for index, (slide, label) in enumerate(zip(slide_ids, labels)):
        slide_label.setdefault(slide, label)
        slide_members[slide].append(index)

    by_class: dict[int, list[str]] = defaultdict(list)
    for slide, label in slide_label.items():
        by_class[label].append(slide)

    train: list[int] = []
    val: list[int] = []
    test: list[int] = []
    for slides in by_class.values():
        order = list(slides)
        rng.shuffle(order)
        n = len(order)
        n_train = int(round(fractions[0] * n))
        n_val = int(round(fractions[1] * n))
        n_train = min(n_train, n)
        n_val = min(n_val, n - n_train)
        for slide in order[:n_train]:
            train.extend(slide_members[slide])
        for slide in order[n_train : n_train + n_val]:
            val.extend(slide_members[slide])
        for slide in order[n_train + n_val :]:
            test.extend(slide_members[slide])
    return sorted(train), sorted(val), sorted(test)
