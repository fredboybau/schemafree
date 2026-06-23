import os

import plac

from schemafree.datum.bearings import (
    DataSpec,
    DistillSpec,
    EncoderSpec,
    FederationSpec,
    OptimSpec,
    PrivacySpec,
    RunSpec,
)
from schemafree.datum.monument import Monument, stash
from schemafree.datum.origin import set_seed, spawn_generator
from schemafree.datum.register import bind_plan, get_logger
from schemafree.sheets._common import build_sources, cohort_sizes
from schemafree.triangulation.budget import compose_epsilon
from schemafree.triangulation.traverse import federated_pretrain


@plac.annotations(
    plan=("path to a gin survey plan", "positional", None, str),
    data_root=("directory holding one image folder per cohort", "option", "d", str),
    synthetic_batches=("batches per cohort when no data root is present", "option", "b", int),
)
def main(plan: str, data_root: str = "", synthetic_batches: int = 4) -> None:
    bind_plan(plan)
    logger = get_logger()
    run = RunSpec()
    set_seed(run.seed)
    generator = spawn_generator(run.seed)

    encoder = EncoderSpec()
    data = DataSpec()
    fed = FederationSpec()
    distill = DistillSpec()
    optim = OptimSpec()
    privacy = PrivacySpec()

    if privacy.enabled:
        epsilon = compose_epsilon(
            privacy.noise_multiplier, privacy.sample_rate, fed.rounds, privacy.delta
        )
        logger.info("composed privacy budget epsilon=%.3f delta=%.1e", epsilon, privacy.delta)

    sources = build_sources(data, encoder, data_root, synthetic_batches)
    sizes = cohort_sizes(data)
    logger.info("federating over cohorts %s for %d rounds", list(sources), fed.rounds)

    model = federated_pretrain(encoder, sources, sizes, fed, distill, optim, privacy, generator)

    os.makedirs(run.out_dir, exist_ok=True)
    target = os.path.join(run.out_dir, "encoder.ckpt")
    stash(
        target,
        Monument(
            round_index=fed.rounds,
            seed=run.seed,
            state={k: v.detach().cpu() for k, v in model.encoder.state_dict().items()},
            extra={"held_out": data.held_out, "beta": fed.beta},
        ),
    )
    logger.info("stashed encoder at %s", target)


if __name__ == "__main__":
    plac.call(main)
