import plac

from schemafree.control.calibration import expected_calibration_error
from schemafree.control.marks import macro_f1, sensitivity, specificity
from schemafree.control.probe import extract_features, fit_probe, predict
from schemafree.control.significance import delong_variance
from schemafree.datum.bearings import DataSpec, EncoderSpec, ProbeSpec, RunSpec
from schemafree.datum.monument import restore
from schemafree.datum.origin import set_seed
from schemafree.datum.register import bind_plan, get_logger
from schemafree.instruments.theodolite import build_encoder
from schemafree.sheets._common import labeled_synthetic


@plac.annotations(
    plan=("path to a gin survey plan", "positional", None, str),
    checkpoint=("encoder checkpoint produced by pretrain", "option", "c", str),
    samples=("labeled samples drawn for the held-out cohort", "option", "n", int),
)
def main(plan: str, checkpoint: str = "", samples: int = 128) -> None:
    bind_plan(plan)
    logger = get_logger()
    run = RunSpec()
    set_seed(run.seed)

    encoder_spec = EncoderSpec()
    data = DataSpec()
    encoder = build_encoder(encoder_spec)
    if checkpoint:
        encoder.load_state_dict(restore(checkpoint).state)

    images, labels = labeled_synthetic(encoder_spec, samples, run.seed)
    features = extract_features(encoder, images)
    split = samples // 2
    probe = fit_probe(features[:split], labels[:split], 2, ProbeSpec())
    pred, prob = predict(probe, features[split:])
    truth = labels[split:].tolist()
    positive_score = prob[:, 1].tolist()

    area, variance = delong_variance(truth, positive_score)
    logger.info(
        "%s macro-F1=%.3f sens=%.3f spec=%.3f auc=%.3f(var=%.4f) ece=%.3f",
        data.held_out,
        macro_f1(truth, pred.tolist()),
        sensitivity(truth, pred.tolist()),
        specificity(truth, pred.tolist()),
        area,
        variance,
        expected_calibration_error(positive_score, truth),
    )


if __name__ == "__main__":
    plac.call(main)
