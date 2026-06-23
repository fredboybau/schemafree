import plac
import torch

from schemafree.datum.bearings import EncoderSpec
from schemafree.datum.monument import restore
from schemafree.datum.register import bind_plan, get_logger
from schemafree.instruments.theodolite import build_encoder


@plac.annotations(
    plan=("path to a gin survey plan", "positional", None, str),
    checkpoint=("encoder checkpoint to export", "option", "c", str),
    out=("destination onnx path", "option", "o", str),
)
def main(plan: str, checkpoint: str = "", out: str = "runs/encoder.onnx") -> None:
    bind_plan(plan)
    logger = get_logger()
    encoder_spec = EncoderSpec()
    encoder = build_encoder(encoder_spec)
    if checkpoint:
        encoder.load_state_dict(restore(checkpoint).state)
    encoder.eval()

    dummy = torch.zeros(1, 3, encoder_spec.image_size, encoder_spec.image_size)
    torch.onnx.export(
        encoder,
        dummy,
        out,
        input_names=["image"],
        output_names=["embedding"],
        dynamic_axes={"image": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version=17,
    )
    logger.info("exported encoder to %s", out)


if __name__ == "__main__":
    plac.call(main)
