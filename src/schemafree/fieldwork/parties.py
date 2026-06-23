import dataclasses
from collections.abc import Mapping

NORMAL = 0
ABNORMAL = 1


@dataclasses.dataclass(frozen=True)
class Cohort:
    key: str
    country: str
    modality: str
    native_classes: tuple[str, ...]
    normal_classes: tuple[str, ...]

    def mapping(self) -> Mapping[str, int]:
        return {
            name: (NORMAL if name in self.normal_classes else ABNORMAL)
            for name in self.native_classes
        }


_HERLEV = Cohort(
    key="herlev",
    country="Denmark",
    modality="conventional_smear",
    native_classes=(
        "superficial_squamous",
        "intermediate_squamous",
        "columnar_epithelial",
        "mild_dysplasia",
        "moderate_dysplasia",
        "severe_dysplasia",
        "carcinoma_in_situ",
    ),
    normal_classes=(
        "superficial_squamous",
        "intermediate_squamous",
        "columnar_epithelial",
    ),
)

_SIPAKMED = Cohort(
    key="sipakmed",
    country="Greece",
    modality="conventional_smear",
    native_classes=(
        "superficial_intermediate",
        "parabasal",
        "metaplastic",
        "koilocytotic",
        "dyskeratotic",
    ),
    normal_classes=("superficial_intermediate", "parabasal", "metaplastic"),
)

_MENDELEY = Cohort(
    key="mendeley_lbc",
    country="India",
    modality="liquid_based",
    native_classes=("nilm", "lsil", "hsil", "scc"),
    normal_classes=("nilm",),
)

_CRIC = Cohort(
    key="cric",
    country="Brazil",
    modality="conventional_smear",
    native_classes=("nilm", "ascus", "lsil", "asch", "hsil", "scc"),
    normal_classes=("nilm",),
)

COHORTS: dict[str, Cohort] = {c.key: c for c in (_HERLEV, _SIPAKMED, _MENDELEY, _CRIC)}


def harmonize(cohort_key: str, native_class: str) -> int:
    cohort = COHORTS[cohort_key]
    mapping = cohort.mapping()
    if native_class not in mapping:
        raise KeyError(f"{native_class} is not native to {cohort_key}")
    return mapping[native_class]


def binary_label(cohort_key: str, native_index: int) -> int:
    cohort = COHORTS[cohort_key]
    return harmonize(cohort_key, cohort.native_classes[native_index])
