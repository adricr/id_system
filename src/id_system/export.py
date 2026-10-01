import json

import pandas as pd
import torch
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge_transform_and_lower
from executorch.runtime import Runtime
from id_system_large import (
METADATA_PATH,
PROCESSED_DIR,
build_model,
)

PTE_PATH = PROCESSED_DIR / "model_large.pte"
LABELS_PATH = PROCESSED_DIR / "labels.json"
# Must match the CenterCrop size in val_transform
INPUT_SIZE = 224

# The react-native-executorch classifier does all pre/post-processing itself, so the
# model is exported untouched:
#   - input:  float32 [1, 3, 224, 224]; the app resizes/crops and normalizes with
#             alpha = 1 / (255 * std), beta = -mean / std (equivalent to v2.Normalize)
#   - output: raw logits float32 [1, N]; the library applies softmax and the label lookup
# Adding Normalize or softmax here would apply them twice.


def main():
    device = torch.device("cpu")
    metadata = pd.read_csv(METADATA_PATH, index_col="label_index").sort_index()
    num_classes = metadata.shape[0]
    model, trained_epochs = build_model(num_classes, device)
    model.eval()

    example_input = (torch.rand(1, 3, INPUT_SIZE, INPUT_SIZE),)
    exported_program = torch.export.export(model, example_input)
    executorch_program = to_edge_transform_and_lower(
        exported_program,
        partitioner=[XnnpackPartitioner()],
    ).to_executorch()

    with open(PTE_PATH, "wb") as f:
        f.write(executorch_program.buffer)
    print(f"Saved {PTE_PATH} ({PTE_PATH.stat().st_size / 1e6:.1f} MB)")

    # The library requires labels[i] to be the name of output i
    labels = [
        {"species": row["species"], "common_name": row["common_name"]}
        for _, row in metadata.iterrows()
    ]
    with open(LABELS_PATH, "w") as f:
        json.dump(labels, f, indent=2)
    print(f"Saved {LABELS_PATH} ({len(labels)} labels)")

    verify(model, example_input)


def verify(model, example_input):
    """Run the .pte through the ExecuTorch runtime and compare against eager PyTorch."""
    program = Runtime.get().load_program(PTE_PATH)
    method = program.load_method("forward")
    pte_logits = method.execute(list(example_input))[0]

    with torch.no_grad():
        eager_logits = model(*example_input)

    max_diff = (pte_logits - eager_logits).abs().max().item()
    same_top1 = pte_logits.argmax(1).item() == eager_logits.argmax(1).item()
    print(f"Verification: max logit diff {max_diff:.2e}, same top-1: {same_top1}")


if __name__ == "__main__":
    main()
