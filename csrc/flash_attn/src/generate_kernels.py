# Copied from Driss Guessous's PR in PyTorch: https://github.com/pytorch/pytorch/pull/105602

# This file is run to generate the kernel instantiations for the flash_attn kernels
# They are written to several files in order to speed up compilation

import argparse
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional

HEAD_DIMENSIONS = [32, 64, 96, 128, 160, 192, 256]
IS_CAUSAL = ["false", "true"]
KERNEL_IMPL_TEMPLATE_FWD = """#include "flash_fwd_launch_template.h"

template<>
void run_mha_fwd_<{DTYPE}, {HEAD_DIM}, {IS_CAUSAL}>(Flash_fwd_params &params, cudaStream_t stream) {{
    run_mha_fwd_hdim{HEAD_DIM}<{DTYPE}, {IS_CAUSAL}>(params, stream);
}}
"""

KERNEL_IMPL_TEMPLATE_FWD_SPLIT = """#include "flash_fwd_launch_template.h"

template void run_mha_fwd_splitkv_dispatch<{DTYPE}, {HEAD_DIM}, {IS_CAUSAL}>(Flash_fwd_params &params, cudaStream_t stream);
"""

KERNEL_IMPL_TEMPLATE_BWD = """#include "flash_bwd_launch_template.h"

template<>
void run_mha_bwd_<{DTYPE}, {HEAD_DIM}, {IS_CAUSAL}>(Flash_bwd_params &params, cudaStream_t stream) {{
    run_mha_bwd_hdim{HEAD_DIM}<{DTYPE}, {IS_CAUSAL}>(params, stream);
}}
"""


@dataclass
class Kernel:
    sm: int
    dtype: str
    head_dim: int
    is_causal: bool
    direction: str

    def template(self, DTYPE_MAP) -> str:
        if self.direction == "fwd":
            return KERNEL_IMPL_TEMPLATE_FWD.format(
                DTYPE=DTYPE_MAP[self.dtype], HEAD_DIM=self.head_dim, IS_CAUSAL=self.is_causal
            )
        elif self.direction == "bwd":
            return KERNEL_IMPL_TEMPLATE_BWD.format(
                DTYPE=DTYPE_MAP[self.dtype], HEAD_DIM=self.head_dim, IS_CAUSAL=self.is_causal
            )
        else:
            return KERNEL_IMPL_TEMPLATE_FWD_SPLIT.format(
                DTYPE=DTYPE_MAP[self.dtype], HEAD_DIM=self.head_dim, IS_CAUSAL=self.is_causal
            )

    @property
    def filename(self) -> str:
        return f"flash_{self.direction}_hdim{self.head_dim}_{self.dtype}_{'causal_' if self.is_causal == 'true' else ''}sm{self.sm}.cu"


def get_all_kernels(DTYPE_MAP, SM) -> List[Kernel]:
    for direction in ["fwd", "fwd_split", "bwd"]:
        for dtype, head_dim, is_causal, sm in itertools.product(DTYPE_MAP.keys(), HEAD_DIMENSIONS, IS_CAUSAL, SM):
            yield Kernel(sm=sm, dtype=dtype, head_dim=head_dim, is_causal=is_causal, direction=direction)


def write_kernel(kernel: Kernel, autogen_dir: Path, DTYPE_MAP) -> None:
    prelude = """// Copyright (c) 2024, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"\n
"""
    (autogen_dir / kernel.filename).write_text(prelude + kernel.template(DTYPE_MAP))


def main(sm: Optional[Literal[75, 80]], output_dir: Optional[str]) -> None:
    if output_dir is None:
        output_dir = Path(__file__).parent
    else:
        output_dir = Path(output_dir)

    SM_DTYPE_MAP = {
        '75': {
            "fp16": "cutlass::half_t",
        },
        '80': {
            "fp16": "cutlass::half_t",
            "bf16": "cutlass::bfloat16_t",
        }
    }

    if sm is None:
        for sm in [75, 80]:
            SM = [sm]
            DTYPE_MAP = SM_DTYPE_MAP[str(sm)]

            for kernel in get_all_kernels(DTYPE_MAP=DTYPE_MAP, SM=SM):
                write_kernel(kernel, output_dir, DTYPE_MAP)
    else:
        SM = [sm]
        DTYPE_MAP = SM_DTYPE_MAP[str(sm)]

        for kernel in get_all_kernels(DTYPE_MAP=DTYPE_MAP, SM=SM):
            write_kernel(kernel, output_dir, DTYPE_MAP)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="generate_kernels",
        description="Generate the flash_attention kernels template instantiations",
    )
    # Set an optional output directory
    parser.add_argument(
        "-o",
        "--output_dir",
        required=False,
        help="Where to generate the kernels "
        " will default to the current directory ",
    )
    parser.add_argument(
        "--sm",
        required=False,
        help="SM to generate kernels for "
        " will default to 75 and 80"
    )
    args = parser.parse_args()
    main(args.sm, args.output_dir)
