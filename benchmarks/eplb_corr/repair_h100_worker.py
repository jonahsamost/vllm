# SPDX-License-Identifier: Apache-2.0
"""Bridge source-only files to the H100 image's installed vLLM package."""

from __future__ import annotations

import subprocess
from pathlib import Path


def main() -> None:
    source = Path("/opt/jonah_vllm")
    system_package = Path("/usr/local/lib/python3.12/dist-packages/vllm")
    third_party = source / "vllm/third_party"
    subprocess.run(
        ["cp", "-an", f"{system_package / 'third_party'}/.", str(third_party)],
        check=True,
    )
    cute = source / "vllm/vllm_flash_attn/cute"
    if not cute.exists():
        subprocess.run(
            ["cp", "-a", str(system_package / "vllm_flash_attn/cute"), str(cute)],
            check=True,
        )

    # This image's MoE binary predates the optional scatter_idx argument. The
    # route-capture model passes None, for which the seven-argument op is exact.
    ops = source / "vllm/_custom_ops.py"
    code = ops.read_text()
    old = """    torch.ops._moe_C.moe_align_block_size(
        topk_ids,
        num_experts,
        block_size,
        sorted_token_ids,
        experts_ids,
        num_tokens_post_pad,
        expert_map,
        scatter_idx,
    )
"""
    new = """    args = (
        topk_ids,
        num_experts,
        block_size,
        sorted_token_ids,
        experts_ids,
        num_tokens_post_pad,
        expert_map,
    )
    if scatter_idx is None:
        torch.ops._moe_C.moe_align_block_size(*args)
    else:
        torch.ops._moe_C.moe_align_block_size(*args, scatter_idx)
"""
    if old in code:
        ops.write_text(code.replace(old, new, 1))
    elif new not in code:
        raise RuntimeError("Unexpected MoE align call in source snapshot")

    entrypoint = source / ".venv/bin/vllm"
    entrypoint.unlink(missing_ok=True)
    entrypoint.write_text(
        "#!/bin/sh\n"
        "exec /opt/jonah_vllm/.venv/bin/python "
        '-m vllm.entrypoints.cli.main "$@"\n'
    )
    entrypoint.chmod(0o755)


if __name__ == "__main__":
    main()
