"""Write `name==version` for every distribution already in the image.

Used as a pip constraints file so installing requirements/dev.txt can never
replace a container package (torch, vllm, flash-attn, triton, transformers, ...).
"""

import importlib.metadata as md

seen: dict[str, str] = {}
for dist in md.distributions():
    name = dist.metadata["Name"]
    if name and name.lower() not in seen:
        seen[name.lower()] = f"{name}=={dist.version}"
print("\n".join(sorted(seen.values(), key=str.lower)))
