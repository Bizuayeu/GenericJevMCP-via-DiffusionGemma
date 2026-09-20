# Upstream provenance

The runtime/overlay files are Apache-2.0 vLLM sources from mmastrac/vllm 6591b093b29536dd070c6af3628b734025c53e23. Original SPDX headers are retained.

The mixed-logprob concatenation fix adapts razorback16/vllm commit 9bbf7418e85020dc76da9f60cdfe6c4e912ec048 (Apache-2.0). The pinned-base overlay approach and Dynamo limit follow mmastrac/djev-spark 4405eef82b7be1fa832a35cb7f87d1f5c3f86c8d. The device cap is independently implemented using PyTorch.

The separate-question execution design and calibration.py are adapted from
ikermoel/open-alternative-jev commit 98774715d1b05c5310df820c8727054787caaf20
(https://github.com/ikermoel/open-alternative-jev, Apache-2.0).
The upstream license is retained in licenses/open-alternative-jev.txt.
Local changes: DiffusionGemma-specific sequential question isolation, stable slot/seed,
self-contained RAG support checks, strict probability/temperature validation, and
disjoint-context held-out evaluation. Calibration stays offline; serving is uncalibrated.
The autoregressive packed-prompt path and missing-label floor fallback are not used.
