FROM vllm/vllm-openai@sha256:dea7fa047caa114167efccdeb42321b12d2c11da1add8728aa2662cb8d6b8cd5
COPY runtime /opt/dg-build
RUN python3 /opt/dg-build/install_runtime.py && python3 /opt/dg-build/test_runtime.py
ENV VLLM_USE_V2_MODEL_RUNNER=1 MAX_JOBS=2 FLASHINFER_NVCC_THREADS=1
LABEL llllm.runtime="gemma-stack-6591b09" llllm.fork="6591b093b29536dd070c6af3628b734025c53e23"
ENTRYPOINT ["vllm", "serve"]
