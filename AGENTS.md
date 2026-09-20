# GenericJevMCP via DiffusionGemma

Read README.md (English) or README.ja.md (Japanese) and docs/validation.md before changing runtime behavior. Preserve pinned source/model revisions and generation defaults. CPU tests and GPU validation are separate. Do not commit weights, credentials, input corpora, client-config.json, state, or private records. Never print credentials.

Python: install requirements.txt, supply the pinned tokenizer as documented, then run python -m unittest discover -s tests -v. MCP: npm ci and npm test. Keep both README languages aligned. Device tests must identify hardware, cold/warm conditions and timing boundaries.
