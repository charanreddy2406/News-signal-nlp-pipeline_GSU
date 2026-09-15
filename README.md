# Two Sigma | News-to-Signal NLP Research Pipeline

An advanced ML research-engineering capstone that mirrors the Two Sigma Quant-SWE GenAI job — translate an NLP research idea into a signal you can actually measure. You build an end-to-end pipeline that ingests and deduplicates a stream of unstructured news/filings, runs LLM/NLP extraction to pull structured signals (entities, events, sentiment) per document, embeds every document into a vector store for semantic retrieval and clustering, aligns each signal to a target return series on the correct as-of timeline, backtests whether the signal has predictive power with an information coefficient and a simple long/short returns test, and finally wraps the whole thing in an eval harness that reports signal quality end-to-end so a researcher can accept, refine, or kill the hypothesis — the paper-to-production discipline that turns a research idea into a measured result fast.

Built step-by-step with [KhwajaLabs Build](https://khwajalabs.com).

## Stack
- Python
- PyTorch
- embeddings
- vector store
- backtesting
