---
title: Your Language Model Is Probably Undertrained
author: david
date: 2026-06-30 10:00:00 -1200
categories: [Vibeblogging, Machine Learning]
tags: [scaling-laws, llm, compute-optimal]
comments: false
pin: false
---

Suppose you have a fixed pile of compute and you want the best language model it can buy. Do you spend it on a bigger model, or on more training data? For a couple of years the field had the wrong answer.

## The Kaplan era: go big

The influential [Kaplan et al. (2020)](https://arxiv.org/abs/2001.08361) scaling laws pointed toward size. Roughly: grow the model much faster than the dataset, train a big model, and stop before it fully converges. The takeaway most people absorbed was "parameters are what matter," and the models of that era reflect it — enormous parameter counts trained on comparatively modest token budgets.

## The Chinchilla correction: feed it more

Then [Hoffmann et al. (2022)](https://arxiv.org/abs/2203.15556) — the Chinchilla paper — redid the analysis across more than 400 training runs and found something different. At a fixed compute budget, model size and data should scale *together*, at roughly equal rates: double the parameters, double the tokens (`N_opt ∝ C^0.50`, `D_opt ∝ C^0.50`).

The demonstration is the memorable part. DeepMind's earlier Gopher had 280B parameters trained on 300B tokens. Chinchilla used about **4× fewer parameters (70B) and about 4× more tokens (1.4T)** — and at matched compute it beat Gopher across the board. The uncomfortable corollary: most large language models of that period were badly *undertrained*. They were too big for the amount of data they had seen.

## The "carefully" footnote

This is now textbook, but it comes with a caution. As Lilian Weng's survey [*Scaling Laws, Carefully*](https://lilianweng.github.io/posts/2026-06-24-scaling-laws/) recounts, the Kaplan–Chinchilla gap (`C^0.73` vs `C^0.50`) was later traced largely to bookkeeping — non-embedding versus total parameter counts — rather than a deep disagreement. And a re-analysis (Besiroglu, 2024) found numerical errors in Chinchilla's own curve fit. The headline recipe survived; the precise numbers were shakier than they looked.

The practical lesson outlives the exact exponents, though. Before you reach for more parameters, ask whether you have fed the model enough. A smaller model on more data is often the better buy — and for a while, almost nobody was making it.

## References

- Kaplan, J. et al. (2020). [*Scaling Laws for Neural Language Models*](https://arxiv.org/abs/2001.08361). arXiv:2001.08361.
- Hoffmann, J. et al. (2022). [*Training Compute-Optimal Large Language Models*](https://arxiv.org/abs/2203.15556) (Chinchilla). arXiv:2203.15556.
- Weng, L. (2026). [*Scaling Laws, Carefully*](https://lilianweng.github.io/posts/2026-06-24-scaling-laws/). Lil'Log.
