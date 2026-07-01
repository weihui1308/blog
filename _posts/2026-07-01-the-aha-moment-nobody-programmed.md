---
title: The Aha Moment Nobody Programmed
author: david
date: 2026-07-01 10:00:00 -1200
categories: [Vibeblogging, Machine Learning]
tags: [reasoning, reinforcement-learning, llm]
comments: false
pin: false
---

One of the more surprising results in recent language-model research is that you can get a model to *reason* without ever showing it how. You reward the right answer, and the reasoning grows in on its own. The clearest case study, surveyed in Lilian Weng's [*Why We Think*](https://lilianweng.github.io/posts/2025-05-01-thinking/), is [DeepSeek-R1](https://arxiv.org/abs/2501.12948).

## Reward the outcome, not the steps

The core trick is reinforcement learning on problems whose answers can be checked automatically — math with a boxed final answer, code that either compiles and passes tests or doesn't. R1's rewards are deliberately blunt: a **format reward** (wrap the reasoning in the right tags) and an **accuracy reward** (is the final answer correct?). Notably, no one grades the individual reasoning steps. The model is judged only on where it lands.

## The "aha moment"

The striking variant is R1-Zero, trained with *pure* RL and no supervised fine-tuning at all — no curated examples of good reasoning to imitate. It still develops reflection and backtracking. It learns to pause, second-guess itself, and try another route — the emergent behavior the authors call the "aha moment." It even learns to spend more thinking tokens on harder questions without being told to. Reasoning behavior, it turns out, can fall out of outcome-based reward alone.

## What didn't work is just as telling

The report is refreshingly honest about dead ends. Two intuitive ideas failed. **Process reward models** — grading each reasoning step against a rubric — proved hard to define well and easy to game. And **Monte Carlo Tree Search**, which works beautifully for board games, buckled here: the space of possible token sequences is astronomically larger than the legal moves in Go, and the value estimates were too hard to train. Rewarding the whole outcome beat trying to score the path.

## The caveat

None of this makes "thinking longer" a universal cheat code. As Weng frames it, test-time reasoning is a genuine new axis for spending compute, complementing model size and data — but it helps most on easy-to-medium gaps and stalls on the genuinely hard ones. It cannot conjure a capability the base model never had. A strong pretrained foundation is still doing most of the quiet work underneath the visible reasoning.

Still, the headline stands. Sometimes the way to teach a skill is to stop teaching the steps, reward the result, and let the behavior grow toward it.

## References

- Weng, L. (2025). [*Why We Think*](https://lilianweng.github.io/posts/2025-05-01-thinking/). Lil'Log.
- DeepSeek-AI (2025). [*DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*](https://arxiv.org/abs/2501.12948). arXiv:2501.12948.
