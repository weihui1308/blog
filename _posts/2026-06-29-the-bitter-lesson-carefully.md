---
title: The Bitter Lesson, Carefully
author: david
date: 2026-06-29 09:00:00 -1200
categories: [Vibeblogging, Machine Learning]
tags: [scaling-laws, deep-learning]
comments: false
pin: false
---

In 2019 Rich Sutton wrote an essay of about a thousand words called [*The Bitter Lesson*](http://www.incompleteideas.net/IncIdeas/BitterLesson.html). Its claim is simple and a little deflating: across 70 years of AI research, the methods that won were the ones that scaled with computation — general search and learning — not the ones where researchers hand-built their own cleverness into the system. Each generation tried to encode what it knew about chess, or vision, or language. Each generation was eventually beaten by a more general method with more compute behind it.

Sutton calls it bitter because it asks us to stop doing the part we enjoy most.

## Scale, made quantitative

For a long time this was a vibe. Then scaling laws turned it into arithmetic. The empirical finding, surveyed by Lilian Weng in [*Scaling Laws, Carefully*](https://lilianweng.github.io/posts/2026-06-24-scaling-laws/), is that a model's loss falls as a *power law* in three quantities: parameters `N`, data `D`, and training compute `C`. Plot it on log-log axes and you get a straight line.

That line is worth a lot. It means a few small, cheap runs can be extrapolated to a model nobody has trained yet — fit small, predict large. It even comes with a budgeting rule, since compute relates to the rest as roughly `C ≈ 6ND` ([Kaplan et al., 2020](https://arxiv.org/abs/2001.08361)). And there is a detail that should give any architecture-tinkerer pause: across translation, vision, language, and speech, a better architecture mostly shifts the line up or down. It rarely changes the *slope*. The slope, this literature argues, belongs to the problem, not to the design.

So the lesson is not merely "scale wins." It is that the returns to scale are predictable, and architectural taste is a smaller knob than it feels like.

## The word that does the work

If the story ended there, the advice would be: stop thinking, buy GPUs. Weng's "carefully" is a deliberate caution against that reading — the *direction* is right, but the *mechanism* is delicate.

Two cracks worth knowing about:

- **The fits are fragile.** These curves extrapolate small models orders of magnitude upward, so trivial choices swing the answer: how parameters are counted, what precision is used, whether the loss is summed or averaged. [Kaplan et al. (2020)](https://arxiv.org/abs/2001.08361) and the [Chinchilla paper (Hoffmann et al., 2022)](https://arxiv.org/abs/2203.15556) disagreed on how to split a compute budget — `N_opt ∝ C^0.73` versus `C^0.50` — and a later reconciliation traced most of the gap to bookkeeping (non-embedding vs total parameter counts). A separate replication (Besiroglu, 2024) even found numerical errors in Chinchilla's own parametric fit; the headline recipe survived, the exact coefficients did not.
- **The data runs out.** The classic laws assume an endless supply of fresh, high-quality tokens and a single pass over them. That assumption is expiring — the "data wall" — which pushes training into the awkward regime of repeating data, where the neat single-epoch math no longer holds cleanly.

None of this refutes Sutton. As Weng frames it, it disciplines him. Scale stays the central lever; it just is not an autopilot.

## What it looks like in practice

The most convincing evidence is not an argument but a model that quietly does the bitter thing. [Masked Autoencoders (He et al., 2022)](https://arxiv.org/abs/2111.06377) state the lesson almost verbatim — "simple algorithms that scale well are the core of deep learning" — and back it with a deliberately plain objective: hide 75% of an image, reconstruct the rest, scale the backbone up. [DINOv3 (Siméoni et al., 2025)](https://arxiv.org/abs/2508.10104) makes the same bet at a different size: one general self-supervised algorithm, a 7B-parameter backbone, 1.7B images, no task-specific labels — and it beats specialized supervision, especially on dense prediction.

The pattern repeats. Pick the most general method you can stomach, remove your own cleverness where you can, and pour in compute — but watch the fit, watch the data, and do not mistake a power law for a promise.

Scale wins. Carefully.

## References

- Sutton, R. (2019). [*The Bitter Lesson*](http://www.incompleteideas.net/IncIdeas/BitterLesson.html).
- Weng, L. (2026). [*Scaling Laws, Carefully*](https://lilianweng.github.io/posts/2026-06-24-scaling-laws/). Lil'Log.
- Kaplan, J. et al. (2020). [*Scaling Laws for Neural Language Models*](https://arxiv.org/abs/2001.08361). arXiv:2001.08361.
- Hoffmann, J. et al. (2022). [*Training Compute-Optimal Large Language Models*](https://arxiv.org/abs/2203.15556) (Chinchilla). arXiv:2203.15556.
- He, K. et al. (2022). [*Masked Autoencoders Are Scalable Vision Learners*](https://arxiv.org/abs/2111.06377). CVPR.
- Siméoni, O. et al. (2025). [*DINOv3*](https://arxiv.org/abs/2508.10104). arXiv:2508.10104.
