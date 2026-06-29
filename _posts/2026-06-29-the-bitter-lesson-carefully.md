---
title: The Bitter Lesson, Carefully
author: david
date: 2026-06-29 09:00:00 -1200
categories: [Vibeblogging, Machine Learning]
tags: [scaling-laws, deep-learning]
comments: false
pin: false
---

In 2019, Rich Sutton wrote an essay of about a thousand words called *The Bitter Lesson*. The claim is simple and a little deflating: over 70 years of AI research, the methods that won were the ones that scaled with computation — general search and learning — not the ones where we hand-built our own cleverness into the system. Every generation of researchers tried to encode what they knew about chess, or vision, or language. Every generation got beaten, eventually, by a more general method with more compute behind it.

It is a bitter lesson because it asks us to stop doing the part we enjoy most.

## Scale, made quantitative

For a long time this was a vibe — a story you told over coffee. Then scaling laws turned it into arithmetic. The empirical finding is that a model's loss falls as a *power law* in three quantities: the number of parameters `N`, the amount of data `D`, and the compute `C` you spend training. Plot it on log-log axes and you get a straight line.

That straight line is worth a lot. It means you can run a few small, cheap experiments and extrapolate to a model you haven't trained yet — fit small, predict large. It even gives you a budgeting rule, since compute relates to the rest as roughly `C ≈ 6ND`. And here is the part that should make any architecture-tinkerer pause: across translation, vision, language, and speech, a better architecture mostly shifts the line up or down. It rarely changes the *slope*. The slope belongs to the problem, not to your design.

So the bitter lesson isn't just "scale wins." It's "the returns to scale are predictable, and your architectural taste is a smaller knob than you think."

## The word that does the work

If the story ended there, the advice would be: stop thinking, buy GPUs. But the more careful reading — and I think the correct one — is that the *mechanism* is delicate even though the *direction* is right.

A few cracks worth knowing about:

- **The fits are fragile.** These curves extrapolate small models orders of magnitude upward, so trivial choices swing the answer: how you count parameters, what precision you use, whether you average or sum the loss, which models you fit on. Two landmark results, Kaplan (2020) and Chinchilla (2022), disagreed on how to split a compute budget — `C^0.73` versus `C^0.50` — and the gap turned out to come largely from bookkeeping. A later replication even found numerical bugs in Chinchilla's own fit.
- **The data runs out.** The classic laws assume an endless supply of fresh, high-quality tokens and a single pass over them. That assumption is expiring. This is the "data wall," and it pushes us into the awkward regime of repeating data, where the neat single-epoch math no longer holds cleanly.

None of this refutes Sutton. It disciplines him. Scale is still the central lever. It just isn't an autopilot.

## What it looks like in practice

The most convincing evidence isn't an argument, it's a model that quietly does the bitter thing. Masked Autoencoders state the lesson almost word for word — "simple algorithms that scale well are the core of deep learning" — and then prove it with a deliberately dumb objective: hide 75% of an image, reconstruct the rest, scale the backbone up. DINOv3 makes the same bet at a different size: one general self-supervised algorithm, 7B parameters, 1.7B images, no task-specific labels — and beats specialized supervision, especially on dense prediction.

The pattern repeats. Pick the most general method you can stomach. Remove your own cleverness where you can. Pour in compute. But watch the fit, watch the data, and don't mistake a power law for a promise.

Scale wins. Carefully.
