---
title: The Model That Got Better and Worse at Once
author: david
date: 2026-07-01 09:00:00 -1200
categories: [Vibeblogging, Computer Vision]
tags: [self-supervised-learning, vision-foundation-models, dense-features]
comments: false
pin: false
---

Train a self-supervised vision model long enough and something strange happens. On the headline benchmark it keeps improving. On another, quieter benchmark it starts getting *worse* — during the same run. The team behind [DINOv3 (Siméoni et al., 2025)](https://arxiv.org/abs/2508.10104) ran into exactly this while scaling a 7-billion-parameter backbone on 1.7 billion images, and the way they cornered it is a small lesson in what "the metric went up" can hide.

## Two kinds of "good"

A vision backbone produces two flavors of feature. **Global** features summarize the whole image — what you need to classify it. **Dense** features live at each patch — what you need to segment an image or estimate its depth, where every pixel gets a label.

Past roughly ViT-Large, DINOv3's global score (ImageNet linear probing) kept climbing with more training. But its dense scores (segmentation on VOC and ADE20k) *peaked around 200k iterations and then declined*. Judged only by the global number, the model was getting better. Judged by the dense number, longer training was actively hurting it.

## Found by looking, not by the loss

The loss curve did not reveal this — it was busy going down. The authors diagnosed it by *looking*: they visualized the cosine-similarity map between one reference patch and every other patch. Early in training the maps were clean and local — a patch was most similar to its genuine neighbors. Later, the maps turned noisy, with unrelated patches across the image lighting up as "similar." The global-summary token was quietly bleeding into the patch features and erasing their sense of place.

## Anchor the structure, free the features

The fix, called **Gram anchoring**, is elegantly narrow. Rather than forcing the late-stage features to match some earlier snapshot directly, it constrains only their *Gram matrix* — the grid of pairwise similarities between patches — to resemble that of an earlier, more local "Gram teacher." In other words: keep the *relationships* between patches well-structured, but let the features themselves keep moving and improving. Applied as a refinement phase, it repairs the dense degradation without giving up the global gains.

The result is a single frozen backbone that beats specialized, fine-tuned supervision across a wide range of tasks — with the biggest leap precisely on the dense predictions that were silently rotting before.

The moral rhymes with an old one: a rising benchmark is not the same as a model getting better at everything. Sometimes it is getting better at one thing while quietly getting worse at another. You only notice if you measure both — and if, now and then, you look at what the features are actually doing.

## References

- Siméoni, O. et al. (2025). [*DINOv3*](https://arxiv.org/abs/2508.10104). arXiv:2508.10104.
