---
title: Hide Most of the Picture
author: david
date: 2026-06-30 09:00:00 -1200
categories: [Vibeblogging, Computer Vision]
tags: [self-supervised-learning, vision-transformer, representation-learning]
comments: false
pin: false
---

How do you teach a vision model to understand images without labels? One of the cleanest answers came from a 2022 paper, [*Masked Autoencoders Are Scalable Vision Learners* (He et al.)](https://arxiv.org/abs/2111.06377). The recipe is almost insultingly simple: hide most of the image and ask the model to fill in the rest.

## The trick is how much you hide

Borrowing from BERT in language, you might mask 15% of an image and reconstruct it. The Masked Autoencoder (MAE) masks **75%**. That number is the whole idea.

Images are spatially redundant — a missing patch can usually be guessed by interpolating from its neighbors. At a low masking ratio that is exactly what happens, and the model learns nothing but local texture-copying. Remove three-quarters of the patches and interpolation collapses. To reconstruct the hole, the model has no choice but to form a more holistic sense of what the scene *is*. The high ratio is what forces understanding instead of cheating.

## Asymmetry makes it cheap

The second idea is structural, and it is what makes the whole thing scale. MAE uses an **asymmetric encoder-decoder**:

- The **encoder** — a standard Vision Transformer — sees only the visible 25% of patches. The masked patches are not fed in at all.
- A **lightweight decoder** then takes the encoded patches plus placeholder mask tokens and reconstructs the missing pixels. After pretraining the decoder is thrown away; only the encoder is kept.

Because the heavy encoder processes a quarter of the patches, pretraining runs several times faster and lighter than if it saw the whole image. That efficiency is not a footnote — it is what makes pretraining very large models on a single dataset practical in the first place.

## And the target is just pixels

There is no fancy tokenizer. The loss is plain mean-squared error on the masked patches, predicting normalized pixel values. The authors tried predicting discrete visual tokens, as some earlier work did, and found no advantage. Simpler and faster won.

## Why it matters

The payoff is that simplicity scales. A vanilla ViT-Huge trained this way reaches **87.8%** on ImageNet using only ImageNet data, and — more tellingly — transfers to detection and segmentation *better* than supervised pretraining, with the gap widening as the model grows.

MAE is also a tidy illustration of a larger thesis. Its own paper states it plainly: "simple algorithms that scale well are the core of deep learning." Mask three-quarters of the picture, reconstruct raw pixels, make the encoder do less work, and let scale carry the rest.

## References

- He, K., Chen, X., Xie, S., Li, Y., Dollár, P., & Girshick, R. (2022). [*Masked Autoencoders Are Scalable Vision Learners*](https://arxiv.org/abs/2111.06377). CVPR.
