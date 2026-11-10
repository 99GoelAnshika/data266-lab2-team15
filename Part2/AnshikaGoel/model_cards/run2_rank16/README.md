---
library_name: diffusers
pipeline_tag: text-to-image
base_model: stable-diffusion-v1-5/stable-diffusion-v1-5
datasets:
- lambdalabs/naruto-blip-captions
tags:
- stable-diffusion
- diffusers
- lora
- text-to-image
- naruto
- data266
---

# DATA 266 Lab 2 Part 2 — Stable Diffusion 1.5 Naruto LoRA Rank 16

This repository contains the **Run 2 rank-16 LoRA adapter** selected as the final fine-tuned model for DATA 266 Lab 2 Part 2.

The adapter was fine-tuned from **Stable Diffusion 1.5** on the **Naruto BLIP Captions** dataset. The experiment compares the original Base model, a rank-4 LoRA adapter, and this rank-16 LoRA adapter under matched training and generation conditions.

## Model Details

| Field | Value |
|---|---|
| Model type | LoRA adapter for Stable Diffusion 1.5 |
| Hugging Face repository | `99AnshikaGoel/data266-lab2-part2-run2-lora-rank16` |
| Base model | `stable-diffusion-v1-5/stable-diffusion-v1-5` |
| Base revision | `451f4fe16113bff5a5d2269ed5ad43b0592e9a14` |
| Dataset | `lambdalabs/naruto-blip-captions` |
| Dataset revision | `1ed13e8ef280bd45e3bbac4cfa8bbd9d64ec9f89` |
| LoRA rank | 16 |
| LoRA alpha | 16 |
| LoRA scale used for evaluation | 1.0 |
| Target modules | `to_q, to_k, to_v, to_out.0` |
| Adapter filename | `pytorch_lora_weights.safetensors` |
| Adapter size | 12.2003 MB |

## Intended Use

This adapter is intended for educational and research use involving domain adaptation of Stable Diffusion 1.5 toward Naruto/anime-style imagery.

It is primarily a course experiment demonstrating parameter-efficient fine-tuning, controlled model comparison, quantitative evaluation, human evaluation, safety filtering, and memorization screening.

It has **not** been validated for production, safety-critical, or commercial deployment.

## Training Data

The experiment used `lambdalabs/naruto-blip-captions` at revision:

`1ed13e8ef280bd45e3bbac4cfa8bbd9d64ec9f89`

The original dataset contained **1221 examples**.

Data-quality auditing found:

- corrupted images removed: **0**
- invalid captions removed: **0**
- exact duplicate rows removed: **0**
- perceptual duplicate candidates under the predefined threshold: **0**

Therefore, **1221 examples** were retained.

A deterministic split with seed **266** produced:

- **1098 training examples**
- **123 held-out test examples**

The held-out test set was excluded from fine-tuning.

### Dataset provenance and license status

The dataset card states that the original images were obtained from Narutopedia and captioned using a pretrained BLIP image-captioning model.

**No explicit dataset license is declared in the pinned Hugging Face repository metadata.** No license is inferred or assigned here when the dataset publisher does not provide one.

The absence of an explicit repository license should not be interpreted as granting unrestricted redistribution or commercial-use rights.

## Training Procedure

Both LoRA runs used the same training data, optimizer-step budget, learning rate, effective batch size, stochastic seed, and deterministic training schedule. The primary controlled difference was LoRA capacity.

| Training setting | Run 2 |
|---|---:|
| LoRA rank | 16 |
| LoRA alpha | 16 |
| Trainable parameters | 3,188,736 |
| Trainable percentage | 0.369619% |
| Optimizer | AdamW |
| Learning rate | 0.0001 |
| LR scheduler | constant |
| Warmup steps | 0 |
| Physical batch size | 1 |
| Gradient accumulation | 4 |
| Effective batch size | 4 |
| Optimizer steps | 1000 |
| Sample exposures | 4000 |
| Equivalent epochs | 3.6430 |
| Mixed precision | fp16 |
| Gradient checkpointing | True |
| Stochastic seed | 266 |

Training loss is reported for reproducibility but is **not** used as the primary quality metric.

- first 50-step mean loss: **0.090991**
- last 50-step mean loss: **0.083833**
- overall mean loss: **0.086895**

## Training Efficiency

Run 2 required:

- **56.01 minutes**
- **0.9336 GPU-hours**
- **2.5116 GB peak allocated GPU memory**
- **2.5703 GB peak reserved GPU memory**

Compared with the rank-4 Run 1 adapter, Run 2 used exactly **4.0x** as many trainable parameters while increasing:

- training time by **1.73%**
- GPU-hours by **1.73%**
- peak allocated GPU memory by **1.59%**

## Official Generation Protocol

The frozen evaluation protocol used:

| Setting | Value |
|---|---|
| Resolution | 512 × 512 |
| Scheduler | PNDMScheduler |
| Inference steps | 30 |
| Guidance scale | 7.5 |
| Seeds | [0, 1] |
| Evaluation prompts | 20 |
| Images per prompt | 2 |
| Official images per model | 40 |
| Safety checker | Enabled |

The 20 frozen prompts consisted of:

- 10 held-out in-domain prompts
- 5 out-of-domain prompts
- 5 compositional prompts

## Evaluation

Exactly four formal evaluation types were used:

1. Kernel Inception Distance (KID)
2. CLIP prompt-image alignment
3. LPIPS seed diversity
4. Blinded human preference

Safety filtering, memorization screening, efficiency measurement, and qualitative failure analysis are reported separately and are not counted as additional evaluation types.

### Quantitative comparison

| Metric | Base | Run 1 rank 4 | Run 2 rank 16 | Direction |
|---|---:|---:|---:|---|
| KID — in-domain | 0.144058 | 0.089212 | **0.034975** | Lower is better |
| CLIP alignment — overall | **0.324159** | 0.309442 | 0.312618 | Higher is better |
| LPIPS seed diversity — official | **0.752897** | 0.663034 | 0.751837 | Higher means greater seed diversity |

### KID

KID was calculated only for the in-domain generated images because the real reference distribution consisted of the **all 123 held out test images**.

Run 2 achieved the best KID:

**0.034975 ± 0.005391**

compared with:

- Run 1: **0.089212**
- Base: **0.144058**

This indicates substantially closer distributional similarity to the held-out Naruto test images under this KID configuration.

KID should not be interpreted as a complete measure of semantic correctness or human-perceived quality.

### CLIP prompt alignment

Mean CLIP prompt-image similarity over all 40 official images per condition was:

- Base: **0.324159**
- Run 1: **0.309442**
- Run 2: **0.312618**

The Base model therefore achieved the best overall CLIP alignment.

This result shows that stronger domain adaptation does not necessarily improve general prompt-image alignment.

### LPIPS seed diversity

Official mean LPIPS between seed 0 and seed 1 for the same prompt was:

- Base: **0.752897**
- Run 1: **0.663034**
- Run 2: **0.751837**

Run 2 remained close to the Base model in official seed diversity.

Safety-filtered black outputs were retained in the official analysis, so LPIPS values involving filtered outputs require caution.

For Run 2, the clean-pair diagnostic excluding safety-affected prompt pairs produced mean LPIPS:

**0.715023**

over **17 clean prompt pairs**.

### Blinded human preference

The final Run 2 model was compared directly with Base using:

- **25 blinded image pairs**
- **3 raters**
- **75 total judgments**

Results:

| Human judgment | Count | Share of all judgments |
|---|---:|---:|
| Run 2 preferred | 15 | 20.00% |
| Base preferred | 31 | 41.33% |
| Tie | 29 | 38.67% |

Among decisive judgments:

- Run 2 win rate: **32.61%**
- Base win rate: **67.39%**

Therefore, although Run 2 improved in-domain KID substantially, the Base model was preferred more frequently by human raters overall.

## Safety Filtering

The Stable Diffusion safety checker remained enabled during official generation.

Run 2 produced:

- **40 official images**
- **3 safety-filtered outputs**

Safety-filtered outputs were retained unchanged in the official evaluation rather than silently regenerated or replaced.

These flags should not automatically be interpreted as confirmed harmful content; safety-checker false positives or domain shift may contribute.

## Memorization Screening

A CLIP nearest-neighbor screening compared **10 in-domain Run 2 generations** against all **1098 training images**.

The highest observed CLIP cosine similarity was:

**0.909221**

The three highest-similarity generated/training pairs were visually inspected.

**No clear visual evidence of direct training-image copying was observed among the three highest-CLIP nearest neighbors.**

Important limitation:

This is a screening analysis, not proof that memorization is absent. CLIP similarity captures semantic and stylistic resemblance as well as visual similarity, and only ten generated images were screened.

## Qualitative Findings

Eight deterministic comparison grids were selected with seed **266** without using generated-image quality for selection.

The grids represented:

- 4 in-domain prompts
- 2 out-of-domain prompts
- 2 compositional prompts

Base SD 1.5 frequently remains photographic or semi-photographic, while both LoRA runs consistently shift the selected in-domain generations toward the Naruto/anime visual domain.

For the selected strawberry and skyscraper prompts, all three models preserve the major requested semantic content, indicating that fine-tuning did not eliminate general text-to-image capability.

### Observed failure modes

**1. Multi-object attribute binding**

The `COMP_02` prompt requires two warriors with distinct held objects. The models often preserve the two-subject relationship but fail to reliably assign a sword to one subject and a lantern to the other.

**2. Multi-element spatial composition**

The `COMP_04` prompt requires a masked ninja jumping over a river with snow-covered mountains in the background. The models usually capture the character/action and mountain setting but do not consistently preserve the river as a distinct spatial component.

## Limitations

This model has several important limitations:

1. **Machine-generated training captions.**  
   Dataset captions were produced using a pretrained BLIP model and may be generic, incomplete, grammatically imperfect, or visually misaligned.

2. **Narrow training domain.**  
   The dataset is Naruto/anime-specific and is not representative of general natural-image distributions.

3. **Human preference did not improve overall.**  
   Despite substantially improved in-domain KID, Base was preferred more frequently in the blinded human evaluation.

4. **Prompt-composition failures remain.**  
   Multi-object binding and complex spatial composition remain unreliable.

5. **Safety-checker sensitivity.**  
   Run 2 produced 3/40 safety-filtered outputs in the official evaluation.

6. **Memorization analysis is limited.**  
   The nearest-neighbor analysis screened only 10 generated images and cannot prove the absence of memorization.

7. **Dataset licensing is unresolved.**  
   The pinned dataset repository does not declare an explicit license. Downstream-use and redistribution rights require separate consideration.

## Loading the Adapter

```python
import torch

from diffusers import StableDiffusionPipeline


base_model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"

base_revision = "451f4fe16113bff5a5d2269ed5ad43b0592e9a14"

lora_repo_id = "99AnshikaGoel/data266-lab2-part2-run2-lora-rank16"


pipe = StableDiffusionPipeline.from_pretrained(
    base_model_id,
    revision=base_revision,
    torch_dtype=torch.float16,
    use_safetensors=True,
)


pipe.load_lora_weights(
    lora_repo_id,
    weight_name="pytorch_lora_weights.safetensors",
)


pipe.enable_model_cpu_offload()

pipe.vae.enable_slicing()


image = pipe(
    "a masked ninja standing beside a red torii gate at night",
    num_inference_steps=30,
    guidance_scale=7.5,
    generator=torch.Generator(
        device="cpu"
    ).manual_seed(
        266
    ),
    height=512,
    width=512,
).images[0]
```

The LoRA adapter is intended to be loaded on top of the exact Stable Diffusion 1.5 base model identified above.

## Related Experiment Repository

Rank-4 comparison adapter:

`99AnshikaGoel/data266-lab2-part2-run1-lora-rank4`

## Reproducibility Notes

Key experiment controls:

- dataset split seed: **266**
- stochastic training seed: **266**
- frozen prompt manifest: **20 prompts**
- official generation seeds: **0 and 1**
- same training schedule for Run 1 and Run 2
- held-out test examples excluded from training
- official safety-filtered outputs retained in evaluation

The complete experiment notebook, evaluation tables, qualitative grids, and audit outputs are maintained in the DATA 266 Lab 2 repository.

## Responsible Interpretation

Run 2 is the strongest fine-tuned candidate in this experiment according to in-domain KID and selected fine-tuned-model comparisons, but it is **not uniformly superior to the Base model**.

The Base model achieved higher overall CLIP prompt alignment and was preferred more frequently by human raters.

The results therefore demonstrate a tradeoff between narrow-domain adaptation and broader prompt-alignment / human-preference behavior rather than a universal quality improvement.
