# RFC: Capability-Aware Workflow Lowering for High-Resolution and Constrained Generation

Status: Draft / fork-only research RFC  
Scope: `kvnloo/ComfyUI` first; no upstream behavior change implied  
Primary use case: requests that exceed a model/provider's native image-generation limits

## Summary

ComfyUI currently exposes extremely flexible execution graphs, while local models and remote API providers each impose different hard and soft constraints:

- maximum width / height / megapixels
- supported aspect ratios
- maximum reference images
- edit / mask / image-to-image support
- frame or duration limits
- batching / concurrency constraints
- provider cost and latency
- local VRAM / RAM constraints

Today those constraints are mostly handled inside individual nodes or workflows.

This RFC proposes a small **capability-aware workflow compiler** between user intent and graph execution:

```text
requested operation
      |
      v
capability IR
      |
      v
strategy planner
      |
      v
lowering passes
      |
      v
temporary executable workflow
      |
      v
ComfyUI runtime
```

The first lowering problem is **oversized image generation**.

If a user requests 4096x4096 from a backend capped at 1024x1024, ComfyUI should be able to determine whether it can safely compile that request into:

- native generation
- native resolution extrapolation
- global composition + tiled refinement
- progressive outpainting
- upscale-only fallback
- or a clear rejection

The planner must never silently fake support. Every transformed request must be explainable, bounded by cost/call limits, and evaluated against quality/coherence benchmarks.

---

## Why this belongs above individual nodes

This is not just an "auto tile" node.

A node knows how to execute one operation. A planner needs to reason about **a request relative to backend capabilities**.

The same mechanism that resolves a resolution mismatch can later resolve:

- max reference-image count
- unsupported aspect ratio
- max video duration
- max frame count
- unavailable mask/edit support
- local VRAM limits
- provider batch limits
- cost ceilings
- latency preferences

That suggests a compiler-style architecture instead of provider-specific conditionals spread across nodes.

---

# Research basis

The design is informed by several generations of high-resolution diffusion research.

## 1. Joint / overlapping diffusion paths

**MultiDiffusion** binds multiple overlapping diffusion processes into a shared optimization problem, showing that large canvases should not be generated as unrelated independent tiles.

- Bar-Tal et al., *MultiDiffusion: Fusing Diffusion Paths for Controlled Image Generation*
- https://arxiv.org/abs/2302.08113

Implication for ComfyUI:

> Independent text-to-image tiles are a failure baseline, not an automatic production strategy.

Where latent-level joint denoising is available, overlapping windows should share state or constraints.

## 2. Synchronization for semantic coherence

**SyncDiffusion** shows that overlap averaging alone can still create globally incoherent montages, and adds perceptual synchronization across windows.

- Lee et al., *SyncDiffusion: Coherent Montage via Synchronized Joint Diffusions*
- https://arxiv.org/abs/2306.05178

Implication:

For local backends, "seamless" and "semantically coherent" are separate properties and should be measured separately.

## 3. Global / local signal separation

**ElasticDiffusion** explicitly separates global structure from local detail and uses a reference image to preserve overall composition.

- Haji-Ali et al., *ElasticDiffusion: Training-free Arbitrary Size Image Generation through Global-Local Content Separation*
- https://arxiv.org/abs/2311.18822

Implication:

The default safe high-res path should preserve a **global scaffold** before local refinement.

## 4. Progressive coarse-to-fine generation

**DemoFusion** and **MegaFusion** use progressive resolution growth and relay information from lower-resolution stages.

- Du et al., *DemoFusion*
- https://arxiv.org/abs/2311.16973
- Wu et al., *MegaFusion*
- https://arxiv.org/abs/2408.11001

Implication:

High-resolution lowering should be modeled as a multi-stage graph, not one oversized call.

Intermediate stages can also serve as previews and early-exit points.

## 5. Patch-specific conditioning

**AccDiffusion** identifies a key cause of repeated objects in patch-wise generation: using the same global prompt indiscriminately for every patch. It introduces patch-content-aware prompts and interacting windows.

- Lin et al., *AccDiffusion*
- https://arxiv.org/abs/2407.10738

Implication:

A future local high-res strategy may use region-specific conditioning, but patch prompts must remain anchored to the global scene to avoid semantic drift.

## 6. Effective receptive field adaptation

**ScaleCrafter** attacks object repetition and structural failure by adapting the effective receptive field during inference.

- He et al., *ScaleCrafter*
- https://arxiv.org/abs/2310.07702

Implication:

For local model strategies, tiling is not the only way to exceed native training resolution. The planner should permit model-native extrapolation strategies where the backend explicitly declares them supported.

## 7. Direct DiT resolution extrapolation

Recent DiT work increasingly treats high resolution as a position/attention extrapolation problem rather than only a tiling problem.

**DyPE** dynamically adjusts positional encoding over denoising steps and reports FLUX generation at ~16 MP without extra sampling cost.

- Issachar et al., *DyPE: Dynamic Position Extrapolation for Ultra High Resolution Diffusion*
- https://arxiv.org/abs/2510.20766

**UltraImage** analyzes repetition through positional frequencies and quality loss through diluted attention, reporting generation up to 6K x 6K from a ~1328px training scale.

- Zhao et al., *UltraImage*
- https://arxiv.org/abs/2512.04504

**RPE-2D** trains position-order generalization rather than memorizing fixed position distances.

- Liu et al., *Boosting Resolution Generalization of Diffusion Transformers with Randomized Positional Encodings*
- https://arxiv.org/abs/2503.18719

Implication:

The planner needs a distinction between:

```text
backend hard native limit
backend safe extrapolation limit
backend experimental extrapolation strategy
```

A local FLUX-like model may have a better strategy than API-level tiling.

## 8. Direct 4K models and evaluation

**Diffusion-4K** highlights that ultra-high-resolution evaluation needs both holistic and fine-detail measurements.

- Zhang et al., *Diffusion-4K*
- https://arxiv.org/abs/2503.18352

Implication:

Our benchmark must measure both global semantics and local detail. A sharp tile mosaic with wrong global composition is not a win.

---

# Existing ComfyUI surfaces to herd rather than replace

This RFC should align with existing work instead of creating a parallel architecture.

## Upstream issue #15310: workflow capability/service layer

The issue already proposes explicit workflow manifests, request validation, node patching, dynamic inputs, and `maxResolution`-style capability metadata.

Our contribution:

> Add a backend/model capability IR and a planner that can transform requests when declared capabilities are insufficient.

This is complementary, not competing.

## Upstream issue #11938: image edit API interface

Image edit/inpaint/outpaint capability is exactly what makes remote high-resolution refinement possible.

Without edit/reference/mask support, many oversized requests should be rejected or fall back to simple upscale.

## Upstream PR #14956: continuous sampler + VAE batching

A future planner may generate many tile/refinement jobs.

If those jobs are local and compatible, batching/scheduling work can reduce the latency penalty.

This RFC should treat scheduling as a downstream execution optimization, not reimplement it.

## Upstream PR #15359: observability

The proposed Prometheus metrics already cover workflow/node latency and VRAM.

The benchmark layer should reuse upstream observability where possible rather than inventing parallel instrumentation.

## Existing tiled VAE paths

Core tiled VAE encode/decode solves **memory pressure for encoding/decoding**.

It does not solve semantic coherence when generation itself exceeds a model/provider's effective resolution.

The planner must keep those two meanings of "tiling" separate.

---

# Architecture

## A. Request IR

A normalized user intent:

```yaml
operation: text_to_image
width: 4096
height: 4096
prompt: "..."
references: []
policy:
  oversize: auto
  max_calls: 24
  max_cost_usd: 0.50
  quality_bias: balanced
  allow_experimental: false
```

This request should be independent of any provider or node implementation.

## B. Capability IR

Example:

```yaml
backend_id: provider/model
media: image

operations:
  text_to_image: true
  image_to_image: true
  edit: true
  inpaint: true
  outpaint: true

limits:
  max_width: 1024
  max_height: 1024
  max_pixels: 1048576
  dimension_multiple: 64
  max_reference_images: 4

conditioning:
  prompt: true
  seed: true
  reference_image: true
  mask: true
  strength: true

resolution:
  native_max: [1024, 1024]
  extrapolation:
    supported: false
    max_tested: null
    strategy: null

execution:
  local_or_remote: remote
  max_concurrency: 4

economics:
  cost_model: per_image
  estimated_cost_usd: 0.02
```

Unknown values must remain unknown.

Never infer a provider capability simply because another provider has it.

## C. Strategy planner

Initial strategies:

### 1. `native`

Use the requested dimensions directly.

Requirements:
- request is within declared safe limits

### 2. `native_extrapolation`

Run a model-native high-resolution method.

Requirements:
- local/backend implementation explicitly declares a supported strategy
- requested dimensions are within a tested range or user allows experimental execution

Possible future local implementations:
- position extrapolation
- receptive-field modification
- attention concentration
- coarse-to-fine native denoising

### 3. `global_then_tiled_refine`

Default high-quality fallback for capability-limited image backends.

```text
global native-size composition
        |
        v
resize to target canvas
        |
        v
overlapping contextual edit tiles
        |
        v
seam / consistency validation
        |
        v
final image
```

Requirements:
- a global generation path
- edit/img2img/inpaint capability
- enough contextual pixels around each writable tile
- call/cost budget allows it

### 4. `progressive_outpaint`

Grow the image region-by-region while retaining previous output as context.

Useful when:
- the requested aspect ratio is outside the backend's native shape set
- edit/outpaint support is stronger than img2img support

### 5. `upscale_only`

Generate at the largest coherent native resolution and use a non-generative or generative super-resolution step.

Use when:
- global composition matters more than adding genuinely new scene content
- no safe tile-edit path exists

### 6. `reject`

Required when the requested semantics cannot be preserved with the declared capabilities.

A rejection must include an actionable reason and alternatives.

---

# High-resolution tiled refinement design

## Global scaffold first

The first pass establishes:
- subject count
- composition
- camera/viewpoint
- coarse geometry
- color/lighting
- text placement if possible

This becomes the source of truth for local refinement.

## Context tile vs writable tile

Do not expose only the final tile crop to the provider/model.

Each task should have:

```text
+-------------------------+
|      context margin     |
|   +-----------------+   |
|   | writable center |   |
|   +-----------------+   |
|      context margin     |
+-------------------------+
```

The model sees context beyond the region it is allowed to modify.

This reduces seam and object-boundary instability.

## Overlap

Tile overlap should be:
- configurable
- derived from target scale and backend behavior
- benchmarked rather than hardcoded globally

## Ordering

Potential modes:
- independent parallel tiles
- checkerboard phases
- wavefront
- saliency-first
- coarse semantic regions first

Remote providers may need partially sequential execution when one edited region must become context for a later region.

## Adaptive tiling

Uniform grids are easy but often waste calls.

A later planner can allocate:
- larger tiles in flat / texture-homogeneous regions
- smaller tiles near faces, text, edges, object boundaries, or high-frequency content
- higher overlap around semantically fragile boundaries

This should be guided by measurable benefit, not aesthetic intuition alone.

## Prompt conditioning

Start conservative:

```text
global prompt + global reference crop
```

Later experimental modes can add:
- crop captions
- detected object labels
- regional prompts
- global + local prompt fusion

Patch-specific prompts should remain optional because over-localization can destroy global consistency.

---

# Remote API vs local model lowering

These should share the planner but not pretend to have the same execution primitives.

## Remote API path

Available primitives may include:
- generation
- image edit
- reference images
- masks
- seeds
- concurrent calls

Unavailable:
- cross-window latent sharing
- per-step attention manipulation
- synchronized diffusion gradients

Therefore remote coherence relies more on:
- global scaffold
- contextual edit crops
- overlap masks
- staged ordering
- compositing

## Local model path

Can additionally support:
- latent overlap consensus
- shared noise
- MultiDiffusion-style window fusion
- SyncDiffusion-style guidance
- receptive-field changes
- position extrapolation
- attention modification
- intermediate feature sharing

The capability IR should expose these advanced primitives explicitly.

---

# Strategy selection

A first deterministic planner might use rules rather than learned routing.

Example:

```text
within native limits?
  yes -> native

declared safe native extrapolation?
  yes -> native_extrapolation

has edit/reference + target fits call budget?
  yes -> global_then_tiled_refine

has outpaint?
  yes -> progressive_outpaint

has upscale path?
  yes -> upscale_only

otherwise -> reject
```

Later this can become Pareto-aware:

```text
minimize:
  cost
  latency
  seam risk
  semantic drift
  VRAM
subject to:
  requested resolution
  hard provider limits
  quality floor
```

---

# Benchmark before automation

`auto` should not exist until we can distinguish good strategies from bad ones.

## Required baselines

1. native max resolution
2. simple resize/upscale
3. naive independent tiles
4. global + overlapping refinement
5. progressive outpaint
6. native high-resolution model/provider when available

## Global metrics

- prompt alignment
- subject/object count consistency
- layout preservation
- repeated-object frequency
- low-res scaffold consistency
- face / text identity stability where applicable

## Local metrics

- seam energy
- overlap perceptual disagreement
- edge discontinuity
- texture detail
- high-frequency preservation

## System metrics

- wall-clock latency
- p50 / p95 latency
- number of model/provider calls
- cost
- peak VRAM
- peak RAM
- data transfer
- failures / retries

## Stress scenes

- single face
- multiple faces
- typography / signage
- repeated architecture
- interiors
- panoramas
- crowds
- fine texture
- line art
- strong perspective
- objects crossing tile boundaries

All output images and machine-readable traces should be retained for audit.

---

# Workstream / patch ladder

This RFC is intentionally larger than anything we should send upstream at once.

The implementation should be decomposed.

## Slice 0: research + benchmark

- establish benchmark fixtures
- implement naive baselines
- produce reproducible quality/cost/latency reports

No production behavior.

## Slice 1: capability IR

- typed capability object
- unknown-value semantics
- adapters for 1-2 existing API nodes
- unit tests

No graph rewriting.

## Slice 2: dry-run planner

Input:
- request IR
- capability IR

Output:
- chosen strategy
- generated tile plan
- call/cost estimate
- explanation

No execution.

## Slice 3: one remote oversize strategy

Pick one provider that supports:
- image edit
- mask/reference input
- deterministic enough behavior

Implement:
- global scaffold
- contextual overlapping tiles
- center masks
- stitch

Keep provider-specific code behind an adapter.

## Slice 4: quality gate

Run the benchmark.

Do not call it `auto` unless it beats naive baselines on the agreed scene set.

## Slice 5: local latent strategy

Prototype a local high-res method using shared latent context.

Possible research branches:
- MultiDiffusion-style overlap consensus
- AccDiffusion-style region conditioning
- native DiT extrapolation

## Slice 6: Pareto planner

Select strategies based on:
- cost
- quality
- latency
- memory
- provider/local availability

---

# Herding strategy for upstream

The fork RFC is the private/full vision.

Upstream contribution should remain narrow.

## Existing threads to monitor / help

- #15310: capability/service description
- #11938: image edit API surface
- #14956: continuous batching
- #15359: observability/metrics
- provider/API node PRs that expose hard limits and capabilities

## Public strategy

Do not post this entire RFC upstream.

Instead:

1. identify a small missing prerequisite
2. connect it to an existing upstream issue/PR
3. contribute evidence, test cases, or a tiny interface
4. let that primitive land
5. move to the next prerequisite

Example possible sequence:

```text
provider hard-limit metadata
        |
        v
standard capability object
        |
        v
dry-run validation/planning
        |
        v
one opt-in lowering transform
        |
        v
benchmark evidence
        |
        v
broader auto mode
```

The aim is to make each upstream step independently useful even if the full compiler never lands.

---

# Non-goals

- Automatically understanding arbitrary custom workflows.
- Replacing user-authored workflows.
- Hiding provider pricing.
- Silently changing image semantics.
- Running dozens of paid API calls without a budget.
- Treating VAE memory tiling as equivalent to generation tiling.
- Claiming all backends support the same constraints.
- Posting a monolithic upstream RFC before prerequisite primitives exist.

---

# Safety / predictability invariants

- `native_only` must never rewrite the request.
- `auto` must have a configurable call and cost ceiling.
- Hard provider limits are never exceeded.
- Experimental native extrapolation is opt-in unless explicitly marked stable.
- A strategy must declare why it is valid for the selected backend.
- If coherence cannot be reasonably preserved, reject instead of silently generating independent tiles.
- Planning is deterministic for identical capability/request inputs.
- Every transformed execution records provenance.

---

# Longer-term direction

The high-resolution planner is the first instance of a broader concept:

## ComfyUI as an intent-to-execution compiler

```text
intent
  |
capability graph
  |
constraint solver
  |
workflow lowering
  |
scheduler
  |
runtime
  |
telemetry
  |
evaluation / learned policy
```

Eventually, a request could specify:

```yaml
goal:
  operation: image_edit
  output_resolution: 4096x4096
constraints:
  cost_usd: <= 0.50
  latency_s: <= 60
  quality: high
resources:
  allow_local: true
  allow_remote: true
```

The system could choose among:
- a local model
- a remote provider
- native high-resolution generation
- staged coarse-to-fine generation
- tiled refinement
- hybrid local/remote execution

That turns ComfyUI from only a static graph editor into a graph execution substrate that can also **compile constrained user intent into workflows**.

This is the agenda this fork should experiment with before asking upstream to absorb any large architectural commitment.
