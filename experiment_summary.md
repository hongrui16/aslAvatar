# Experiment Summary: ASL Motion Generation with Phonological Conditioning

## Research Objective

Develop a text-to-3D sign language motion generation system that replaces traditional CLIP conditioning with phonological attribute-based conditioning, using linguistic features as supervision signals for more accurate sign production.

---

## ASLLVD Skeleton Dataset

### Dataset Overview

- **Source**: ASL-Skeleton3D + ASL-Phono (derived from ASLLVD)
- **Scale**: ~9,000 JSON files, ~5-6 samples per gloss
- **Representation**: 3D skeleton coordinates (72 joints × 3 = 216 dimensions)
  - Upper body: 14 joints
  - Face (key landmarks): 16 joints
  - Hands: 21 joints × 2

### Frame Distribution Analysis

| Frames | Count | Percentage | Cumulative |
|--------|-------|------------|------------|
| 1      | 13    | 0.18%      | 0.18%      |
| 2      | 2,082 | 29.37%     | 29.56%     |
| 3      | 3,263 | 46.04%     | 75.59%     |
| 4      | 1,392 | 19.64%     | 95.23%     |
| 5      | 281   | 3.96%      | 99.20%     |
| 6      | 49    | 0.69%      | 99.89%     |
| 7      | 8     | 0.11%      | 100.00%    |

**Key Finding**: 75% of samples have only 2-3 frames (at 3 fps), representing merely 0.6-1 second of extremely sparse motion sampling.

### Training Attemp_1

**Architecture**: Transformer-based CVAE with curriculum learning

**Configuration**:
- Input dimension: 216 (72 joints × 3)
- Latent dimension: 256
- Model dimension: 512
- Attention heads: 8
- Layers: 4
- Sequence interpolation: Short sequences interpolated to minimum length

#### Result(Attemp_1): Loss became NaN at ~100 epochs (without gradient clipping)

![Training Loss Curve](figures/training_curves_exp1.png)
*Figure 1: Training loss curve showing divergence around epoch 100.*
---


### Training Attemp_2
1. Clamp `logvar` in `aslAvatarModel.py`

Find where the encoder outputs `mu` and `logvar`, and add:

```python
logvar = torch.clamp(logvar, min=-20, max=20)
```

2. Add gradient clipping in `train_v1.py`

In the training loop, after `backward()` and before `optimizer.step()`, add:

```python
self.accelerator.backward(loss)
torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # Add this line
self.optimizer.step()
```


#### Result(Attemp_2): 

![Training Loss Curve](figures/training_curves_exp2.png)
*Figure 2: Training loss curve.*

![Test example 1](figures/v2_013_lock.png)
*Figure 3: Result examples.*

![Test example 2](figures/v2_032_library.png)
*Figure 4: Result examples.*


---

## ASL SignBank 

### Keyframe Distribution Analysis


Total gloss folders: 3,186

| Frames | Count | Ratio | Frames | Count | Ratio | Frames | Count | Ratio |
|--------|-------|-------|--------|-------|-------|--------|-------|-------|
| 8      | 1     | 0.03% | 33     | 99    | 3.11% | 53     | 7     | 0.22% |
| 13     | 2     | 0.06% | 34     | 79    | 2.48% | 54     | 7     | 0.22% |
| 15     | 5     | 0.16% | 35     | 70    | 2.20% | 55     | 9     | 0.28% |
| 16     | 15    | 0.47% | 36     | 76    | 2.39% | 56     | 7     | 0.22% |
| 17     | 21    | 0.66% | 37     | 46    | 1.44% | 57     | 8     | 0.25% |
| 18     | 50    | 1.57% | 38     | 46    | 1.44% | 58     | 3     | 0.09% |
| 19     | 71    | 2.23% | 39     | 35    | 1.10% | 59     | 7     | 0.22% |
| 20     | 100   | 3.14% | 40     | 30    | 0.94% | 60     | 4     | 0.13% |
| 21     | 135   | 4.24% | 41     | 31    | 0.97% | 61     | 1     | 0.03% |
| 22     | 156   | 4.90% | 42     | 27    | 0.85% | 62     | 3     | 0.09% |
| 23     | 213   | 6.69% | 43     | 32    | 1.00% | 63     | 5     | 0.16% |
| 24     | 183   | 5.74% | 44     | 18    | 0.57% | 64     | 2     | 0.06% |
| 25     | 204   | 6.40% | 45     | 26    | 0.82% | 66     | 2     | 0.06% |
| 26     | 192   | 6.03% | 46     | 23    | 0.72% | 67     | 1     | 0.03% |
| 27     | 155   | 4.87% | 47     | 20    | 0.63% | 68     | 2     | 0.06% |
| 28     | 181   | 5.68% | 48     | 16    | 0.50% | 70–77  | 6     | 0.19% |
| 29     | 178   | 5.59% | 49     | 19    | 0.60% | 85     | 1     | 0.03% |
| 30     | 199   | 6.25% | 50     | 12    | 0.38% | 116    | 1     | 0.03% |
| 31     | 186   | 5.84% | 51     | 14    | 0.44% |        |       |       |
| 32     | 130   | 4.08% | 52     | 14    | 0.44% |        |       |       |



**Key Finding**: Distribution peaks at 23 frames (6.69%), with ~75% of glosses containing 20–32 keyframes. Median is approximately 27 frames, indicating substantially denser temporal sampling compared to the skeleton dataset.

## Next Steps

### Proposed Solution: ASL Signbank Video Processing

**Why ASL Signbank**:
- Complete videos (15-30 fps, 2-5 seconds per sign)
- Standardized recording environment (frontal view, clean background)
- Corresponding phonological annotations available
- ~14,000+ signs

**Proposed Pipeline**:

```
ASL Signbank Videos
        ↓
   SMPL-X Extraction (SMPLer-X / OSX)
        ↓
   Quality Filtering (confidence, occlusion)
        ↓
   Phonology Alignment (ASL-LEX 2.0)
        ↓
   Training Data
```

### Tool Options for SMPL-X Extraction

| Tool | Pros | Cons |
|------|------|------|
| SMPLer-X | Best quality, full body | Slower |
| OSX | Fast | Slightly lower quality |
| 4D-Humans | Good temporal consistency | May need adaptation |

### Expected Outcome

- ~10,000+ sign videos with dense SMPL-X sequences
- 30-150 frames per sign (vs. current 2-3 frames)
- Real motion trajectories instead of interpolated lines
- Combined with ASL-LEX phonological features for conditioning

---

## References

- ASL-Skeleton3D / ASL-Phono dataset
- ASL Signbank: https://www.signbank.org/signpuddle/
- ASL-LEX 2.0: Phonological database
- SMPLer-X: https://github.com/caizhongang/SMPLer-X
