## OpenADMET Blind Challenge: Predicting CYP Inhibition

The repository contains a description of the solution to the 
**OpenADMET Blind Challenge: Predicting CYP Inhibition** by the user `Shorku`.

Author: Oleg Gromov

Date: Aug 2026 - Nov 2026

### TL;DR

The main objective is to field-test learning from the full electronic
structure of a molecule on the contemporary data in real competition
conditions to obtain a fair performance and usability assessment. Hence, 
the solution is (for now) constrained to a single model architecture.
The focus is more on development and research than on benchmaxxing.

### Approach

<img src="https://github.com/Shorku/rhnet2/raw/main/images/intro.jpg" width="801" alt="302">

The core project's idea is to use raw quantum-chemical data as input for neural
networks to predict e.g. ADME properties. In the current implementation, the trick 
is to view 1-electron density matrix as a molecular graph directly consumable by GNNs.
More technical details can be found in [JCTC paper](https://doi.org/10.1021/acs.jctc.5c00425).

The original [RhNet2](https://github.com/Shorku/rhnet2) model was succeeded by 
[RhNet2TB](https://github.com/Shorku/rhnet2tb). It replaced expensive DFT input data 
with cheaper xTB input data (2-3 orders of magnitude). RhNet2TB prototype demonstrated some
promising results in the past **OpenADMET + ExpansionRx Blind Challenge** (e.g. ranked 11-th
in Human Liver Microsomal Clint prediction). 

In the ongoing challenge, I'm evaluating 
[RhNet2TBequi](https://github.com/Shorku/rhnet2tb/blob/main/rhnet2tbequi.py) 
model prototype, which replaces most
of the MLP components with SO(3)-equivariant machinery and dramatically reduces the 
memory footprint (no rotational augmentation needed anymore).

### General plan outline
- 17 Aug 2026 - 24 Sep 2026 (challenge start - intermediate LB): Baseline performance evaluation
    - Fitting and other technical settings, debug, etc.
    - Basic architecture features selection (e.g. depth, pooling options etc.)
    - Only the RhNet2TBequi model, only the challenge data, no calibration
- 25 Sep 2026 - 3 Nov 2026 (LB freeze - challenge ends): Everything else including external data (for sure),
ensembles (maybe), and other mischief (I solemnly swear).

### Validation

As the cross-validation (for the current setup and available resources) would
have severely limited the number of experiments possible within the challenge 
period, I've opted for a persistent train-val split. The split emulated the
challenge data story: I've chosen the top 'hits' for CYP1A2, CYP2C9, and CYP3A4 
endpoints from the challenge data, and their closest analogues (Tanimoto) were 
selected for the joint validation set i.e. if some molecule is selected for one 
isoform validation, then it was going to be used for validation within all the other 
isoforms as these will typically share a GNN-encoder and such hard split is the only way 
to avoid leakage. Generally, throughout the challenge the local validation metrics
for all endpoints but the CYP2D6 remained well-correlated with the public LB metrics.
Regarding the CYP2D6, it (retrospectively) turns out, that my local validation 
metrics could have been ignored all the way and an issue is not the particular
split type as it was a very common problem many other teams experienced.

### Data augmentation, aggregation, sampling

The input for a model here is a graph constructed from the density matrix, which 
is the result of a quantum-chemical calculation (xTB in this case) and is tied to 
a particular 3D-structure. A molecule may have several 
energetically accessible conformers i.e. a single label corresponds to multiple
different input graphs. The number of available conformers may range from 1 
to... wherever one sets a limit for the conformer generator. In the very basic 
case, to avoid the overrepresentation of the conformer-rich molecules in the 
training set, one may e.g. each epoch for each molecule sample at random equal 
number of conformers, which is the way I used e.g. in the OpenADMET+ExpansionRx Blind Challenge.

Generally, there is a following hierarchy: 

a single data **entry** may correspond to several **enantiomers**,

|__ each **enantiomer** may correspond to several **tautomers**,

|____ each **tautomer** may have several **conformers**,

|______ each **conformer**'s density may be calculated for infinite number of its spatial **orientations**

Here, I used the most straightforward approach: everything is aggregated and sampled by 
the parent entry's InChIKey. 

#### Augmentation with rotated structures

The equivariant implementation made inclusion of multiple differently oriented copies
of the same structure into the dataset optional. Unlike synthetically rotated 
densities, the densities that are actually calculated from the rotated structures
are a reasonable source of numerical noise and can be viewed as a way of regularization.
The effect of such regularization is especially visible when using small datasets, e.g.
with 3767 molecules (challenge data with fully specified stereochemistry only) the LB
metrics changed this way:

| Endpoint | MAE    | R²    | Spearman | Kendall |
| --- |--------|-------|----------|---------|
| 1A2 | -0.002⬇️| +0.00⬆️| -0.001🔽️| -0.001🔽️|
| 2C9 | -0.007⬇️| +0.02⬆️| +0.009⬆️| +0.010⬆️|
| 2D6 | -0.021⬇️| +0.04⬆️| -0.016🔽️| -0.011🔽️|
| 3A4 | -0.025⬇️| +0.03⬆️| +0.014⬆️| +0.014⬆️|

Whereas for an encoder trained on a dataset of 39750 molecules (~350,000 unique structures)
the effect of such augmentation was almost undetectable. 

#### Augmentation with conformers

Each molecule was represented by a set of the lowest energy conformers within 10kJ/mol window,
that were sampled randomly and equally. No experiments here. Although it may be beneficial to
tune e.g. the energy cutoff or to sample according to some distribution say Boltzmann, I expect
the gains to be small and highly case-specific. 

As a side note, it turned out, that the embeddings of two different conformers produced by a trained encoder
usually deviate significantly, but the linear head will still output almost matching predictions.   

#### Augmentation with tautomers

Augmenting the data with tautomers may be both quite beneficial or very harmful. 
At the previous PXR Induction challenge I came up with the following (quite conservative)
setup:

- 20kJ/mol energy window
- max 3-4 jumps while generating the tautomeric structures
- don't touch C(sp³)

This will usually introduce ~15% new structures and improve the metrics. Here, I used 
this setup as is and got a slightly controversial results at LB:

| Endpoint | MAE      | R²      | Spearman  | Kendall   |
| --- |----------|---------|-----------|-----------|
| 1A2 | +0.005🔼️ | -0.00🔽️ | -0.001🔽️ | -0.002🔽️ |
| 2C9 | +0.002🔼️ | -0.01🔽️ | +0.003⬆️ | +0.002⬆️ |
| 2D6 | +0.004🔼️ | -0.01🔽️ | +0.008⬆️ | +0.007⬆️ |
| 3A4 | -0.007⬇️ | +0.01⬆️ | +0.005⬆️ | +0.006⬆️ |

Augmentation with tautomers significantly improved the metrics for the most well-behaved 
endpoint i.e. CYP3A4, while for the other endpoints Spearman and Kendall got solid gains
while MAE slightly increased. 

#### Handling enantiomers

Approximately 40% of the molecules in the challenge data lack the full stereochemistry
specification which presented me with a dilemma: simply dumping all of this into the 
training set would mean pushing the (actually chirality-sensitive!!!) model towards full 
invariance (same label for both enantiomers), whereas ignoring this data would mean 
wasting lots of potentially useful information. As a compromise, I included this 
uncompletely specified data as follows:
- an entry is excluded, if at least one of its enantiomers has its own entry
- all the enantiomers get the parent entry's identifier and are aggregated together

This way the model could learn on the whole dataset, but the impact of the mixed entries
was slightly marginalized (the probability of each of their enantiomer's structures to get sampled
was at least halved compared to the fully specified structures).

| Endpoint | MAE       | R²      | Spearman | Kendall  |
| --- |-----------|---------|----------|----------|
| 1A2 | -0.009⬇️| +0.01⬆️| +0.025⬆️| +0.022⬆️|
| 2C9 | -0.007⬇️| +0.02⬆️| +0.004⬆️| +0.005⬆️|
| 2D6 | +0.064🔼️| -0.13🔽️| +0.044⬆️| +0.034⬆️|
| 3A4 | -0.005⬇️| +0.00⬆️| +0.001⬆️| +0.005⬆️|

Solid gains everywhere except for CYP2D6.

The test set contains ~30% entries without a complete stereo specification. These were
handled in a straightforward fashion: the prediction for a particular entry is some 
kind of average of the predictions made for all corresponding enantiomers.

### Multitasking

### Auxiliary targets

### Test predictions aggregation

The model is processing a particular structure while a single test set entry may correspond
[to dozens of structures](#data-augmentation-aggregation-sampling). I was doing it as 
follows:
- weighted average (Boltzmann) of per-conformer predictions -> model's prediction for a molecule
- median of predictions made for a particular molecule by the whole ensemble of model replicas -> prediction for a molecule
- harmonic average (taken for the predicted concentrations in mol/l) of the per-enantiomer (molecule) predictions -> prediction for a dataset entry 

Each of these tricks is expected to improve the LB MAE by 0.000-0.002 compared to the plain mean.   

## Phase 1 (17 Aug 2026 - 24 Sep 2026)

The primary Phase 1 objectives are to check whether the whole setup from 
the massive data generation to test predictions can be deployed in ~1 month and 
to evaluate the baseline performance. The public LB ranks may sometimes be fishy and 
are not suitable for baseline assessment. Luckily, the provided official baselines 
are hard enough for performance comparison (e.g. the TabICL-baseline uses one of 
winning approaches from the previous blind challenge, while the CheMeleon-baseline
employs one of the de facto standard models in the field). Hence, for the Phase 1, 
the data is restricted to what is provided by the challenge, calibration and 
exhaustive fine-tuning are also left for the next phase. 

### Stage 0 (17 Aug - 25 Aug): Preliminary 
Challenge data analysis and processing, quantum-chemical data preparation, first ever attempt to fit a model prototype (arbitrary setup).

**Model**: 2-layer RhNet2TBequi

**Data**: pIC50 direct and TDI inhibition for 3767 molecules (challenge data with fully specified stereochemistry only)

**External data**: None

**Endpoints**: 'CYP1A2_pIC50_direct_inhibition', 'CYP2C9_pIC50_direct_inhibition', 'CYP2D6_pIC50_direct_inhibition', 'CYP3A4_pIC50_direct_inhibition'

**Regime**: 1 multitask model for 4 endpoints

**Regularization**: None

| Endpoint | local MAE | LB MAE | LB rank | XGB-baseline | LGBM-baseline | CheMeleon-baseline | TabICL-baseline |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1A2 | 0.56 | 1.00 | 28 | ✅ | ✅ | ✅ | ⛔️ |
| 2C9 | 0.42 | 0.59 | 44 | ⛔️ | ✅ | ⛔️ | ⛔️ |
| 2D6 | 0.65 | 1.82 | 56 | ⛔️ | ⛔️ | ⛔️ | ⛔️ |
| 3A4 | 0.56 | 0.67 | 42 | ✅ | ⛔️ | ⛔️ | ⛔️ |

### Stage 1 (26 Aug - 31 Aug): Base model setup 
A quick overview of options and choosing model architecture details e.g. depth, pooling variants, etc.

**Model**: 4-layer RhNet2TBequi

**Data**: pIC50 direct inhibition for 3767 molecules (challenge data with fully specified stereochemistry only)

**External data**: None

**Endpoints**: 'CYP1A2_pIC50_direct_inhibition', 'CYP2C9_pIC50_direct_inhibition', 'CYP2D6_pIC50_direct_inhibition', 'CYP3A4_pIC50_direct_inhibition', 'CYP2D6_pIC50_TDI_condition', 'CYP3A4_pIC50_TDI_condition'

**Regime**: 1 multitask model for 6 endpoints

**Regularization**: None


| Endpoint | local MAE | LB MAE | LB rank | XGB-baseline | LGBM-baseline | CheMeleon-baseline | TabICL-baseline |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1A2 | 0.55 | 0.96 | 37 | ✅ | ✅ | ✅ | ⛔️ |
| 2C9 | 0.42 | 0.57 | 55 | ✅ | ✅ | ✅ | ⛔️ |
| 2D6 | 0.65 | 1.76 | 68 | ⛔️ | ⛔️ | ⛔️ | ⛔️ |
| 3A4 | 0.55 | 0.61 | 53 | ✅ | ✅ | ⛔️ | ⛔️ |

### Stage 2 (1 Sep - 4 Sep): Pretrained encoder  

In the previous **OpenADMET PXR Blind Challenge**, I experimented a bit with pretraining, but the results were, to put it mildly, confusing. 
Here, I explored two alternatives: pretraining with synthetic data (CheMeleon-style, RDKit and quantum-chemical descriptors) and pretraining with some arbitrary abundant ADME data (microsomal intrinsic clearance from ChEMBL in this case). The effect of both these options on the validation metrics turned out to be negligible. However, ADME-pretraining allowed to fit my models 5x faster, while the use of synthetic data in pretraining resulted in 2x... slower convergence. Well, there you go.

### Stage 3 (5 Sep - 24 Sep): Baseline 
Optimizing performance on the full challenge training data. 

**Model**: 4-layer RhNet2TBequi

**Data**: pIC50 direct and TDI inhibition, single concentration data  (challenge training set only)

**External data**: None

**Endpoints**: 'CYP1A2_pIC50_direct_inhibition', 'CYP2C9_pIC50_direct_inhibition', 'CYP2D6_pIC50_direct_inhibition', 'CYP3A4_pIC50_direct_inhibition', 'CYP2D6_pIC50_TDI_condition', 'CYP3A4_pIC50_TDI_condition', 'CYP1A2_log2fc', 'CYP2C9_log2fc', 'CYP2D6_log2fc', 'CYP3A4_log2fc', 

**Regime**: 3 multitask models

**Regularization**: weight decay, normal noise

| Endpoint | local MAE | LB MAE | LB rank | XGB-baseline | LGBM-baseline | CheMeleon-baseline | TabICL-baseline |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1A2 | 0.50 | 0.88 | 82 | ✅ | ✅ | ✅ | ⛔️ |
| 2C9 | 0.39 | 0.53 | 122 | ✅ | ✅ | ✅ | ⛔️ |
| 2D6 | 0.62 | 1.59 | 103 | ✅ | ✅ | ✅ | ⛔️ |
| 3A4 | 0.49 | 0.53 | 78 | ✅ | ✅ | ✅ | ✅ |

## Phase 2 (25 Sep 2026 - 3 Nov 2026)

In this Phase, I'm mostly experimenting with two options: enhancing the GNN-encoder with external data (Track A) and squeezing more from the molecular embeddings produced by the already trained GNNs (Track B). The first direction requires calculating new data on ~40.000 new molecules. Luckily, the quantum-chemical data generation can be done in parallel with experimentation using the molecular embeddings produced by the models fitted in the previous Phase.

### Track B: Fixed embeddings

#### Stage B.1: Linear

Slight improvement (stability and fitting regime mostly) of the linear predictors on the embeddings produced by the best Phase I models.

| Endpoint | local MAE | LB MAE | LB rank | XGB-baseline | LGBM-baseline | CheMeleon-baseline | TabICL-baseline |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1A2 | n/a | 0.86 | 88 | ✅ | ✅ | ✅ | ⛔️ |
| 2C9 | n/a | 0.51 | 124 | ✅ | ✅ | ✅ | ⛔️ |
| 2D6 | n/a | 1.55 | 110 | ✅ | ✅ | ✅ | ✅ |
| 3A4 | n/a | 0.51 | 75 | ✅ | ✅ | ✅ | ✅ | 
