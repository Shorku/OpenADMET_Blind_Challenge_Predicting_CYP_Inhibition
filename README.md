## OpenADMET Blind Challenge: Predicting CYP Inhibition

The repository contains a description of the solution to the 
regression track of the [**OpenADMET Blind Challenge: Predicting CYP Inhibition**]((https://huggingface.co/spaces/openadmet/cyp-challenge)) 
by the user `Shorku`.

Author: Oleg Gromov

Date: Aug 2026 - Nov 2026

### TL;DR

**Objective**:  to field-test learning from the full electronic
structure of a molecule on the contemporary data in real competition
conditions to obtain a fair performance and usability assessment. Hence, 
the solution is constrained to a single model architecture.
The focus is more on development and research than on benchmaxxing.

**Results**: _on the challenge data_, the single architecture solution without 
calibration beats all four official baselines (XGB, LGBM, CheMeleon, TabICL) 
on CYP3A4 endpoint and three out of four baselines on the other endpoints. 
With _the external data_, it landed mid-table overall and 30th on CYP3A4 
(no calibration for this one btw). Given the objectives, constraints, and a global
shift towards a massive use of agents, I'm considering the model trials successful.  



## Approach

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

## General plan outline
- 17 Aug 2026 - 24 Sep 2026 (challenge start - intermediate LB): Baseline performance evaluation
    - Fitting and other technical settings, debug, etc.
    - Basic architecture features selection (e.g. depth, pooling options etc.)
    - Only the RhNet2TBequi model, only the challenge data, no calibration
- 25 Sep 2026 - 3 Nov 2026 (LB freeze - challenge ends): external data, fixed embeddings.

## Experiments design 

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

### Endpoints compatibility

By the end of the challenge I was working with two models: one for CYP1A2, CYP2C9, and CYP3A4, and
the other one for CYP2D6. CYP2C9 and CYP3A4 are somewhat aligned (~0.2), and combining them in a
single model improves performance for the both endpoints. CYP1A2 is mostly orthogonal (~0.0) to 
CYP2C9 and CYP3A4 with marginal mutual influence on metrics. It was combined with CYP2C9 and CYP3A4
only to minimize the number of models to train. CYP2D6 was deliberately separated due to both the
gradients conflict and metrics deterioration for a joint model. 

### Auxiliary targets

- Challenge data: TDI inhibition, log2fc
- Synthetic: average dipole, average volume
- External data: AC50 from PubChem and Tox21, IC50 from ChEMBL

Various Emax labels from the challenge data had literally no impact and were not used. The situation
with the synthetic quantum-chemical descriptors is way more interesting. These are _per se_ 
harmful (no new information, too easy to predict), e.g. inclusion of the calculated dipole results in 
something like this: 

| Endpoint | MAE      | R²       | Spearman  | Kendall   |
| --- |----------|----------|-----------|-----------|
| 2C9 | +0.007🔼️| -0.01🔽️| -0.014🔽️| -0.012🔽️|
| 3A4 | +0.018🔼️| -0.04🔽️| -0.034🔽️| -0.030🔽️|

Disastrous, actually. However, the reason I'm using average (over conformers ensemble) dipole 
and average volume is that these allow to include the test set molecules into the training set!
And if used for this purpose, the gain is sometimes significant:

| Endpoint | MAE      | R²      | Spearman | Kendall  |
| --- |----------|---------|----------|----------|
| 1A2 | +0.001🔼️ | -0.00🔽️ | +0.003⬆️ | +0.003⬆️ |
| 2C9 | +0.004🔼️ | -0.01🔽️ | -0.008🔽️ | -0.008🔽️ |
| 2D6 | -0.018⬇️ | +0.03⬆️ | +0.014⬆️ | +0.007⬆️ |
| 3A4 | -0.020⬇️ | +0.02⬆️ | +0.008⬆️ | +0.009⬆️ |

External data usage as auxiliary targets is just the safest option to include them without the 
risks of performance deterioration. 

### Test predictions aggregation

The model is processing a particular structure while a single test set entry may correspond
[to dozens of structures](#data-augmentation-aggregation-sampling). So the raw predictions should
be somehow aggregated to produce the final test predictions. I was doing it as 
follows:
- weighted average (Boltzmann) of per-conformer predictions -> model's prediction for a molecule
- median of predictions made for a particular molecule by the whole ensemble of model replicas -> prediction for a molecule
- only the lowest energy tautomers i.e. single tautomeric form for every enantiomer
- harmonic average (taken for the predicted concentrations in mol/l) of the per-enantiomer (molecule) predictions -> prediction for a dataset entry 

Each of these tricks is expected to improve the LB MAE by 0.000-0.002 compared to the plain mean.

### Model stability

GNNs are known for their instability and RhNet2 is no exclusion. Though, modelling setup optimization-wise,
a fixed initialization (e.g. a fixed pretrained encoder) and a fixed seed are not enough when
it comes to experiments on e.g. expanding the dataset. To reduce the variance of this kind I 
stacked several model replicas, initially 5 pretrained encoders x 3-4 seeds, and ended up with
3 pretrained encoders x 3 seeds. 

Performance-wise, averaging over an ensemble of 15 models (5 pretrains x 3 seeds) consistently 
improved both local and LB MAE by 0.01-0.02 compared to a single model. Removing a couple of
suspiciously good and bad pretrains further improved MAEs by another 0.01-0.02. Notably, 
the different sets of endpoints may prefer slightly different sets of pretrains.

From the previous occasional experiments it seems that preparation of another 16-25 pretrained 
encoders, picking a dozen of the best performing ones and using them to fit a pack of
~25 models may further improve MAE by at least 0.02 or more. One fit takes ~3h with RTX 5070 Ti, RTX 5080 
or similar, hence the whole run is one week of compute on a single GPU. Would be worth trying if 
the aim was just the score optimization. I obviously didn't do it because of different objectives.  

## Phase 1: Baseline
 
*17 Aug 2026 - 24 Sep 2026 (challenge start - intermediate LB)*

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

*✅ : the baseline is beaten; ⛔️: the baseline holds*

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

*✅ : the baseline is beaten; ⛔️: the baseline holds*

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

*✅ : the baseline is beaten; ⛔️: the baseline holds*

## Phase 2: External data and fixed embeddings
 
*25 Sep 2026 - 3 Nov 2026 (LB freeze - challenge ends)*

In this Phase, I'm mostly experimenting with two options: 
squeezing more from the molecular embeddings produced by the already trained GNNs and 
enhancing the GNN-encoder with external data.

### Fixed embeddings: challenge data

#### Linear

Slight improvement (stability and fitting regime mostly) of the linear predictors on 
the embeddings produced by the best Phase I models.

| Endpoint | local MAE | LB MAE | LB rank | XGB-baseline | LGBM-baseline | CheMeleon-baseline | TabICL-baseline |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1A2 | n/a | 0.86 | 88 | ✅ | ✅ | ✅ | ⛔️ |
| 2C9 | n/a | 0.51 | 124 | ✅ | ✅ | ✅ | ⛔️ |
| 2D6 | n/a | 1.55 | 110 | ✅ | ✅ | ✅ | ✅ |
| 3A4 | n/a | 0.51 | 75 | ✅ | ✅ | ✅ | ✅ | 

*✅ : the baseline is beaten; ⛔️: the baseline holds*

#### TabICL
``TabICLRegressor`` on top of the embeddings produced by the best Phase I models. 
Something similar to the TabICL baseline but with RhNet2TBequi instead of Chemeleon.

| Endpoint | local MAE | LB MAE | LB rank | XGB-baseline | LGBM-baseline | CheMeleon-baseline | TabICL-baseline |
| --- | --- |--------|---------| --- | --- | --- | --- |
| 1A2 | n/a | 0.88   | 93      | ✅ | ✅ | ✅ | ⛔️ |
| 2C9 | n/a | 0.49   | 103     | ✅ | ✅ | ✅ | ⛔️ |
| 2D6 | n/a | 1.58   | 119     | ✅ | ✅ | ✅ | ⛔️ |
| 3A4 | n/a | 0.50   | 63      | ✅ | ✅ | ✅ | ✅ | 

*✅ : the baseline is beaten; ⛔️: the baseline holds*

Compared to the linear predictor:

| Endpoint | MAE       | R²      | Spearman | Kendall  |
| --- |-----------|---------|---------|----------|
| 1A2 | +0.017🔼️| -0.02🔽️| -0.001🔽️| -0.001🔽️|
| 2C9 | -0.019⬇️| +0.04⬆️| +0.013⬆️| +0.017⬆️|
| 2D6 | +0.037🔼️| -0.07🔽️| +0.032⬆️| +0.031⬆️|
| 3A4 | -0.008⬇️| +0.01⬆️| +0.003⬆️| +0.004⬆️|

It seems like TabICL improves things that are already pretty good and makes everything 
even weirder where something is already going wrong.

### External data: the cavalry has arrived

External data sources:
- Tox21 CYP screen: ``AC50`` filtered by purity (D,F,I,Z and derived were excluded)
- PubChem AID 1851, the Veith qHTS panel: ``Fit_LogAC50``, single points and poor fits excluded
- ChEMBL: curated

| Source  | CYP1A2 | CYP2C9 | CYP2C19 | CYP2D6 | CYP3A4 | New mol entries | New structures |
|---------|--------|--------|---------|--------|--------|-----------------|----------------|
| Tox21   | 2866   | 2555   | 2724    | 1571   | 1319   | 3779            | 5162           |
| PubChem | 7893   | 5667   | 6543    | 3214   | 7708   | 12321           | 17264          |
| ChEMBL  | 521    | 1114   | -       | 1101   | 2199   | 3644            | 4247           |
| Total   |        |        |         |        |        | 19507           | 26418          |

Each endpoint from each source was used as a separate auxiliary endpoint. 

#### Linear

_Note: the performance of the models trained using the external data will not be compared 
to the baselines_

Difference of performances of linear predictors on the embeddings produced by the models 
trained on the extended dataset and the challenge dataset only:  

| Endpoint | MAE       | R²       | Spearman | Kendall  |
| --- |-----------|----------|----------|----------|
| 1A2 | +0.002🔼️ | -0.00🔽️ | +0.008⬆️| +0.006⬆️|
| 2C9 | -0.043⬇️  | +0.05⬆️  | +0.034⬆️ | +0.044⬆️ |
| 2D6 | -0.059⬇️  | +0.13⬆️  | +0.048⬆️ | +0.033⬆️ |
| 3A4 | -0.056⬇️  | +0.01⬆️  | +0.023⬆️ | +0.032⬆️ |

### The final results

#### Final setup

**Model**: 4-layer RhNet2TBequi

**Data**: pIC50 direct and TDI inhibition, single concentration data

**External data**: Tox21, PubChem, ChEMBL

**Regime**: 2 multitask models

**Regularization**: weight decay, equivariant dropout, normal noise


| Endpoint | Predictor       | Shares encoder with | Unspec. stereo entries | Tautomers | Calibration |
|----------|-----------------|---------------------|------------------------|-----------|-------------|
| 1A2      | Linear + TabICL¹| 2C9, 3A4            | Yes                    | Yes       | No¹         |
| 2C9      | TabICL          | 1A2, 3A4            | Yes                    | Yes       | No          |
| 2D6      | Linear          |                     | No                     | No        | Yes         |
| 3A4      | TabICL          | 1A2, 2C9            | Yes                    | Yes       | No          |

¹ Learned linear combination 0.965 * Linear + 0.017 TabICL, which is essentially a slight downscaling  

#### Public LB

| Endpoint | Rank | ST-RAE |MAE | R²      | Spearman | Kendall  |
| --- |------------|-------------|------------|---------|---------|----------|
| 1A2 | 96 | 0.570 | 0.827 | 0.45 | 0.780 | 0.582|
| 2C9 | 92 | 0.430 | 0.463 | 0.63 | 0.805 | 0.620|
| 2D6 | 116 | 0.762 | 0.968 | 0.34 | 0.443 | 0.301|
| 3A4 | 23 | 0.380 | 0.447 | 0.75 | 0.861 | 0.687|

#### Final LB

| Endpoint | Rank | ST-RAE |MAE | R²      | Spearman | Kendall  |
| --- |------|-------------|------------|---------|---------|----------|
| 1A2 | 55   | 0.5282|0.8112|0.4676|0.7835|0.5876|
| 2C9 | 47   | 0.4178|0.5104|0.6342|0.8267|0.6447|
| 2D6 | 61   | 0.8359|1.0494|0.3064|0.3996|0.2684|
| 3A4 | 30   | 0.3732|0.456|0.7613|0.8704|0.6933|

The ranking improvement compared to the public LB here is deceptive as a large number
of entries were purged for various reasons. Notably, ST-RAE metric on the total test
set compared to the public LB improved by 0.01-0.04 everywhere except for CYP2D6, which
is the only heavily calibrated endpoint. 

## Conclusions

### What I did wrong

I can, probably, recall only a couple of things I strategically missed on:
- the choice to minimize MAE and ST-RAE
of the CYP2D6 model, while retrospectively it is clear that Spearman and Kendall optimization
plus calibration is the winning approach.
- I didn't try multitasking with TabICL, which could have been a relatively cheap but 
highly promising experiment

P.S I totally forgot about the classification track! With the already prepared within 
regression track embeddings it was a matter of one prompt to try it...

### More unexplored options

- more auxiliary targets like pKaH, LogD, etc.
- learning 2D6 inhibition from electronic structures of the protonated structures
- stack more model replicas
- use less pruning while converting densities to graphs to capture the intramolecular non-bonding interactions
- dedicated models for every endpoint (maintaining endpoints grouping though)
- more parameters tuning, e.g. most of the auxiliary targets just got unity weights without further optimization

What unites most of these options is that they are relatively straightforward and just require more
compute for experimentation. 

### So, where We Stand?

Basically, I'm almost done with this particular research project dedicated to the RhNet2 model development.
The whole thing now:
- uses cheap input data, i.e. xTB calculation results, so that a dataset for e.g. 40,000 molecules may
be generated in days on regular machines
- requires reasonable amount of memory: ~10-15GB VRAM for a 4-layer model, 30-35GB RAM for a dataset of
35,000-40,000 molecules
- full deployment including the data preparation fits the 1-2 month timeframe
- established (at least approximately) fitting and inference regimes
- competitive performance: this mono-architecture solution beats baselines on the challenge data, 
and scores mid-table when using the public external data 
- room for improvement: and it is straightforward, just add more compute.

Cheers =)
