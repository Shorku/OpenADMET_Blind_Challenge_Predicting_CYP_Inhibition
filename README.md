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

In the ongoing challenge, I'm evaluating RhNet2TBequi model prototype, which replaces most
of the MLP components with SO(3)-equivariant machinery and dramatically reduces the 
memory footprint (no rotational augmentation needed anymore).

### General plan outline
- 17 Aug 2026 - 24 Sep 2026 (challenge start - intermediate LB): Baseline performance evaluation
    - Fitting and other technical settings, debug, etc.
    - Basic architecture features selection (e.g. depth, pooling options etc.)
    - Only the RhNet2TBequi model, only the challenge data, no calibration
- 25 Sep 2026 - 3 Nov 2026 (LB freeze - challenge ends): Everything else including external data (for sure),
ensembles (maybe), and other mischief (I solemnly swear).

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
