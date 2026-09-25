## OpenADMET Blind Challenge: Predicting CYP Inhibition

The repository contains a description of the solution to the 
**OpenADMET Blind Challenge: Predicting CYP Inhibition** by the user `Shorku`.

Author: Oleg Gromov

Date: Aug 2026 - Nov 2026

# TL;DR

The main objective is to field-test learning from the full electronic
structure of a molecule on the contemporary data in real competition
conditions to obtain a fair performance and usability assessment. Hence, 
the solution is (for now) constrained to a single model architecture.
The focus is more on development and research than on benchmaxxing.

# Approach

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
of the MLP components with SO(3)-equivariant machinery which dramatically reduces the 
memory footprint (no rotational augmentation needed anymore).

# General plan outline
- 17 Aug 2026 - 24 Sep 2026 (challenge start - intermediate LB): Baseline performance evaluation
    - Fitting and other technical settings, debug, etc.
    - Basic architecture features selection (e.g. depth, pooling options etc.)
    - Only the RhNet2TBequi model, only the challenge data
- 25 Sep 2026 - 3 Nov 2026 (LB freeze - challenge ends): Everything else including external data (for sure), ensembles (maybe), etc.
