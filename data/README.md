## OpenADMET Blind Challenge: Predicting PXR Induction

The repository contains the calculated data for the solution to the 
**OpenADMET Blind Challenge: Predicting CYP Inhibition** by the user `Shorku`.

Author: Oleg Gromov

Date: Aug 2026 - Nov 2026


## Contents

- [Indexes](#indexes)
- [Geometries](#geometries)
- [Molecular properties](#molecular-properties)
- [Methods](#methods)
- [Data stats](#data-stats)


### Indexes

challenge structures set: `index.csv`


**Columns:**

`structure_id`: internal molecule id, a BLAKE2b hash of molecules canonical SMILES

`SMILES`: molecules canonical SMILES

`InChIKey`: International Chemical Identifier hash


### Geometries

`geoms.zip`


**Format:**

Each `<structure_id>` folder contains geometries in `.xyz` XMOL format. 
Each `<structure_id>.<conf_id>.xyz` file contains Cartesian coordinates for the `conf_id`-th 
conformer of a molecule with internal id `structure_id`.


### Molecular properties

**Conformer-specific molecular properties**

file: `conf_props.zip`

`structure_id`: internal molecule id, a BLAKE2b hash of molecules canonical SMILES

`conf_id`: conformer id

`PTB_dipole_au`: molecular dipole from PTB calculation, a.u.

`PTB_homo_lumo_eV`: HOMO-LUMO gap from PTB calculation, eV

`xTB_E_Eh`: total energy from xTB caclulation, Eh

`vol`: molecular volume

`surf`: molecular surface

`Rg`: gyration radius

`Ellips1`:

`Ellips2`:

`Ellips3`:


### Methods

The general idea is outlined in the [JCTC paper](https://doi.org/10.1021/acs.jctc.5c00425). 

- Embed up to 50 conformers (RDKit)
- Geometry optimization (RDKit, only if MMFF94 parameters are available)
- Remove duplicates
- Geometry optimization (xTB)
- Remove high energy conformers (10 kJ/mole window)
- Remove duplicates
- Calculate molecular orbitals (xTB, PTB parametrization)

### Data stats

Molecules: 12,037

Conformers: 102,624


