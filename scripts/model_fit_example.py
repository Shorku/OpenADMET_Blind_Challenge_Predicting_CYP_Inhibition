import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from rhnet2tbequi import RhNet2TBequi, classify_params
from rhnet2tbequi import PackedDataset, merge_targets, ResampleSampler
from rhnet2tbequi import make_loader, make_eval_loader
from rhnet2tbequi import WeightedMultitaskLoss, EMAModel
from rhnet2tbequi import rhnet2tbequi_train, rhnet2tbequi_eval
from rhnet2tbequi import rhnet2tbequi_embed


###############################################################################
# Column groups
###############################################################################
target_cols = ['CYP2C9pIC50', 'CYP3A4pIC50', 'CYP1A2pIC50',
               'CYP3A4pIC50TDI', 'CYP1A2log10FC', 'CYP2C9log10FC', 'CYP3A4log10FC',
               'CYP1A2pAC50TX', 'CYP2C9pAC50TX', 'CYP2C19pAC50TX', 'CYP3A4pAC50TX',
               'CYP1A2pAC50PC', 'CYP2C9pAC50PC', 'CYP2C19pAC50PC', 'CYP3A4pAC50PC',
               'CYP1A2pIC50CE', 'CYP2C9pIC50CE', 'CYP3A4pIC50CE',
               'logV', 'PTB_dipole_au'
               ]
eval_cols = ['CYP2C9pIC50', 'CYP3A4pIC50',
             'CYP1A2pIC50', 'CYP3A4pIC50TDI', 'CYP1A2log10FC', 'CYP2C9log10FC',
             'CYP3A4log10FC']
train_agg_on = 'mixinchikey'
val_agg_on = 'inchikey'


###############################################################################
# Pre-built monoliths (identity-only): produced beforehand by pack_densities.py,
# e.g.  python3 pack_densities.py --src train_example \
#           --index tables/index_train_example.csv --out dataset/train_example
# This script only READS them; the aggregation key is joined in below via attach.
###############################################################################
train_prefix = '/workspace/cyp_extended_full_1a2_2c9_3a4_tauE0-20kJ_rot0-2'
test_prefix = '/workspace/cyp_test_tauE0kJ_rot0'


###############################################################################
# Read training tabular data
###############################################################################
train_targets_csv_path = Path('/workspace/cyp_1a2_2c6_3a4_extended_full.csv')
train_targets_csv_path2 = Path('/workspace/xtb_properties2.csv')
train_targets_csv_path3 = Path('/workspace/ptb_properties2.csv')
train_index_csv_path = Path('/workspace/cyp_extended_full_1a2_2c9_3a4_tauE0-20kJ_rot0-2.csv')
train_df = pd.read_csv(train_targets_csv_path)
train_df2 = pd.read_csv(train_targets_csv_path2)[[train_agg_on, 'logV']]
train_df3 = pd.read_csv(train_targets_csv_path3)[[train_agg_on, 'PTB_dipole_au']]
train_df = train_df\
    .merge(train_df2, on=train_agg_on, how='outer')\
    .merge(train_df3, on=train_agg_on, how='outer')
train_df = train_df[[train_agg_on] + target_cols]
train_index_df = pd.read_csv(train_index_csv_path, dtype={'structure_id': str})

###############################################################################
# Read test tabular data
###############################################################################
test_index_csv_path = Path('/workspace/cyp_test_tauE0kJ_rot0.csv')
test_index_df = pd.read_csv(test_index_csv_path, dtype={'structure_id': str})

###############################################################################
# Scale targets
###############################################################################
with open('/workspace/scalings_v9.json', 'r') as jf:
    scl_dict = json.load(jf)

for feat, scalings in scl_dict.items():
    if feat in target_cols:
        train_df[feat] = (train_df[feat] - scalings['shift']) / scalings['scale']

###############################################################################
# Open the pre-built monoliths, attach the aggregation key, bind targets
###############################################################################
# Identity keys present in every packed monolith; used to join the experiment
# index table (and hence the agg_on aggregation key) onto the dataset.
graph_keys = ['structure_id', 'conf_id', 'rot_id']

# Hold the whole payload resident; __getitem__ returns the row's numid, batches
# are vectorized index_selects handled by the Collator (see rhnet2tbequi sec. 7).
train_data_full = PackedDataset.open(train_prefix, mode='ram')
test_data = PackedDataset.open(test_prefix, mode='ram')

# The monolith is identity-only, so join the experiment index table on the graph
# keys to bring in agg_on (inchikey) for grouping/sampling. attach preserves
# numid order and asserts a 1:1 cover of the packed rows.
train_data_full.attach(train_index_df, on=graph_keys)
test_data.attach(test_index_df, on=graph_keys)

# Late-bind targets from the (already scaled) per-inchikey tables, joined on the
# now-attached agg_on column. Each (table, key) spec supplies whichever targets
# it carries; per-graph targets (e.g. energies) would instead use
# bind_targets(ds, targets, (graph_table, graph_keys)).
merge_targets(train_data_full, target_cols, (train_df, train_agg_on))

# Restrict training to structures whose agg_id carries a sampling scale, i.e. the
# inner-merge semantics of fit_example.py (structures with at least one target).
# The full dataset is kept (train_data_full) so importance weights can be indexed
# by GLOBAL numid; the sampler runs over the subset.
train_keep = train_data_full.index[train_agg_on].isin(train_df[train_agg_on]).to_numpy()
train_data = train_data_full.subset(train_keep)


################################################################################
# Torch setup
################################################################################
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


################################################################################
# Training settings
################################################################################
exp_id = Path(__file__).stem
work_dir = f'./{exp_id}'
os.makedirs(work_dir, exist_ok=True)
parent_dir = os.getcwd()
os.chdir(work_dir)

# =============================================================================
# Model parameters
# =============================================================================
backend = 'cueq'
graph_depth = 4
atom_embed = '64x0e+32x1e+16x2e'
atom_hidden = [atom_embed for _ in range(graph_depth)]
link_hidden = '32x0e+16x1e+8x2e'
edge_hidden = '32x0e+16x1e+8x2e'
msg_hidden = '32x0e+16x1e+8x2e'
ovlp_hidden = '32x0e+16x1e+8x2e'

global_node = True
global_irreps = atom_embed
global_gather = 'attn'
global_inject = 'gate'           # 'residual' | 'gate'
overlap_before_inject = False    # only with global_inject='residual'

depthwise = False
update_overlap = True
residual = True
self_interaction = True
receiver_interaction = 'none'    # 'none' | 'film' | 'tp'
explicit_populations = True
link_carries_overlap = True
append_highl_norms = True
compact_inputs = True
per_l_norm = False
pool = ("mean", "attn", "gate")

msg_dropout = 0.05
upd_dropout = 0.0
ovlp_dropout = 0.05

# =============================================================================
# Fit parameters
# =============================================================================
weight_decay = {
    'heads_dense_weight':       3.0,
    'heads_dense_bias':         0.0,
    'graph_dense_weight':       3.0,
    'graph_dense_bias':         0.0,
    'graph_equivariant_weight': 4.0,
    'graph_gain':               0.0}
noise_scales = torch.tensor([0.05, 0.0, 0.06,
                             0.1, 0.1, 0.1, 0.1,
                             0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                             0.05, 0.05], dtype=torch.float32)

weights = [1.0, 1.0, 1.0,
           1.0, 1.0, 1.0, 1.0,
           1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
           0.05, 0.05]
sampling_mode = 'oversample'
train_sample_size = 1
train_batch_size = 64
eval_batch_size = 64

base_loss = 'huber'
huber_delta = 0.4
clip_value = 1.0
clip_norm = None
betas = (0.9, 0.98)
eps = 0.000001

lr = {
    'heads_dense_weight':       0.0004,
    'heads_dense_bias':         0.0004,
    'graph_dense_weight':       0.0001,
    'graph_dense_bias':         0.0001,
    'graph_equivariant_weight': 0.0001,
    'graph_gain':               0.0001}

ema = True
ema_decay = 0.999
ema_warmup = True

# =============================================================================
# Schedule parameters
# =============================================================================
ref_epochs = 100
warmup_steps = 200
decay_share = 0.15
steps_per_epoch = len(train_df) // train_batch_size
epochs = ref_epochs + int(decay_share * ref_epochs)
total_steps  = epochs * steps_per_epoch
decay_steps  = steps_per_epoch * int(decay_share * ref_epochs)
main_fit_steps = total_steps - warmup_steps - decay_steps
print('Schedule:')
print(f'Steps per epoch: {steps_per_epoch}')
print(f'Total steps:     {total_steps}')
print(f'Warmup steps:    {warmup_steps}')
print(f'Main fit steps:  {main_fit_steps}')
print(f'Decay steps:     {decay_steps}')

# =============================================================================
# Training runs
# =============================================================================
seeds = [472080, 447318, 825936]
encoders = [1,3,5]
for enc in encoders:
    for seed in seeds:
        run_id = f'{exp_id}_encoder{enc}_seed{seed}'
        torch.manual_seed(seed)
        np.random.seed(seed)
        # =============================================================================
        # Model and optimizer
        # =============================================================================
        model = RhNet2TBequi(
            atom_hidden=atom_hidden,
            link_hidden=link_hidden,
            edge_hidden=edge_hidden,
            msg_hidden=msg_hidden,
            ovlp_hidden=ovlp_hidden,
            atom_embed=atom_embed,
            targets=target_cols,
            backend=backend,
            depthwise=depthwise,
            update_overlap=update_overlap,
            residual=residual,
            self_interaction=self_interaction,
            receiver_interaction=receiver_interaction,
            msg_dropout=msg_dropout,
            upd_dropout=upd_dropout,
            ovlp_dropout=ovlp_dropout,
            explicit_populations=explicit_populations,
            link_carries_overlap=link_carries_overlap,
            append_highl_norms=append_highl_norms,
            compact_inputs=compact_inputs,
            per_l_norm=per_l_norm,
            pool=pool,
            global_node=global_node,
            global_irreps=global_irreps,
            global_gather=global_gather,
            global_inject=global_inject,
            overlap_before_inject=overlap_before_inject).to(device)

        checkpoint = torch.load(
            f'/workspace/encoder{enc}_1a2_2c9_3a4.pt',
            weights_only=False, map_location=device)
        model.load_state_dict(checkpoint['model'], strict=False)

        ema_model = EMAModel(model, decay=ema_decay, warmup=ema_warmup).to(device) if ema else None

        task_weights = torch.tensor(weights, dtype=torch.float32).to(device)
        loss_fn = WeightedMultitaskLoss(base_loss=base_loss, huber_delta=huber_delta).to(device)

        param_roles, param_groups = classify_params(model)
        optimizer = torch.optim.AdamW([{'params': params,
                                        'weight_decay': weight_decay[group],
                                        'lr': lr[group],
                                        'betas': betas,
                                        'eps': eps}
                                       for group, params in param_groups.items()])

        scheduler1 = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
        scheduler2 = torch.optim.lr_scheduler.ConstantLR(
            optimizer, factor=1.0, total_iters=main_fit_steps)
        scheduler3 = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=0.0, total_iters=decay_steps)
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[scheduler1, scheduler2, scheduler3],
            milestones=[warmup_steps, warmup_steps + main_fit_steps])

        train_sampler = ResampleSampler(
            train_data, key=train_agg_on, n_per=train_sample_size,
            batch_size=train_batch_size, shuffle=True, drop_last=True, seed=seed)
        train_loader = make_loader(train_data, train_sampler, with_y=True, pin_memory=False)

        # =============================================================================
        # Training loop
        # =============================================================================
        print()
        print('#######################################################################')
        print(run_id)
        print('#######################################################################')
        hist = []
        for epoch in range(1, epochs + 1):
            start = time.time()
            train_loss = rhnet2tbequi_train(model, train_loader, optimizer, scheduler,
                                            loss_fn, device, task_weights,
                                            ema_model=ema_model, monitor=None,
                                            clip_value=clip_value, clip_norm=clip_norm,
                                            noise_scales=noise_scales,
                                            print_batch_count=(epoch == 1))
            train_time = time.time() - start
            eval_results = ''
            mae = [np.nan for _ in eval_cols]
            hist.append([epoch, train_loss] + mae)
            pd.DataFrame(hist, columns=['epoch', 'train_loss'] + eval_cols).to_csv(
                f'{run_id}_hist.zip', index=False)
            eval_time = time.time() - start - train_time
            print(f'Epoch {epoch:03d} | Train: {train_loss:.6f} | {train_time:5.1f} s | '
                  f'{eval_time:5.1f} s | ')

        ################################################################################
        # Final dumps
        ################################################################################
        test_loader = make_eval_loader(test_data, batch_size=eval_batch_size, with_y=False)
        namespace_test = f'{run_id}_ep{epoch}_test'
        save_targets = eval_cols
        scored = ema_model if ema else model
        test_results, test_mae = rhnet2tbequi_eval(
            scored, test_loader, device, target_cols, eval_cols, scl_dict,
            namespace_test, save_targets=save_targets, with_y=False)
        checkpoint = {'epoch': epoch,
                      'model': model.state_dict(),
                      'optimizer': optimizer.state_dict(),
                      'scheduler': scheduler.state_dict(),
                      'hist': hist,
                      'loss_fn': loss_fn.state_dict()}
        if ema:
            checkpoint['ema_model'] = ema_model.state_dict()
        torch.save(checkpoint, f'{run_id}_ep{epoch}_checkpoint.pt')
        del model, ema_model, optimizer, scheduler, scheduler1, scheduler2, scheduler3, train_loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# =============================================================================
# Do embeddings
# =============================================================================
train_df = train_df[(~train_df['CYP1A2pIC50'].isna()) |
                    (~train_df['CYP2C9pIC50'].isna()) |
                    (~train_df['CYP3A4pIC50'].isna())]
train_keep = train_data_full.index[train_agg_on].isin(train_df[train_agg_on]).to_numpy()
train_data = train_data_full.subset(train_keep)
for enc in encoders:
    for seed in seeds:
        run_id = f'{exp_id}_encoder{enc}_seed{seed}'
        model = RhNet2TBequi(
            atom_hidden=atom_hidden,
            link_hidden=link_hidden,
            edge_hidden=edge_hidden,
            msg_hidden=msg_hidden,
            ovlp_hidden=ovlp_hidden,
            atom_embed=atom_embed,
            targets=target_cols,
            backend=backend,
            depthwise=depthwise,
            update_overlap=update_overlap,
            residual=residual,
            self_interaction=self_interaction,
            receiver_interaction=receiver_interaction,
            msg_dropout=msg_dropout,
            upd_dropout=upd_dropout,
            ovlp_dropout=ovlp_dropout,
            explicit_populations=explicit_populations,
            link_carries_overlap=link_carries_overlap,
            append_highl_norms=append_highl_norms,
            compact_inputs=compact_inputs,
            per_l_norm=per_l_norm,
            pool=pool,
            global_node=global_node,
            global_irreps=global_irreps,
            global_gather=global_gather,
            global_inject=global_inject,
            overlap_before_inject=overlap_before_inject,
            embed_output=True).to(device)

        checkpoint = torch.load(
            f'{run_id}_ep{epoch}_checkpoint.pt',
            weights_only=False, map_location=device)
        model.load_state_dict(checkpoint['model'], strict=False)
        ema_model = EMAModel(model, decay=ema_decay, warmup=ema_warmup).to(device)
        ema_model.load_state_dict(checkpoint['ema_model'], strict=False)

        train_loader = make_eval_loader(train_data, batch_size=eval_batch_size, with_y=False)
        namespace = f'{run_id}_emamodel'
        rhnet2tbequi_embed(ema_model, train_loader, device, namespace)

        test_loader = make_eval_loader(test_data, batch_size=eval_batch_size, with_y=False)
        namespace = f'{run_id}_emamodel_test'
        rhnet2tbequi_embed(ema_model, test_loader, device, namespace)

        del model, ema_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


os.chdir(parent_dir)
os.system(f'tar -cf {exp_id}.tar {exp_id}')
