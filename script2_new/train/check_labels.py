import sys
from pathlib import Path
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np
from importlib import util
from config import Config


def _load_create_dataloaders(dataset_processor_path: str):
    spec = util.spec_from_file_location("ds_proc", dataset_processor_path)
    mod = util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.create_dataloaders


def _as_list(x):
    if x is None:
        return None
    if isinstance(x, (list, tuple)):
        return list(x)
    try:
        return x.view(-1).cpu().tolist()
    except Exception:
        return [x]


def main():
    cfg = Config()

    feature_names = [
        'depth',
        'pollut_BODf',
        'pollut_NH4',
        'pollut_DO',
        'depth_residual',
        'pollut_BODf_residual',
        'pollut_NH4_residual',
        'pollut_DO_residual',
        'head',
        'volume',
        'lateral_inflow',
        'total_inflow',
        'head_residual',
        'volume_residual',
        'lateral_inflow_residual',
        'total_inflow_residual',
        'flooding',
        'flooding_residual',
    ]

    create_dataloaders = _load_create_dataloaders(cfg.dataset_processor_path)

    train_loader, val_loader, test_loader, dataset = create_dataloaders(
        node_timeseries_file=cfg.node_timeseries_file,
        adjacency_matrix_file=cfg.adjacency_matrix_file,
        node_list_file=cfg.node_list_file,
        defect_matrix_file=cfg.defect_matrix_file,
        batch_size=8,
        sequence_length=cfg.sequence_length,
        window_stride=cfg.window_stride,
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio,
        normalize=True,
        random_seed=cfg.random_seed,
        feature_names=feature_names,
        candidate_nodes_file=cfg.candidate_nodes_file,
        soft_label_sigma=cfg.soft_label_sigma,
        use_full_graph=cfg.train_with_full_graph,
        signal_threshold=getattr(cfg, "dataset_signal_threshold", 0.0),
        loc_ratio_clip=getattr(cfg, "dataset_loc_ratio_clip", None),
        label_mode=getattr(cfg, "dataset_label_mode", "auto"),
        overlap_threshold=getattr(cfg, "dataset_overlap_threshold", 0.5),
        always_on_force_target=getattr(cfg, "dataset_always_on_force_target", True),
        e_class_oversample_ratio=getattr(cfg, "dataset_e_class_oversample_ratio", 1),
    )

    if len(dataset) == 0:
        raise RuntimeError("dataset has 0 samples; check time gating window overlap and defect_matrix start_hour/duration_h")

    def check_one(loader_name, loader):
        batch = next(iter(loader))
        soft = batch['soft_label'].numpy()
        tgt = batch['target_node_idx'].numpy()
        has_def = batch.get('has_defect', np.ones_like(tgt))
        cand = batch['candidate_mask'].numpy()
        scen = _as_list(batch.get('scenario_id'))
        ws = _as_list(batch.get('window_start_time'))
        we = _as_list(batch.get('window_end_time'))

        print(f"\n[{loader_name}] batch_size={len(tgt)}")
        for i in range(min(8, len(tgt))):
            t = int(tgt[i])
            hd = int(has_def[i])
            if cand.ndim == 2:
                cand_ok = bool(cand[i, t])
            else:
                cand_ok = bool(cand[t])
            soft_sum = float(soft[i].sum())
            soft_argmax = int(soft[i].argmax())
            sc = scen[i] if scen is not None and i < len(scen) else 'N/A'
            w0 = ws[i] if ws is not None and i < len(ws) else 'N/A'
            w1 = we[i] if we is not None and i < len(we) else 'N/A'
            print(
                f"  i={i} scenario={sc} window=({w0}~{w1}) has_defect={hd} target_idx={t} cand_ok={cand_ok} soft_sum={soft_sum:.6f} soft_argmax={soft_argmax}"
            )

    check_one('train', train_loader)
    check_one('val', val_loader)
    check_one('test', test_loader)


if __name__ == '__main__':
    main()
