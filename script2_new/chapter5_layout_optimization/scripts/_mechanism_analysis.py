import json, csv
import numpy as np
from pathlib import Path

reports = Path('E:/11.16/script2_new/outputs/reports')
layouts_dir = Path('E:/11.16/script2_new/chapter5_layout_optimization/outputs/layouts')
out = Path('E:/11.16/thesis_writing_repo/figures/ch5/source_data')
out.mkdir(parents=True, exist_ok=True)

methods_info = {
    'Degree': {
        'metrics': [
            'last_run_metrics_48h_control_ie420_normal20_raw_plus_residual_loc0p5_degree_N25_s7.json',
            'last_run_metrics_48h_control_ie420_normal20_raw_plus_residual_loc0p5_degree_N25_s42.json',
            'last_run_metrics_48h_control_ie420_normal20_raw_plus_residual_loc0p5_degree_N25_s123.json',
        ],
        'layout': 'degree/monitor_nodes_degree_N25.json',
    },
    'Betweenness': {
        'metrics': [
            'last_run_metrics_48h_control_ie420_normal20_raw_plus_residual_loc0p5_betweenness_N25_s7.json',
            'last_run_metrics_48h_control_ie420_normal20_raw_plus_residual_loc0p5_betweenness_N25_s42.json',
            'last_run_metrics_48h_control_ie420_normal20_raw_plus_residual_loc0p5_betweenness_N25_s123.json',
        ],
        'layout': 'betweenness/monitor_nodes_betweenness_N25.json',
    },
    'Cand-Obs': {
        'metrics': [
            'last_run_metrics_ch5_fixed_candidate_observability_N25_normal20_rawres_loc0p5_s7.json',
            'last_run_metrics_ch5_fixed_candidate_observability_N25_normal20_rawres_loc0p5_s42.json',
            'last_run_metrics_ch5_fixed_candidate_observability_N25_normal20_rawres_loc0p5_s123.json',
        ],
        'layout': 'candidate_observability/monitor_nodes_candidate_observability_N25.json',
    },
    'Two-stage v1': {
        'metrics': [
            'last_run_metrics_ch5_fixed_two_stage_balanced_layout_v1_N25_normal20_rawres_loc0p5_s7.json',
            'last_run_metrics_ch5_fixed_two_stage_balanced_layout_v1_N25_normal20_rawres_loc0p5_s42.json',
            'last_run_metrics_ch5_fixed_two_stage_balanced_layout_v1_N25_normal20_rawres_loc0p5_s123.json',
        ],
        'layout': 'two_stage_balanced_layout_v1/monitor_nodes_two_stage_balanced_layout_v1_N25.json',
    },
    'Node-Feedback': {
        'metrics': [
            'last_run_metrics_ch5_fixed_nf_val_N25_normal20_rawres_loc0p5_s7.json',
            'last_run_metrics_ch5_fixed_nf_val_N25_normal20_rawres_loc0p5_s42.json',
            'last_run_metrics_ch5_fixed_nf_val_N25_normal20_rawres_loc0p5_s123.json',
        ],
        'layout': 'learnable_layout_network_v0_scenario/monitor_nodes_learnable_layout_network_v0_scenario_N25.json',
    },
    'Embedding-Guided': {
        'metrics': [
            'last_run_metrics_ch5_fixed_embedding_guided_clean_new_N25_normal20_rawres_loc0p5_s7.json',
            'last_run_metrics_ch5_fixed_embedding_guided_clean_new_N25_normal20_rawres_loc0p5_s42.json',
            'last_run_metrics_ch5_fixed_embedding_guided_clean_new_N25_normal20_rawres_loc0p5_s123.json',
        ],
        'layout': 'embedding_guided_clean_fixed/monitor_nodes_embedding_guided_clean_N25.json',
    },
}

# 1. I/E breakdown CSV
rows_ie = []
for method, info in methods_info.items():
    i_mrrs, i_t1s, i_t3s, i_t5s = [], [], [], []
    e_mrrs, e_t1s, e_t3s, e_t5s = [], [], [], []
    for f in info['metrics']:
        m = json.load(open(str(reports / f)))
        bdt = m.get('by_defect_type', {})
        if 'I' in bdt:
            i_mrrs.append(bdt['I']['mrr']); i_t1s.append(bdt['I']['top1'])
            i_t3s.append(bdt['I']['top3']); i_t5s.append(bdt['I']['top5'])
        if 'E' in bdt:
            e_mrrs.append(bdt['E']['mrr']); e_t1s.append(bdt['E']['top1'])
            e_t3s.append(bdt['E']['top3']); e_t5s.append(bdt['E']['top5'])
    rows_ie.append({'method': method, 'defect_type': 'I',
        'mrr_mean': np.mean(i_mrrs), 'mrr_std': np.std(i_mrrs),
        'top1_mean': np.mean(i_t1s), 'top3_mean': np.mean(i_t3s), 'top5_mean': np.mean(i_t5s)})
    rows_ie.append({'method': method, 'defect_type': 'E',
        'mrr_mean': np.mean(e_mrrs), 'mrr_std': np.std(e_mrrs),
        'top1_mean': np.mean(e_t1s), 'top3_mean': np.mean(e_t3s), 'top5_mean': np.mean(e_t5s)})

with open(out / 'CH5-N25_by_defect_type_analysis.csv', 'w', newline='', encoding='utf-8-sig') as fh:
    w = csv.DictWriter(fh, fieldnames=['method', 'defect_type', 'mrr_mean', 'mrr_std', 'top1_mean', 'top3_mean', 'top5_mean'])
    w.writeheader(); w.writerows(rows_ie)
print('Saved CH5-N25_by_defect_type_analysis.csv')

# 2. Pairwise Jaccard
node_sets = {}
for method, info in methods_info.items():
    layout_path = layouts_dir / info['layout']
    data = json.load(open(str(layout_path)))
    node_sets[method] = set(data['monitor_nodes'])

jac_rows = []
for n1 in node_sets:
    for n2 in node_sets:
        s1, s2 = node_sets[n1], node_sets[n2]
        jac_rows.append({'method_a': n1, 'method_b': n2,
            'jaccard': len(s1 & s2) / len(s1 | s2),
            'shared': len(s1 & s2), 'union': len(s1 | s2)})

with open(out / 'CH5-N25_pairwise_jaccard.csv', 'w', newline='', encoding='utf-8-sig') as fh:
    w = csv.DictWriter(fh, fieldnames=['method_a', 'method_b', 'jaccard', 'shared', 'union'])
    w.writeheader(); w.writerows(jac_rows)
print('Saved CH5-N25_pairwise_jaccard.csv')

# 3. Layout structure
struct_rows = []
for method, info in methods_info.items():
    layout_path = layouts_dir / info['layout']
    data = json.load(open(str(layout_path)))
    m = data.get('layout_metrics', {})
    struct_rows.append({
        'method': method,
        'direct': m.get('direct'), 'near': m.get('near'), 'far': m.get('far'),
        'overlap_count': m.get('overlap_count'), 'mean_hop': m.get('mean_hop'),
        'max_hop': m.get('max_hop'),
        'monitor_dispersion_mean_hop': m.get('monitor_dispersion_mean_hop'),
    })

with open(out / 'CH5-N25_layout_structure.csv', 'w', newline='', encoding='utf-8-sig') as fh:
    w = csv.DictWriter(fh, fieldnames=['method', 'direct', 'near', 'far', 'overlap_count', 'mean_hop', 'max_hop', 'monitor_dispersion_mean_hop'])
    w.writeheader(); w.writerows(struct_rows)
print('Saved CH5-N25_layout_structure.csv')

# 4. Summary print
print()
print('=== I-E Gaps ===')
for r in rows_ie:
    if r['defect_type'] == 'I':
        for r2 in rows_ie:
            if r2['method'] == r['method'] and r2['defect_type'] == 'E':
                gap = r['mrr_mean'] - r2['mrr_mean']
                print(f"{r['method']:<20}: I={r['mrr_mean']:.4f} E={r2['mrr_mean']:.4f} gap={gap:.4f}")

print()
print('=== Unique Nodes ===')
for method in node_sets:
    other_union = set()
    for n2, s2 in node_sets.items():
        if n2 != method:
            other_union |= s2
    unique = node_sets[method] - other_union
    print(f'{method:<20}: {len(unique)} unique of 25')

print()
print('=== EG vs NF overlap ===')
eg = node_sets['Embedding-Guided']
nf = node_sets['Node-Feedback']
shared = eg & nf
print(f'Shared: {len(shared)}/25, Jaccard: {len(shared)/len(eg|nf):.4f}')
