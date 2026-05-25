
import numpy as np

def segment_ranking_metrics(score_matrix,target_indices,topk=(1,3,5)):
    scores=np.asarray(score_matrix); targets=np.asarray(target_indices).astype(int)
    valid=targets>=0; scores=scores[valid]; targets=targets[valid]
    if len(targets)==0: return {"mrr":0.0, **{f"top{k}":0.0 for k in topk}}
    order=np.argsort(-scores,axis=1)
    pos=(order==targets[:,None]).argmax(axis=1)+1
    out={"mrr":float(np.mean(1.0/pos))}
    for k in topk: out[f"top{k}"]=float(np.mean(pos<=k))
    return out

def macro_f1_from_logits(logits,target_labels,num_classes=2):
    pred=np.asarray(logits).argmax(axis=1); y=np.asarray(target_labels).astype(int); vals=[]
    for c in range(num_classes):
        tp=np.sum((pred==c)&(y==c)); fp=np.sum((pred==c)&(y!=c)); fn=np.sum((pred!=c)&(y==c))
        p=tp/max(tp+fp,1); r=tp/max(tp+fn,1); vals.append(0.0 if p+r==0 else 2*p*r/(p+r))
    return float(np.mean(vals))

def summarize_segment_joint_metrics(segment_scores,target_segment_idx,type_logits=None,type_targets=None):
    out=segment_ranking_metrics(segment_scores,target_segment_idx)
    if type_logits is not None and type_targets is not None: out["type_macro_f1"]=macro_f1_from_logits(type_logits,type_targets)
    return out
