
import json, numpy as np, pandas as pd, torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, Subset

class SegmentDetectionDataset(Dataset):
    def __init__(self,node_timeseries_file,edge_timeseries_file,adjacency_matrix_file,node_list_file,segment_list_file,defect_matrix_file,sequence_length=36,window_stride=6,allowed_defect_types=None):
        self.sequence_length=int(sequence_length); self.window_stride=max(int(window_stride),1)
        self.node_data=pd.read_parquet(node_timeseries_file) if str(node_timeseries_file).endswith(".parquet") else pd.read_csv(node_timeseries_file)
        self.edge_data=pd.read_parquet(edge_timeseries_file) if str(edge_timeseries_file).endswith(".parquet") else pd.read_csv(edge_timeseries_file)
        self.node_data["datetime"]=pd.to_datetime(self.node_data["datetime"]); self.edge_data["datetime"]=pd.to_datetime(self.edge_data["datetime"])
        self.adj_matrix=np.load(adjacency_matrix_file)
        node_info=json.loads(Path(node_list_file).read_text(encoding="utf-8"))
        self.node_list=[str(n) for n in (node_info.get("node_list",node_info) if isinstance(node_info,dict) else node_info)]
        self.node_to_idx={str(k):int(v) for k,v in node_info.get("node_to_idx",{}).items()} if isinstance(node_info,dict) and node_info.get("node_to_idx") else {str(n):i for i,n in enumerate(self.node_list)}
        seg_payload=json.loads(Path(segment_list_file).read_text(encoding="utf-8"))
        self.segment_list=seg_payload.get("segments",[])
        self.node_to_primary_out={str(k):str(v) for k,v in seg_payload.get("node_to_primary_out_segment",{}).items()}
        self.segment_ids=[str(s["segment_id"]) for s in self.segment_list]
        self.segment_to_idx={str(s["segment_id"]):int(s["segment_idx"]) for s in self.segment_list}
        self.link_to_segment={str(s["link_id"]):str(s["segment_id"]) for s in self.segment_list}
        self.edge_index=np.array([[self.node_to_idx.get(str(s["from_node"]),-1),self.node_to_idx.get(str(s["to_node"]),-1)] for s in self.segment_list],dtype=np.int64).T
        dm=pd.read_csv(defect_matrix_file)
        if allowed_defect_types is not None and "defect_type" in dm.columns:
            allowed={str(x).upper() for x in allowed_defect_types}; dm=dm[dm["defect_type"].astype(str).str.upper().isin(allowed)].copy()
        self.node_feature_cols=[c for c in ["depth","pollut_BODf","depth_residual","pollut_BODf_residual","head","total_inflow"] if c in self.node_data.columns]
        self.edge_feature_cols=[c for c in ["flow","velocity","depth","depth_diff_ud","head_diff_ud","inflow_diff_ud","flux_proxy","length","diameter","slope"] if c in self.edge_data.columns]
        self.feature_cols=list(self.node_feature_cols)
        self.samples=[]
        ng={int(sid):df.sort_values("datetime") for sid,df in self.node_data.groupby("scenario_id")}; eg={int(sid):df.sort_values("datetime") for sid,df in self.edge_data.groupby("scenario_id")}
        for _,row in dm.iterrows():
            sid=int(row["defect_id"]) if "defect_id" in row else int(row["scenario_id"])
            if sid not in ng or sid not in eg: continue
            n=len(sorted(ng[sid]["datetime"].drop_duplicates().tolist()))-self.sequence_length+1
            if n<1: continue
            dt=str(row.get("defect_type","")).upper().strip(); link_id=str(row.get("link_id","")).strip(); node_id=str(row.get("node_id","")).strip(); target=-1
            if dt=="E": target=self.segment_to_idx.get(self.link_to_segment.get(link_id,""),-1)
            elif dt=="I": target=self.segment_to_idx.get(self.node_to_primary_out.get(node_id,""),-1)
            for start in range(0,n,self.window_stride): self.samples.append({"scenario_id":sid,"start":start,"target_segment_idx":int(target),"defect_type_str":dt,"defect_node_id":node_id})
    def __len__(self): return len(self.samples)
    def _window(self,df,id_col,id_order,feat_cols,start):
        times=sorted(df["datetime"].drop_duplicates().tolist())[start:start+self.sequence_length]; win=df[df["datetime"].isin(times)]
        arr=np.zeros((len(times),len(id_order),len(feat_cols)),dtype=np.float32); im={str(v):i for i,v in enumerate(id_order)}; tm={t:i for i,t in enumerate(times)}
        for row in win.itertuples(index=False):
            ii=tm[getattr(row,"datetime")]; jj=im.get(str(getattr(row,id_col)))
            if jj is not None: arr[ii,jj,:]=np.asarray([getattr(row,c,0.0) if pd.notna(getattr(row,c,0.0)) else 0.0 for c in feat_cols],dtype=np.float32)
        return arr
    def __getitem__(self,idx):
        s=self.samples[idx]; sid=int(s["scenario_id"]); start=int(s["start"]); ndf=self.node_data[self.node_data["scenario_id"]==sid]; edf=self.edge_data[self.edge_data["scenario_id"]==sid]
        return {"features":torch.tensor(self._window(ndf,"node_id",self.node_list,self.node_feature_cols,start),dtype=torch.float32),"edge_features":torch.tensor(self._window(edf,"segment_id",self.segment_ids,self.edge_feature_cols,start),dtype=torch.float32),"adj_matrix":torch.tensor(self.adj_matrix,dtype=torch.float32),"edge_index":torch.tensor(self.edge_index,dtype=torch.long),"candidate_segment_mask":torch.ones(len(self.segment_ids),dtype=torch.float32),"target_segment_idx":torch.tensor(s["target_segment_idx"],dtype=torch.long),"defect_type_str":s["defect_type_str"],"scenario_id":sid,"defect_node_id":s["defect_node_id"]}

def create_dataloaders(node_timeseries_file,edge_timeseries_file,adjacency_matrix_file,node_list_file,segment_list_file,defect_matrix_file,batch_size=16,sequence_length=36,window_stride=6,train_ratio=0.7,val_ratio=0.15,allowed_defect_types=None,random_seed=42,**kwargs):
    ds=SegmentDetectionDataset(node_timeseries_file=node_timeseries_file,edge_timeseries_file=edge_timeseries_file,adjacency_matrix_file=adjacency_matrix_file,node_list_file=node_list_file,segment_list_file=segment_list_file,defect_matrix_file=defect_matrix_file,sequence_length=sequence_length,window_stride=window_stride,allowed_defect_types=allowed_defect_types)
    mp={}
    for i,s in enumerate(ds.samples): mp.setdefault(int(s["scenario_id"]),[]).append(i)
    ids=sorted(mp.keys()); rng=np.random.default_rng(random_seed); rng.shuffle(ids); n=len(ids); nt=int(n*train_ratio); nv=int(n*val_ratio)
    def gather(xs):
        out=[]
        for sid in xs: out.extend(mp[sid])
        return out
    return DataLoader(Subset(ds,gather(ids[:nt])),batch_size=batch_size,shuffle=True), DataLoader(Subset(ds,gather(ids[nt:nt+nv])),batch_size=batch_size,shuffle=False), DataLoader(Subset(ds,gather(ids[nt+nv:])),batch_size=batch_size,shuffle=False), ds
