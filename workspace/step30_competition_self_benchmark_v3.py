"""Audit-corrected participant benchmark with two-sided leave-one-model/seed-out."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from typing import Sequence
import numpy as np
import step30_competition_self_benchmark_v1 as b
from step30_material_model_calibration import finnie_volume_mm3_per_kg,mansouri_mass_ratio_kg_per_kg,oka_volume_mm3_per_kg,tabakoff_volume_mm3_per_kg
from step30_validation_common import canonical_sha256,safe_json_dump,safe_npz_dump
VERSION='step30-competition-self-benchmark-v3-two-sided-lomo'
MODELS=list(b.MODEL_NAMES)
ANCHOR={'speed_m_s':100.0,'angle_deg':45.0,'d50_um':50.0,'hardness_gpa':4.0}
def _anchor(name):
 s,a,d=ANCHOR['speed_m_s'],ANCHOR['angle_deg'],ANCHOR['d50_um']
 if name=='oka': return float(oka_volume_mm3_per_kg(s,a,d,hardness_gpa=ANCHOR['hardness_gpa']))
 if name=='finnie': return float(finnie_volume_mm3_per_kg(s,a))
 if name=='tabakoff': return float(tabakoff_volume_mm3_per_kg(s,a))
 if name=='mansouri': return float(mansouri_mass_ratio_kg_per_kg(s,a)) / 7800.0 * 1.0e9
 raise ValueError(name)
ANCHORS={m:_anchor(m) for m in MODELS}
def _scaled(m): return np.asarray(m,dtype=float)/max(ANCHORS[m],1e-30)
def _combine(maps,names): return np.mean(np.stack([_scaled_maps(maps,m) for m in names],axis=0),axis=0)
def _scaled_maps(maps,m): return np.asarray(maps[m],dtype=float)/max(ANCHORS[m],1e-30)
def _summary(pred,ref,region,areas):
 rows=[]
 for sc in b._scenario_grid():
  # Material uncertainty scales both participant prediction and reference.
  rr=b._scenario_reference(ref,region,material_factor=sc['material_factor'],region_expansion=sc['region_expansion'])
  pp=pred*sc['material_factor']
  rows.append({**sc,**b._pair_score(pp,rr,areas)})
 nom=min(rows,key=lambda x:abs(x['material_factor']-1)+abs(x['region_expansion']-1)); cons=min(rows,key=lambda x:x['score']); opt=max(rows,key=lambda x:x['score'])
 return {'nominal':nom,'conservative':cons,'optimistic':opt,'score_interval':{'lower':float(cons['score']),'upper':float(opt['score'])},'scenario_count':len(rows)}
def run(merged_path,seeds,lit_path,region_path,out):
 merged=b._load_prior(merged_path); items=[b._load_prior(p) for p in seeds]; region,_=b._load_optional_region_field(region_path,len(merged['areas']))
 if region is None: region=np.ones(len(merged['areas']))
 areas=merged['areas']; pred_maps=merged['maps']; model_loo={}; seed_loo={}; arrays={}
 for omit in MODELS:
  names=[m for m in MODELS if m!=omit]
  pred=_combine(pred_maps,names)
  reference_maps={m:np.mean(np.stack([x['maps'][m] for x in items]),axis=0) for m in names}
  ref=_combine(reference_maps,names)
  _,shape=b._reference_from_candidates([reference_maps[m] for m in names],region)
  model_loo[omit]=_summary(pred,ref,shape,areas); arrays[f'prediction_omit_{omit}']=pred.astype(np.float32); arrays[f'reference_omit_{omit}']=ref.astype(np.float32)
  per_seed=[]
  for leave in range(len(items)):
   loo_maps={m:np.mean(np.stack([x['maps'][m] for i,x in enumerate(items) if i!=leave]),axis=0) for m in names}
   ref_seed=_combine(loo_maps,names)
   _,seed_shape=b._reference_from_candidates([loo_maps[m] for m in names],region)
   per_seed.append(_summary(_combine(items[leave]['maps'],names),ref_seed,seed_shape,areas))
  seed_loo[omit]=per_seed
 model_nom=float(np.median([v['nominal']['score'] for v in model_loo.values()])); model_cons=float(np.median([v['conservative']['score'] for v in model_loo.values()])); model_opt=float(np.median([v['optimistic']['score'] for v in model_loo.values()]))
 seed_nom=float(np.median([v['nominal']['score'] for vals in seed_loo.values() for v in vals])); seed_cons=float(np.median([v['conservative']['score'] for vals in seed_loo.values() for v in vals])); seed_opt=float(np.median([v['optimistic']['score'] for vals in seed_loo.values() for v in vals]))
 report={'version':VERSION,'status':'conditional','benchmark_kind':b.BENCHMARK_KIND,'reference_independence':'two_sided_leave_one_model_out_and_leave_one_seed_out','participant_built':True,'measured_truth':False,'competition_self_benchmark_gate':True,'competition_self_benchmark_status':'conditional_two_sided_internal_consistency','model_combination':{'weights':{m:1/4 for m in MODELS},'normalization':'fixed 100 m/s, 45 deg, d50=50 um, density=7800 kg/m3 response anchors; equal-weight mean','anchor_responses':ANCHORS,'posthoc_weight_change_allowed':False},'competition_self_benchmark_score':model_cons,'competition_self_benchmark_score_interval':{'lower':min(model_cons,seed_cons),'upper':max(model_opt,seed_opt)},'nominal_score':model_nom,'conservative_score':model_cons,'optimistic_score':model_opt,'seed_leave_one_out_summary':{'nominal':seed_nom,'conservative':seed_cons,'optimistic':seed_opt},'official_accuracy_gate':False,'official_reference_gate':False,'fem_gate':'blocked','model_leave_one_out':model_loo,'seed_leave_one_out':seed_loo,'inputs':{'merged_prior':str(Path(merged_path).resolve()),'seed_count':len(items),'n_faces':len(areas),'region_field':str(Path(region_path).resolve()),'literature_reference':str(Path(lit_path).resolve())},'claim_scope':'conditional participant-built internal-consistency score; not official accuracy and not measured wear truth','limitations':['The score is not an independent physical accuracy measurement.','Absolute amplitude remains prior-sensitive despite common fixed-anchor normalization.','Official gates remain closed.']}
 report['report_sha256']=canonical_sha256(report); safe_json_dump(report,Path(out)); safe_npz_dump(arrays,Path(out).with_suffix('.npz'),metadata={'report_sha256':report['report_sha256'],'version':VERSION}); return report
def main(argv:Sequence[str]|None=None):
 p=argparse.ArgumentParser();p.add_argument('--merged-prior',type=Path,required=True);p.add_argument('--seed-prior',type=Path,action='append',required=True);p.add_argument('--literature-reference',type=Path,required=True);p.add_argument('--region-field',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args(argv);print(json.dumps(run(a.merged_prior,a.seed_prior,a.literature_reference,a.region_field,a.output),ensure_ascii=False,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
