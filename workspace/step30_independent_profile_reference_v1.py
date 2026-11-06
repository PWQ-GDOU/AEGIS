"""Create a participant-built independent baseline/worn face-depth reference.

This is a synthetic registered reference, not measured truth. It deliberately
reads only the official face domain and participant DEC activity prior, never
any evaluated erosion-law amplitude map.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Sequence
import numpy as np
from step30_validation_common import canonical_sha256, safe_json_dump, safe_npz_dump, sha256_file
VERSION='step30-independent-profile-reference-v1'
def build(domain_path:str|Path, region_path:str|Path, output_prefix:str|Path)->dict:
    domain=Path(domain_path).resolve(); region_path=Path(region_path).resolve(); out=Path(output_prefix).resolve()
    with np.load(domain,allow_pickle=False) as z:
        centers=np.asarray(z['face_centers_local_mm'],dtype=float); areas=np.asarray(z['face_area_m2'],dtype=float)
    with np.load(region_path,allow_pickle=False) as z:
        region=np.asarray(z['literature_region_prior'],dtype=float)
    if centers.ndim!=2 or centers.shape[1]!=3 or len(centers)!=len(areas) or len(region)!=len(areas): raise ValueError('domain mismatch')
    if not np.isfinite(centers).all() or not np.isfinite(areas).all() or np.any(areas<=0) or not np.isfinite(region).all(): raise ValueError('nonfinite domain')
    # Frozen geometry-only construction: region selects candidate faces; depth is
    # an independent low-frequency synthetic contour, not an erosion-law map.
    xyz=centers-centers.mean(axis=0)
    scale=np.std(xyz,axis=0); scale=np.where(scale>1e-12,scale,1.0); q=xyz/scale
    geom=np.exp(-0.5*((q[:,0]/1.35)**2+(q[:,1]/1.10)**2))*(0.65+0.35*np.clip((q[:,2]-q[:,2].min())/(np.ptp(q[:,2])+1e-12),0,1))
    support=np.clip(region,0,1)
    active=support>0
    if not np.any(active): raise ValueError('no active faces')
    wear=0.012*(0.25+0.75*support**0.60)*(0.35+0.65*geom)
    wear[~active]=0.0
    # deterministic microtexture is bounded and independent of model output
    texture=0.90+0.10*np.sin(0.071*q[:,0]+0.113*q[:,1]+0.157*q[:,2])
    wear=np.maximum(wear*texture,0.0)
    baseline=np.zeros(len(areas),dtype=float)
    worn=baseline-wear
    npz=out.with_suffix('.npz'); report_path=out.with_suffix('.json')
    safe_npz_dump({'face_centers_local_mm':centers.astype(np.float32),'face_area_m2':areas.astype(np.float32),'baseline_depth_mm':baseline.astype(np.float32),'worn_depth_mm':worn.astype(np.float32),'wear_depth_mm':wear.astype(np.float32),'active_face_mask':active.astype(np.uint8)},npz,metadata={'version':VERSION,'units':'mm','source':'participant_built_synthetic','measured_truth':False})
    registration={'version':VERSION,'status':'registered_synthetic_only','source_type':'local_synthetic_fixture','reference_role':'synthetic_pipeline_fixture','coordinate_system':'official_face_local_mm','face_id_namespace':'official_face_domain_index','domain_n_faces':int(len(areas)),'domain_centers_sha256':canonical_sha256(centers.tolist()),'domain_areas_sha256':canonical_sha256(areas.tolist()),'profile_npz_path':str(npz),'profile_npz_sha256':sha256_file(npz),'baseline_field':'baseline_depth_mm','worn_field':'worn_depth_mm','wear_field':'wear_depth_mm','units':'mm','construction':'geometry envelope times participant DEC activity support; fixed 0.012 mm nominal amplitude; deterministic bounded texture','independent_of_evaluated_model_maps':True,'measured_truth':False,'official_reference':False,'claim_scope':'participant-built synthetic baseline/worn depth reference for conditional benchmark only'}
    registration['registration_sha256']=canonical_sha256(registration); safe_json_dump(registration,report_path)
    manifest={'version':VERSION,'status':'participant_built_synthetic_registered_profile','profile':{'path':str(npz),'sha256':sha256_file(npz)},'registration':{'path':str(report_path),'sha256':sha256_file(report_path)},'inputs':{'domain_path':str(domain),'domain_sha256':sha256_file(domain),'region_path':str(region_path),'region_sha256':sha256_file(region_path)},'n_faces':int(len(areas)),'active_face_fraction':float(np.mean(active)),'wear_depth_stats_mm':{'p50':float(np.quantile(wear[active],.5)),'p95':float(np.quantile(wear[active],.95)),'max':float(np.max(wear))},'measured_truth':False,'official_accuracy_gate':False,'official_reference_gate':False,'fem_gate':'blocked'}
    manifest['manifest_sha256']=canonical_sha256(manifest); safe_json_dump(manifest,out.with_name(out.name+'_manifest.json')); return manifest
def main(argv:Sequence[str]|None=None)->int:
    p=argparse.ArgumentParser();p.add_argument('--domain',type=Path,required=True);p.add_argument('--region',type=Path,required=True);p.add_argument('--output-prefix',type=Path,required=True);a=p.parse_args(argv);print(json.dumps(build(a.domain,a.region,a.output_prefix),ensure_ascii=False,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
