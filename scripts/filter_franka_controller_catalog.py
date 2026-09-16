#!/usr/bin/env python3
"""Exclude complete grasp families that fail a recorded reference-controller check."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    with np.load(args.catalog,allow_pickle=False) as source:data={k:source[k].copy() for k in source.files}
    report=json.loads(args.report.read_text())
    if report['contract']!=json.loads(str(data['contract_json'].item())):
        raise ValueError('Controller report contract differs from the catalog')
    case=report['cases']['oracle']
    if case['target_ids']!=data['target_ids'].tolist():
        raise ValueError('Controller report must cover every catalog target in order')
    failed=~np.asarray(case['episodes_raw']['success'],dtype=bool)
    excluded=set(data['source_grasp_ids'][failed].tolist())
    keep=~np.isin(data['source_grasp_ids'],list(excluded))
    if not all(np.any(keep & (data['split']==split)) for split in ('train','validation','test')):
        raise ValueError('Filtering would empty a split')
    filtered={k:(v[keep] if v.ndim>0 and len(v)==len(keep) and not k.startswith('source_bundle_') else v)
              for k,v in data.items()}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output,**filtered)
    record={'source_catalog':str(args.catalog),'source_catalog_sha256':hashlib.sha256(args.catalog.read_bytes()).hexdigest(),
            'controller_report':str(args.report),'excluded_source_grasp_ids':sorted(excluded),
            'kept':int(keep.sum()),'removed':int((~keep).sum()),
            'split_counts':{split:int((filtered['split']==split).sum()) for split in ('train','validation','test')}}
    args.output.with_suffix('.filter.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
