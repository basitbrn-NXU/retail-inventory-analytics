"""Run with python -m flows.m5_pipeline --source LOCAL_ZIP_OR_DIRECTORY."""
import argparse
import hashlib
import importlib.metadata as md
import json
import os
import subprocess
import uuid
from pathlib import Path
os.environ.setdefault('PREFECT_SERVER_ANALYTICS_ENABLED','false')
os.environ.setdefault('DO_NOT_TRACK','1')
from prefect import flow, task
from prefect.cache_policies import NONE
from src.prepare import (Audit,read_inputs,integrate,prepare_features,anonymize,
                         assert_output,write_json)
from src.validate import run_validation
from src.bias import representation

@task(name='Ingest and verify sources',cache_policy=NONE,persist_result=False)
def ingest(source, audit):
    return read_inputs(source,'manifests/cohort_manifest.csv','manifests/cohort_audit.json',audit)

@task(name='Integrate daily sales calendar and prices',cache_policy=NONE,persist_result=False)
def join_sources(inputs,audit):
    frame=integrate(*inputs[:3]);audit.record('transform','daily_integration',rows=len(frame));return frame

@task(name='Transform with chronological safeguards',cache_policy=NONE,persist_result=False)
def transform(frame,audit):
    prepared=prepare_features(frame);audit.record('transform','features_and_splits',rows=len(prepared));return prepared

@task(name='Anonymize and check contact information',cache_policy=NONE,persist_result=False)
def privacy(frame,audit):
    clean,removed=anonymize(frame);audit.record('transform','privacy_screen',removed_column_count=len(removed));return clean,removed

@task(name='Validate prepared output',cache_policy=NONE,persist_result=False)
def validate(frame,out,audit):
    assert_output(frame);stats=run_validation(frame,out);audit.record('validate','great_expectations',**stats);return stats

@task(name='Check cohort representation',cache_policy=NONE,persist_result=False)
def bias_check(metadata,ids,out,audit):
    result=representation(metadata,ids);write_json(Path(out)/'bias_report.json',result)
    audit.record('validate','representation_bias',review_flags=len(result['flags']));return result

@task(name='Publish local Parquet and lineage',cache_policy=NONE,persist_result=False)
def publish(frame,inputs,stats,bias,removed,out,audit):
    out=Path(out);tmp=out/'prepared.tmp.parquet';destination=out/'prepared.parquet'
    frame.to_parquet(tmp,index=False);tmp.replace(destination)
    # Reading back the persisted output is logged and its schema/grain rechecked.
    audit.record('write','prepared.parquet',rows=len(frame))
    import pandas as pd
    audit.record('read','prepared.parquet',purpose='persistence verification')
    assert_output(pd.read_parquet(destination))
    try:
        commit=subprocess.check_output(['git','rev-parse','HEAD'],stderr=subprocess.DEVNULL,text=True).strip()
    except (subprocess.CalledProcessError,FileNotFoundError):
        commit='uncommitted_local_working_copy'
    code={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for folder in ['src','flows'] for p in Path(folder).glob('*.py')}
    lineage=dict(run_id=audit.run_id,source_sha256=inputs[4],source_checks=inputs[5],code_commit=commit,code_sha256=code,
        output_rows=len(frame),output_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
        splits=frame.groupby('split').size().to_dict(),date_range=[str(frame.date.min().date()),str(frame.date.max().date())],
        missing_prices=int(frame.price_missing.sum()),pii_columns_removed=removed,validation=stats,bias_review_flags=bias['flags'],
        versions={p:md.version(p) for p in ['pandas','numpy','prefect','great-expectations','fairlearn','pyarrow']},
        holdout_features='Post-2016-03-27 actual sales excluded before lag/rolling calculation; future prices masked. Holdout labels are evaluation only. Module 4 must generate recursive/direct horizon inputs.',
        stages=['source verification','ingestion','integration','feature transformation','privacy screen','validation','bias review','local publication'])
    write_json(out/'lineage.json',lineage);audit.record('write','lineage.json')
    return lineage

@flow(name='M5 retail analytics preparation',log_prints=False,persist_result=False)
def m5_pipeline(source,output='data/processed'):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    run_id=str(uuid.uuid4());audit=Audit(out/'privacy_audit.jsonl',run_id)
    audit.record('start','pipeline')
    try:
        inputs=ingest(source,audit)
        joined=join_sources(inputs,audit)
        frame=transform(joined,audit)
        frame,removed=privacy(frame,audit)
        stats=validate(frame,out,audit)
        bias=bias_check(inputs[3],set(frame.id),out,audit)
        result=publish(frame,inputs,stats,bias,removed,out,audit)
        audit.record('complete','pipeline',rows=result['output_rows'])
        return result
    except Exception as e:
        audit.record('failure','pipeline',error_type=type(e).__name__)
        raise

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',required=True);parser.add_argument('--output',default='data/processed')
    args=parser.parse_args();result=m5_pipeline(args.source,args.output)
    print(json.dumps({'run_id':result['run_id'],'output_rows':result['output_rows'],'splits':result['splits'],'validation':result['validation'],'bias_review_flags':result['bias_review_flags']},indent=2))
