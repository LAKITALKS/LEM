"""Bounded production/continuation fixtures; no cloud calls or study observations."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, MagicMock
import numpy as np

from lem_ii import production, study, evaluation
from lem_ii.design import load_config, build_sessions
from lem_ii.storage import atomic_json, canonical_hash, file_hash
from lem_ii.features import METHODS, LeakageError


class ProductionTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config()

    def test_only_six_execution_gates_change_across_all_phases(self):
        for phase in production.PHASES:
            enabled = production.phase_config(self.config, phase)
            self.assertEqual(study._scientific_hash(enabled), study._scientific_hash(self.config))
            for key in set(self.config) - {'execution'}:
                self.assertEqual(enabled[key], self.config[key])
            self.assertEqual(enabled['execution']['confirmation_generation_allowed'], phase == 'confirmation')

    def test_budget_reserves_all_later_phases_and_never_refunds_failures(self):
        ledger = {'schema':'lem-ii.production-ledger.v1','attempts':[], 'completed_phases':[]}
        for i in range(2):
            attempt=production.reserve_attempt(ledger, f'fixture-{i}', 'development')
            attempt['status']='failed'
        production.reserve_attempt(ledger,'fixture-2','development')['status']='complete'
        ledger['completed_phases'].append('development')
        production.reserve_attempt(ledger,'fixture-3','validation')['status']='failed'
        with self.assertRaises(ValueError):
            production.reserve_attempt(ledger,'fixture-4','validation')
        self.assertEqual(sum(a['reserved_usd'] for a in ledger['attempts']),18)

    def test_phase_order_active_and_consumed_attempt_rejected(self):
        ledger={'schema':'lem-ii.production-ledger.v1','attempts':[],'completed_phases':[]}
        with self.assertRaises(ValueError): production.reserve_attempt(ledger,'x','confirmation')
        production.reserve_attempt(ledger,'x','development')
        with self.assertRaises(ValueError): production.reserve_attempt(ledger,'y','development')
        ledger['attempts'][0]['status']='failed'
        with self.assertRaises(ValueError): production.reserve_attempt(ledger,'x','development')

    def test_remote_source_attestation_checks_all_bytes_without_git(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'lem_ii').mkdir()
            files={}
            for name in study.SOURCE_FILES:
                path=root/'lem_ii'/name; path.write_text('fixture:'+name)
                files['lem_ii/'+name]=file_hash(path)
            sources={n:files['lem_ii/'+n] for n in study.SOURCE_FILES}
            manifest={'schema':'lem-ii.source-attestation.v1','commit':'a'*40,'files':files,
                      'source_sha256':sources,'source_set_sha256':canonical_hash(sources)}
            self.assertEqual(production.verify_source(manifest,root),'a'*40)
            (root/'lem_ii/model_adapter.py').write_text('tampered')
            with self.assertRaises(ValueError): production.verify_source(manifest,root)

    def test_complete_inventory_skips_model_and_preserves_existing_records(self):
        config=production.phase_config(self.config,'development')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for spec in (s for s in build_sessions(config) if s.split=='development'):
                folder=root/'data'/spec.session_id; folder.mkdir(parents=True)
                atomic_json(folder/'manifest.json',{'binding':{'fixture_only':True},'status':'complete','completed_turns':24})
            with patch.object(production,'verify_source',return_value='fixture'), \
                 patch.object(study,'_validate_binding'), patch.object(production,'SessionStore') as store:
                store.return_value.manifest={'status':'complete'}
                loader=MagicMock()
                result=production.collect_phase(root,self.config,'development','fixture',{},adapter_loader=loader)
                self.assertEqual(result['new_model_calls'],0)
                self.assertEqual(len(result['complete_sessions']),36)
                loader.assert_not_called()
                self.assertFalse((root/'study-usage-ledger.json').exists())

    def test_confirmation_collection_requires_freeze_before_adapter_or_reservation(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(production,'verify_source',return_value='fixture'):
            loader=MagicMock()
            with self.assertRaises(FileNotFoundError):
                production.collect_phase(tmp,self.config,'confirmation','fixture',{},adapter_loader=loader)
            loader.assert_not_called()
            self.assertFalse((Path(tmp)/'study-usage-ledger.json').exists())

    def test_continuation_guard_counts_prefix_and_only_reserves_new_tokens(self):
        with tempfile.TemporaryDirectory() as tmp:
            limits={'deadline_epoch':100,'max_sessions':2,'max_prefill_tokens':1000,'max_generation_tokens':384,
                    'completed_prefixes':{'partial':23,'other':22}}
            guard=study.StudyRuntimeGuard(limits,Path(tmp)/'guard.json',clock=lambda:0)
            guard.start_session('partial',24); guard.reserve_turn(10,128); guard.finish_turn(128)
            with self.assertRaises(RuntimeError): guard.reserve_turn(10,128)
            guard.start_session('other',24)
            for _ in range(2): guard.reserve_turn(10,128); guard.finish_turn(128)
            self.assertEqual(len(guard.state['turns']),3)

    def test_live_preflight_rejects_wrong_limit_live_jobs_and_insufficient_headroom(self):
        from datetime import datetime, timezone
        from lem_ii import launch_study
        proof={'verified_at_utc':datetime.now(timezone.utc).isoformat(),'workspace':'fixture-workspace',
               'environment':'main','workspace_gross_usage_limit_usd':27,'workspace_usage_upper_usd':0.1,
               'rates':{'L4':0.000222,'CPU_core':0.0000131,'RAM_GiB':0.00000222}}
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'proof.json'; atomic_json(path,proof)
            with patch.object(launch_study,'command',side_effect=['fixture-workspace','[]']):
                self.assertEqual(launch_study.read_preflight(path),proof)
            with patch.object(launch_study,'command',side_effect=['fixture-workspace','[{"state":"running","tasks":1}]']):
                with self.assertRaises(ValueError): launch_study.read_preflight(path)
            proof['workspace_usage_upper_usd']=3.1; atomic_json(path,proof)
            with patch.object(launch_study,'command',side_effect=['fixture-workspace','[]']):
                with self.assertRaises(ValueError): launch_study.read_preflight(path)
            proof['workspace_gross_usage_limit_usd']=30; atomic_json(path,proof)
            with self.assertRaises(ValueError): launch_study.read_preflight(path)

    def test_export_cannot_replace_completed_raw_data_or_reset_ledger(self):
        import zipfile
        from lem_ii.launch_study import import_export
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); destination=root/'local'; destination.mkdir()
            session=destination/'data/study-fixture'; session.mkdir(parents=True)
            manifest={'status':'complete','completed_turns':1,'binding':{'fixture':True},'turn_hashes':[{'fixture':'hash'}]}
            atomic_json(session/'manifest.json',manifest)
            (session/'states-fixture.txt').write_text('original fixture')
            archive=root/'bad.zip'
            with zipfile.ZipFile(archive,'w') as handle:
                handle.writestr('data/study-fixture/manifest.json',json.dumps(manifest))
                handle.writestr('data/study-fixture/states-fixture.txt','changed fixture')
            with self.assertRaises(ValueError): import_export(archive,destination,file_hash(archive))
            self.assertEqual((session/'states-fixture.txt').read_text(),'original fixture')
            ledger={'source_manifest_sha256':'fixture-source','scientific_config_sha256':'fixture-config',
                    'attempts':[{'attempt_id':'fixture','reserved_usd':4.5}]}
            atomic_json(destination/'study-usage-ledger.json',ledger)
            archive=root/'ledger.zip'; changed=deepcopy(ledger); changed['attempts']=[]
            with zipfile.ZipFile(archive,'w') as handle:
                handle.writestr('study-usage-ledger.json',json.dumps(changed))
            with self.assertRaises(ValueError): import_export(archive,destination,file_hash(archive))
            self.assertEqual(json.loads((destination/'study-usage-ledger.json').read_text()),ledger)

    def fixture_confirmation(self,directory):
        config=production.phase_config(self.config,'confirmation')
        rows=[{'dialogue_id':f'fixture-{i}','profile_id':f'fixture-profile-{i}','regime':label,
               'topic_family':'fixture-topic','formulation_family':'fixture-phrasing',
               'contexts':['fixture-only text'],'states':np.zeros((1,2),dtype=np.float32)}
              for i,label in enumerate(('a','b','c'))]
        for endpoint in (12,16,20,24):
            (directory/f'study_endpoint_{endpoint}.joblib').write_bytes(f'fixture-{endpoint}'.encode())
        def load(path,*args):
            endpoint=int(path.stem.split('_')[-1])
            return {'endpoint':endpoint,'development_ids':['fixture-dev'],'validation_ids':['fixture-val'],
                    'pipeline':MagicMock()}
        return config,rows,load

    def test_confirmation_recovers_same_trial_and_never_predicts_saved_endpoint_again(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp); config,rows,load=self.fixture_confirmation(directory)
            output=directory/'result.json'; predictions={m:np.array(['a','b','c']) for m in METHODS}
            def fail_second(fitted,records):
                if fitted['endpoint']==16: raise OSError('fixture interruption')
                return predictions
            with patch.object(evaluation,'_validate_confirmation_input'), \
                 patch.object(evaluation,'_validate_confirmation_artifact'), \
                 patch.object(evaluation,'load_frozen_artifact',side_effect=load), \
                 patch.object(evaluation,'time_countercheck',return_value={'fixture_only':True}), \
                 patch.object(evaluation,'paired_cluster_intervals',return_value={}), \
                 patch.object(evaluation,'simultaneous_grid_intervals',return_value={}):
                with patch.object(evaluation,'predict',side_effect=fail_second):
                    with self.assertRaises(OSError):
                        evaluation.predict_confirmation_grid_once(directory,rows,config,output)
                first=file_hash(directory/'confirmation_predictions_12.json')
                trial=json.loads((directory/'confirmation_grid.lock').read_text())['trial_id']
                with patch.object(evaluation,'predict',return_value=predictions) as predict:
                    with self.assertRaises(LeakageError):
                        evaluation.predict_confirmation_grid_once(directory,rows,config,directory/'other.json',resume=True)
                    result=evaluation.predict_confirmation_grid_once(directory,rows,config,output,resume=True)
                    self.assertEqual(predict.call_count,3)
                    self.assertEqual(result['trial_id'],trial)
                    self.assertEqual(file_hash(directory/'confirmation_predictions_12.json'),first)
                    again=evaluation.predict_confirmation_grid_once(directory,rows,config,output,resume=True)
                    self.assertEqual(again,json.loads(json.dumps(result))); self.assertEqual(predict.call_count,3)
                    with self.assertRaises(LeakageError):
                        evaluation.predict_confirmation_grid_once(directory,rows,config,output)
                    changed=deepcopy(rows); changed[0]['states'][0,0]=1
                    with self.assertRaises(LeakageError):
                        evaluation.predict_confirmation_grid_once(directory,changed,config,output,resume=True)


if __name__=='__main__': unittest.main()
