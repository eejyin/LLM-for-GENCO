"""Behavioral checks for field mappings, safe routing and solver evidence."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from fastapi.testclient import TestClient
from backend import service
from backend.configuration import apply_values, describe
from backend.evidence import summarize_csv
from backend.tasks import JobManager
from backend.reporting import compact, render, validate_observations
from fastapi import HTTPException

class LanguageChecks(unittest.TestCase):
    def test_repository_sources_contain_no_chinese_characters(self):
        root = Path(__file__).resolve().parents[1]
        sources = [root / "README.md", root / ".env.example"]
        for folder in ("backend", "frontend", "scripts", "tests"):
            sources.extend(path for path in (root / folder).rglob("*") if path.suffix in {".py", ".js", ".html", ".css", ".md", ".sh"})
        for path in sources:
            self.assertFalse(any("\u3400" <= character <= "\u9fff" for character in path.read_text()), str(path))

class StudioTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(service.app)
        self.name = "hello_world_train_case14.yaml"
        if self.name not in service.CONFIGS:
            self.skipTest("Set GENCO_ROOT to the local OpenGridFM tree")

    def test_layer_parameter_updates_real_loss_and_checkpoint(self):
        data = apply_values(service.get_config(self.name), {"num_layers": 3, "base_weight": .7})
        self.assertEqual(data["callbacks"]["checkpoint_monitor"], "Validation layer_2_residual")
        self.assertEqual(data["training"]["loss_args"][0]["base_weight"], .7)
        self.assertNotIn("base_weight", data["training"])
        weights = describe(data)["layer_weights"]
        self.assertAlmostEqual(sum(weights), 1)
        self.assertGreater(weights[-1], weights[0])

    def test_reject_invalid_parameters_before_training(self):
        for changes in ({"epochs": -1}, {"epochs": True}, {"num_layers": 1.5},
                        {"val_ratio": .6, "test_ratio": .6}, {"loss_weight_0": 0, "loss_weight_1": 0}):
            response = self.client.post("/api/parameters/preview", json={"config": self.name, "values": changes})
            self.assertEqual(response.status_code, 400, response.text)

    def test_parameter_route_returns_validated_changes(self):
        with patch.object(service, "structured", return_value={"route": "PARAMETER", "changes": {"epochs": 3}}):
            result = self.client.post("/api/chat", json={"config": self.name, "message": "set epochs to 3"}).json()
        self.assertEqual(result["changes"]["epochs"], 3)
        self.assertEqual(result["trace"], ["CLASSIFIER", "PARAMETER", "END"])

    def chat(self, **kwargs):
        return self.client.post('/api/chat', json={"config": self.name, "message": "test", **kwargs}).json()

    def test_both_question_routes_are_read_only(self):
        for route in ("QUESTION_GENERAL", "QUESTION_PARAMETER"):
            with patch.object(service, "structured", return_value={"route": route, "changes": {"epochs": 999}}), patch.object(service, "complete", return_value="Explanation"):
                result = self.chat()
            self.assertEqual(result["route"], route)
            self.assertEqual(result["changes"], {})
            self.assertEqual(result["reply"], "Explanation")

    def test_explicit_question_skips_classifier(self):
        with patch.object(service, "structured") as classifier, patch.object(service, "complete", return_value="Answer"):
            result = self.chat(route="QUESTION_GENERAL")
        classifier.assert_not_called()
        self.assertEqual(result["trace"], ["QUESTION_GENERAL", "END"])

    def test_generation_dispatches_validated_parameters(self):
        with patch.object(service, "structured", return_value={"route":"GENERATION", "changes":{"epochs":2}}), patch.object(service, "train", return_value={"id":"test", "status":"queued"}) as train:
            result = self.chat(values={"num_layers":3})
        self.assertEqual(train.call_args.args[0].values, {"num_layers":3,"epochs":2})
        self.assertEqual(result["route"], "GENERATION")

    def test_analysis_uses_explicit_run_and_comparison(self):
        with patch.object(service, "report", return_value={"markdown":"report"}) as report:
            result = self.chat(route="ANALYSIS",job_id="one",compare_job_id="two")
        self.assertEqual(report.call_args.args[0].job_id,"one")
        self.assertEqual(report.call_args.args[0].compare_job_id,"two")
        self.assertEqual(result["route"], "ANALYSIS")

    def test_bad_model_outputs_never_launch_training(self):
        invalid=[{"route":"NOPE"},{"route":"PARAMETER","changes":{}},
                 {"route":"GENERATION","changes":{"epochs":False}},
                 {"route":"PLANNER","steps":[{"route":"PLANNER"}]},
                 {"route":"PLANNER","steps":[{"route":"PARAMETER","changes":{}}]}]
        for value in invalid:
            with patch.object(service,"structured",return_value=value),patch.object(service,"train") as train:
                result=self.chat()
            self.assertEqual(result["route"],"ERROR_HANDLER",value)
            train.assert_not_called()

    def test_missing_model_recovers_without_action(self):
        with patch.object(service,"structured",side_effect=HTTPException(503,"Ollama unavailable")):
            result=self.chat()
        self.assertTrue(result["error"])
        self.assertIn("recovery",result)

    def test_parameter_aliases_and_before_after(self):
        with patch.object(service,"structured",return_value={"route":"PARAMETER","changes":{"lr":.002}}):
            result=self.chat(values={"learning_rate":.001})
        self.assertEqual(result["diff"],[{"key":"learning_rate","before":.001,"after":.002}])

    def test_preflight_blocks_job_creation(self):
        with patch.object(service,"inspect_python",return_value={"ready":False,"issues":["missing torch"]}),patch.object(service.jobs,"create") as create:
            result=self.chat(route="GENERATION")
        create.assert_not_called()
        self.assertEqual(result["route"],"ERROR_HANDLER")
        self.assertIn("preflight",result["reply"])

    def test_planner_preserves_top_level_changes(self):
        with patch.object(service,"structured",return_value={"route":"PLANNER","changes":{"epochs":3},"steps":[{"route":"QUESTION_PARAMETER","instruction":"Explain parameters"}]}):
            result=self.chat()
        plan=result["plan"]
        self.assertEqual(plan["steps"][0]["route"],"PARAMETER")
        self.assertEqual(plan["steps"][0]["changes"],{"epochs":3})
        service.PLANS.pop(plan["id"])

    def test_multiple_parameter_edits_are_one_action_unless_explicit_plan(self):
        parsed={"route":"PLANNER","steps":[{"route":"PARAMETER","changes":{"epochs":3}},
                                              {"route":"PARAMETER","changes":{"num_layers":3}}]}
        with patch.object(service,"structured",return_value=parsed):
            result=self.chat()
            explicit=self.chat(route="PLANNER")
        self.assertEqual(result['route'],'PARAMETER')
        self.assertEqual(result['changes'],{'epochs':3,'num_layers':3})
        self.assertEqual(explicit['route'],'PLANNER')
        service.PLANS.pop(explicit['plan']['id'])

    def test_dialogue_context_reaches_model(self):
        history=[{'role':'user','content':'Explanationnum_layers'},{'role':'assistant','content':'currently 12 layers'}]
        with patch.object(service,'structured',return_value={'route':'PARAMETER','changes':{'num_layers':3}}) as model:
            result=self.chat(message='then set it to 3',history=history)
        self.assertIn('Explanationnum_layers',model.call_args.args[1])
        self.assertEqual(result['changes'],{'num_layers':3})

    def test_plan_completes_and_does_not_repeat_side_effects(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(service,"jobs",JobManager(Path(temp))):
            with patch.object(service,"structured",return_value={"route":"PLANNER","steps":[
                {"route":"PARAMETER","changes":{"epochs":2}},
                {"route":"GENERATION"},{"route":"ANALYSIS"}]}):
                plan=self.chat()["plan"]
            url=f"/api/plans/{plan['id']}/advance"
            self.client.post(url,json={})
            job=service.jobs.create("training")
            with patch.object(service,"train",return_value=job) as train:
                self.client.post(url,json={})
                self.client.post(url,json={})
            self.assertEqual(train.call_count,1)
            self.assertEqual(train.call_args.args[0].values["epochs"],2)
            job["status"]="completed"
            with patch.object(service,"report",return_value={"markdown":"ok"}) as report:
                result=self.client.post(url,json={}).json()
                again=self.client.post(url,json={}).json()
            self.assertEqual(report.call_count,1)
            self.assertEqual(report.call_args.args[0].job_id,job["id"])
            self.assertEqual(result["plan"]["status"],"completed")
            self.assertEqual(again["next"],"END")
            service.PLANS.pop(plan["id"])

    def test_cancel_plan_does_not_cancel_existing_job(self):
        with patch.object(service,"structured",return_value={"route":"PLANNER","steps":[{"route":"GENERATION"}]}):
            plan=self.chat()["plan"]
        with patch.object(service.jobs,"cancel") as cancel:
            result=self.client.post(f"/api/plans/{plan['id']}/cancel",json={}).json()
            advance=self.client.post(f"/api/plans/{plan['id']}/advance",json={}).json()
        cancel.assert_not_called()
        self.assertEqual(result["plan"]["status"],"cancelled")
        self.assertEqual(advance["route"],"ERROR_HANDLER")
        service.PLANS.pop(plan["id"])

    def test_concurrent_plan_step_rejected(self):
        with patch.object(service,"structured",return_value={"route":"PLANNER","steps":[{"route":"GENERATION"}]}):
            plan=self.chat()["plan"]
        service.PLANS[plan['id']]['status']='executing'
        result=self.client.post(f"/api/plans/{plan['id']}/advance",json={})
        self.assertEqual(result.status_code,409)
        service.PLANS.pop(plan['id'])

    def test_report_survives_model_outage_with_explicit_warning(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(service,"RUNS",Path(temp)),patch.object(service,"structured",side_effect=HTTPException(503,"offline")):
            result=self.client.post('/api/report',json={"case":"case39"}).json()
            self.assertIn('Bus count | 39',result['markdown'])
            self.assertEqual(result['interpretation_status'],'unavailable')
            self.assertIn('offline',result['warning'])
            self.assertEqual(self.client.get(result['download']).status_code,200)

    def test_plan_waits_for_training_before_analysis(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(service, "jobs", JobManager(Path(temp))):
            with patch.object(service, "structured", return_value={"route": "PLANNER", "changes": {}, "steps": [
                {"route": "GENERATION", "instruction": "train", "changes": {}},
                {"route": "ANALYSIS", "instruction": "analyze", "changes": {}}]}):
                plan = self.client.post("/api/chat", json={"config": self.name, "message": "train then analyze"}).json()["plan"]
            job = service.jobs.create("training")
            with patch.object(service, "train", return_value=job):
                first = self.client.post(f"/api/plans/{plan['id']}/advance", json={}).json()
            self.assertIn("ADVANCE_STEP", first["trace"])
            waiting = self.client.post(f"/api/plans/{plan['id']}/advance", json={}).json()
            self.assertEqual(waiting["next"], "WAIT")
            job["status"] = "failed"
            result = self.client.post(f"/api/plans/{plan['id']}/advance", json={}).json()
            self.assertEqual(result["route"], "ERROR_HANDLER")
            service.PLANS.pop(plan["id"])

    def test_evidence_scans_beyond_first_five_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"metric.csv"
            path.write_text("loss\n"+"\n".join(map(str,range(10))))
            result = summarize_csv(path)
            self.assertEqual(result["rows_scanned"], 10)
            self.assertEqual(result["numeric_summary"]["loss"]["mean"], 4.5)

    def test_prediction_wrong_bus_ids_rejected(self):
        r = self.client.post("/api/predictions", json={"case":"case39","csv_text":"bus_id,vm_pu,va_deg\n999,1.0,0"})
        self.assertEqual(r.status_code,400)

    def test_report_numerical_table_comes_from_evidence(self):
        payload = compact({"primary": {"case":"case39","n_bus":39,"vm_min_pu":.982},
                           "comparison": None, "provenance":"DataKit saved", "source":"case39/summary.json"})
        text = render(payload, [])
        self.assertIn("Bus count | 39", text)
        self.assertIn("Minimum voltage magnitude | 0.982", text)
        self.assertIn("DataKit", text)

    def test_report_accepts_rounded_cited_numbers_but_not_new_numbers(self):
        values={"primary.loss":{"value":43.641126}}
        good={"evidence":["measurements.primary.loss"],"text":"Loss is 43.64 MW."}
        bad={"evidence":["primary.loss"],"text":"Loss is 50 MW."}
        unknown={"evidence":["invented.loss"],"text":"Loss is lower."}
        result=validate_observations([good,bad,unknown],values)
        self.assertEqual(len(result),1)
        self.assertEqual(result[0]['evidence'],['primary.loss'])

    def test_report_rejects_unsupported_qualitative_rankings(self):
        values={"primary.seconds":{"value":0.0016}}
        claim={"evidence":["primary.seconds"],"text":"The result is very fast at 0.0016 s."}
        self.assertEqual(validate_observations([claim],values),[])

    def test_job_records_survive_restart(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = JobManager(Path(temp))
            entry = manager.create("training")
            restored = JobManager(Path(temp))
            self.assertEqual(restored.get(entry["id"])["status"], "interrupted")

if __name__ == "__main__":
    unittest.main()
