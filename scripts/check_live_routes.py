"""Opt-in local Ollama/API smoke test. Never starts training: probes first.

Run against a locally running Studio: python scripts/check_live_routes.py
Writes only reports and plans through the application, plus a validation record.
"""
import json
import time
from pathlib import Path
from urllib.request import Request, urlopen

BASE = "http://127.0.0.1:8017"

def api(path, body=None):
    req = Request(BASE+path, data=json.dumps(body).encode() if body is not None else None,
                  headers={"Content-Type":"application/json"})
    with urlopen(req, timeout=300) as response:
        return json.load(response)

def main():
    config = "hello_world_train_case14.yaml"
    check = api('/api/train/preflight',{'config':config})
    if check['ready']:
        raise SystemExit('This classifier smoke test requires an unready training environment to prevent accidental training. Use unit tests for dispatch validation.')
    records = []
    messages = [
        ("QUESTION_GENERAL", "What is GENCO's overall architecture? Explain in two sentences."),
        ("QUESTION_PARAMETER", "Where does num_layers act? Answer briefly without changing it."),
        ("PARAMETER", "Set epochs to 3 and num_layers to 3."),
        ("ANALYSIS", "Generate an analysis report for the saved case39 results."),
        ("PLANNER", "Set epochs to 2, then explain num_layers. Do not train."),
    ]
    for expected, message in messages:
        start = time.monotonic()
        result = api("/api/chat", {"config":config,"message":message})
        record = {"expected":expected,"actual":result.get("route"),"seconds":round(time.monotonic()-start,2),"result":result}
        records.append(record)
        print(expected, '=>', record['actual'], record['seconds'], flush=True)
        if result.get("plan"):
            plan = result['plan']
            # Only execute verified non-training smoke steps.
            if all(s['route'] in ('PARAMETER','QUESTION_PARAMETER','QUESTION_GENERAL') for s in plan['steps']):
                record['steps'] = [api('/api/plans/'+plan['id']+'/advance',{}) for _ in plan['steps']]
            else:
                record['cancelled'] = api('/api/plans/'+plan['id']+'/cancel',{})
    check = api('/api/train/preflight',{'config':config})
    if not check['ready']:
        result = api('/api/chat',{'config':config,'message':'Start training now using the current parameters.'})
        records.append({'expected':'GENERATION preflight rejection','actual':result.get('route'),'result':result})
        print('GENERATION preflight =>',result.get('route'),flush=True)
    else:
        records.append({'expected':'GENERATION','actual':'skipped','reason':'Training environment ready; smoke test does not launch expensive jobs.'})
    target = Path(__file__).resolve().parents[1]/'runs'/'route-validation.json'
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps({'preflight':check,'records':records},ensure_ascii=False,indent=2))
    print(target,flush=True)
    failed = [r['expected'] for r in records if r['expected'] in dict(messages) and r['actual'] != r['expected']]
    if failed:
        raise SystemExit('Unexpected route: '+', '.join(failed))

if __name__ == '__main__':
    main()
