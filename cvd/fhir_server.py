"""A small FHIR R4 REST server standing in for the hospital EHR's FHIR endpoint.

It implements the subset of the FHIR RESTful API the application needs:

    GET  /fhir/metadata                      CapabilityStatement
    POST /fhir                               transaction Bundle (PUT entries)
    GET  /fhir/{type}/{id}                   read
    PUT  /fhir/{type}/{id}                   create / update
    GET  /fhir/Patient?name=&_count=         search (searchset Bundle)
    GET  /fhir/Observation?patient={id}      search (also subject=Patient/{id})

Because it follows the standard API, the desktop app and the upload script work
unchanged against a full server such as HAPI FHIR; only the base URL changes.
This server has no authentication (no SMART-on-FHIR OAuth; DEVIATIONS.md P-02).

Run from the project root:
    .venv\\Scripts\\python -m cvd.fhir_server            (http://127.0.0.1:8080/fhir)
"""
import argparse
import json
import threading
import uuid
from datetime import datetime, timezone

from flask import Flask, Response, request

from . import config as C

STORE_PATH = C.ROOT / "fhir" / "store.json"
FHIR_JSON = "application/fhir+json"
SUPPORTED = ("Patient", "Observation")

app = Flask(__name__)
_lock = threading.Lock()
_store: dict[str, dict[str, dict]] = {}


def _load() -> None:
    global _store
    _store = json.loads(STORE_PATH.read_text()) if STORE_PATH.exists() else {}


def _save() -> None:
    STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STORE_PATH.write_text(json.dumps(_store))


def _json(body: dict, status: int = 200) -> Response:
    return Response(json.dumps(body), status=status, mimetype=FHIR_JSON)


def _outcome(status: int, text: str, code: str = "processing") -> Response:
    return _json({"resourceType": "OperationOutcome",
                  "issue": [{"severity": "error", "code": code, "diagnostics": text}]}, status)


def _put(rtype: str, rid: str, resource: dict) -> tuple[dict, bool]:
    bucket = _store.setdefault(rtype, {})
    created = rid not in bucket
    version = 1 if created else int(bucket[rid].get("meta", {}).get("versionId", "1")) + 1
    resource = dict(resource, id=rid, resourceType=rtype)
    resource["meta"] = {"versionId": str(version),
                        "lastUpdated": datetime.now(timezone.utc).isoformat()}
    bucket[rid] = resource
    return resource, created


def _searchset(resources: list[dict]) -> dict:
    base = request.host_url.rstrip("/") + "/fhir"
    return {"resourceType": "Bundle", "id": str(uuid.uuid4()), "type": "searchset",
            "total": len(resources),
            "entry": [{"fullUrl": f"{base}/{r['resourceType']}/{r['id']}", "resource": r,
                       "search": {"mode": "match"}} for r in resources]}


@app.get("/fhir/metadata")
def metadata():
    return _json({
        "resourceType": "CapabilityStatement", "status": "active", "kind": "instance",
        "fhirVersion": "4.0.1", "format": ["json"],
        "date": datetime.now(timezone.utc).date().isoformat(),
        "software": {"name": "cvd-risk-project local FHIR server"},
        "rest": [{"mode": "server", "resource": [
            {"type": t, "interaction": [{"code": "read"}, {"code": "update"}, {"code": "search-type"}]}
            for t in SUPPORTED]}],
    })


@app.post("/fhir")
def transaction():
    bundle = request.get_json(force=True, silent=True)
    if not bundle or bundle.get("resourceType") != "Bundle" or bundle.get("type") != "transaction":
        return _outcome(400, "Expected a transaction Bundle", "invalid")
    responses = []
    with _lock:
        for entry in bundle.get("entry", []):
            req, res = entry.get("request", {}), entry.get("resource", {})
            parts = req.get("url", "").split("/")
            if req.get("method") != "PUT" or len(parts) != 2 or parts[0] not in SUPPORTED:
                return _outcome(400, f"Unsupported entry: {req}", "not-supported")
            stored, created = _put(parts[0], parts[1], res)
            responses.append({"response": {
                "status": "201 Created" if created else "200 OK",
                "location": f"{parts[0]}/{parts[1]}/_history/{stored['meta']['versionId']}"}})
        _save()
    return _json({"resourceType": "Bundle", "type": "transaction-response", "entry": responses})


@app.get("/fhir/<rtype>/<rid>")
def read(rtype, rid):
    r = _store.get(rtype, {}).get(rid)
    return _json(r) if r else _outcome(404, f"{rtype}/{rid} not found", "not-found")


@app.put("/fhir/<rtype>/<rid>")
def update(rtype, rid):
    if rtype not in SUPPORTED:
        return _outcome(400, f"{rtype} not supported", "not-supported")
    body = request.get_json(force=True, silent=True) or {}
    with _lock:
        stored, created = _put(rtype, rid, body)
        _save()
    return _json(stored, 201 if created else 200)


@app.get("/fhir/Patient")
def search_patients():
    name = request.args.get("name", "").lower()
    count = int(request.args.get("_count", 1000))
    hits = []
    for p in _store.get("Patient", {}).values():
        text = " ".join(" ".join(n.get("given", [])) + " " + n.get("family", "")
                        for n in p.get("name", [])).lower()
        if name in text or name in p["id"].lower():
            hits.append(p)
    hits.sort(key=lambda p: p["id"])
    return _json(_searchset(hits[:count]))


@app.get("/fhir/Observation")
def search_observations():
    pid = request.args.get("patient") or request.args.get("subject", "")
    pid = pid.split("/")[-1]
    hits = [o for o in _store.get("Observation", {}).values()
            if o.get("subject", {}).get("reference", "").split("/")[-1] == pid]
    return _json(_searchset(hits))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()
    _load()
    print(f"FHIR server on http://{args.host}:{args.port}/fhir "
          f"({sum(len(v) for v in _store.values())} resources loaded)", flush=True)
    app.run(host=args.host, port=args.port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
