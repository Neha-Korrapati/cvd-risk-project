"""Minimal FHIR R4 REST client used by the desktop app and the upload step."""
import requests

from . import fhir_mapping

DEFAULT_BASE = "http://127.0.0.1:8080/fhir"
HEADERS = {"Accept": "application/fhir+json", "Content-Type": "application/fhir+json"}


class FhirClient:
    def __init__(self, base_url: str = DEFAULT_BASE, timeout: float = 10):
        self.base = base_url.rstrip("/")
        self.timeout = timeout

    def _get(self, path: str, **params) -> dict:
        r = requests.get(f"{self.base}/{path}", params=params, headers=HEADERS, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def ping(self) -> str:
        """Return the server's FHIR version; raises if unreachable."""
        return self._get("metadata")["fhirVersion"]

    def search_patients(self, name: str = "", count: int = 500) -> list[dict]:
        params = {"_count": count}
        if name:
            params["name"] = name
        return [e["resource"] for e in self._get("Patient", **params).get("entry", [])]

    def patient(self, pid: str) -> dict:
        return self._get(f"Patient/{pid}")

    def observations(self, pid: str) -> list[dict]:
        return [e["resource"] for e in self._get("Observation", patient=pid).get("entry", [])]

    def patient_inputs(self, pid: str) -> tuple[dict, dict, list[dict]]:
        """(model inputs, Patient, Observations) for one patient."""
        patient, obs = self.patient(pid), self.observations(pid)
        return fhir_mapping.patient_inputs(patient, obs), patient, obs

    def transaction(self, bundle: dict) -> dict:
        r = requests.post(self.base, json=bundle, headers=HEADERS, timeout=120)
        r.raise_for_status()
        return r.json()
