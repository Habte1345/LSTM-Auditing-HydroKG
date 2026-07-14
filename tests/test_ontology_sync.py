"""Guards against the Python schema constants (hydrokg/graph/schema.py) silently drifting
from the human-readable ontology (hydrokg/ontology/hydrokg_ontology.ttl). Uses plain text
matching rather than a full RDF parse to avoid adding rdflib as a hard dependency for
something this simple; if the ontology grows more complex, swap to rdflib.Graph().parse().
"""

from pathlib import Path

from hydrokg.graph.schema import RULE_IDS, RULE_METADATA

_TTL_PATH = Path(__file__).resolve().parents[1] / "hydrokg" / "ontology" / "hydrokg_ontology.ttl"


def test_all_rule_ids_declared_in_ontology():
    ttl_text = _TTL_PATH.read_text()
    for rule_id in RULE_IDS:
        assert f"hkg:{rule_id} a hkg:Rule" in ttl_text, f"{rule_id} missing from ontology"


def test_rule_names_match_ontology():
    ttl_text = _TTL_PATH.read_text()
    for rule_id, meta in RULE_METADATA.items():
        assert f'hkg:ruleName "{meta["name"]}"' in ttl_text, (
            f"Name mismatch for {rule_id}: schema.py says '{meta['name']}'"
        )


def test_violation_classes_declared():
    ttl_text = _TTL_PATH.read_text()
    for meta in RULE_METADATA.values():
        assert f"hkg:{meta['violation_class']} a hkg:ViolationClass" in ttl_text
