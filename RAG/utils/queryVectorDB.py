import os
import re
from typing import Any, Dict, Optional, Tuple, List
from dotenv import load_dotenv
from langchain_community.graphs import Neo4jGraph
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sentence_transformers import CrossEncoder

load_dotenv()


# Load model Rerank
reranker = CrossEncoder(os.getenv('ENCODE_MODEL'))

graph = Neo4jGraph(
    url=os.getenv('NEO4J_URI'),
    username=os.getenv('NEO4J_USERNAME'),
    password=os.getenv('NEO4J_PASSWORD')
)
print("Connected successfully!")

def _build_filter(kvs: Optional[Dict[str, Any]]) -> Optional[Filter]:
    if not kvs:
        return None
    return Filter(must=[FieldCondition(key=k, match=MatchValue(value=v)) for k, v in kvs.items()])

def _gather_case_sections(
    client: QdrantClient,
    collection: str,
    case_id: str,
    want_sections: Tuple[str, ...] = (
        "Disease Name Short",
        "Final Diagnosis",
        "Vitals",
    ),
) -> Dict[str, Optional[str]]:
    """Scroll all points for a case and extract the text from desired sections."""
    out = {sec: None for sec in want_sections}
    f = Filter(must=[FieldCondition(key="case", match=MatchValue(value=case_id))])

    # fetch a small page; most cases won't have many chunks
    points, _ = client.scroll(
        collection_name=collection,
        scroll_filter=f,
        with_payload=True,
        with_vectors=False,
        limit=128,
    )

    for p in points or []:
        payload = p.payload or {}
        sec = payload.get("section")
        if sec in out and out[sec] is None:
            # these section points carry their value in payload["text"]
            out[sec] = payload.get("text")

        # short-circuit if we already found everything
        if all(v is not None for v in out.values()):
            break

    return out

# ---- main function -----------------------------------------------------------

def search_vectors(
    query_text_or_image,
    client: QdrantClient,
    collection_name: str = "nomic_text_vectors",
    embedding_fn=None,
    top_k: int = 5,
    filters: Optional[Dict[str, Any]] = None,
    using_vector: str = "nomic-embed-text",
    verbose: bool = True,
):
    """
    Vector search + label enrichment.

    Returns: list[ScoredPoint] where each .payload is augmented with:
        - 'Disease Name Short': str | None
        - 'Final Diagnosis'   : str | None
        - 'Vitals': str | None
    """
    if embedding_fn is None:
        raise ValueError("embedding_fn is required")


    query_vector = embedding_fn(query_text_or_image)

    # 2) (optional) payload filter during search
    qdrant_filter = _build_filter(filters)

    # 3) run vector search (prefer query_points, fallback to search)
    try:
        qp = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=top_k,
            query_filter=qdrant_filter,
            using=using_vector,
            with_payload=True,
            with_vectors=False,
        )
        results = list(qp.points)
    except Exception as e:
        if verbose:
            print(f"Query error: {e}")
        raise

    # 4) enrich with diagnosis fields from same case (one scroll per case)
    cases = { (r.payload or {}).get("case") for r in results or [] }
    cases.discard(None)

    case_labels: Dict[str, Dict[str, Optional[str]]] = {}
    for case_id in cases:
        case_labels[case_id] = _gather_case_sections(client, collection_name, case_id)

    for r in results or []:
        payload = r.payload or {}
        case_id = payload.get("case")
        if case_id and case_id in case_labels:
            labels = case_labels[case_id]
            # inject/overwrite keys; keep it explicit
            if labels.get("Disease Name Short") is not None:
                payload["Disease Name Short"] = labels["Disease Name Short"]
            if labels.get("Final Diagnosis") is not None:
                payload["Final Diagnosis"] = labels["Final Diagnosis"]
            if labels.get("Vitals") is not None:
                payload["Vitals"] = labels["Vitals"]
            r.payload = payload

    if verbose:
        print("\n🔎 Enriched results:\n")
        for i, r in enumerate(results or [], 1):
            py = r.payload or {}
            print(
                f"     Payload:{py}"
                f"{i}. ID: {getattr(r, 'id', None)}, Score: {getattr(r, 'score', None):.4f}\n"
                f"   case={py.get('case')}  section={py.get('section')}\n"
                f"   Disease Name Short={py.get('Disease Name Short')}\n"
                f"   Final Diagnosis={(py.get('Final Diagnosis') or '')[:120]}{'…' if py.get('Final Diagnosis') and len(py['Final Diagnosis'])>120 else ''}\n"
                f"   Vitals={(py.get('Vitals') or '')[:120]}{'…' if py.get('Vitals') and len(py['Vitals'])>120 else ''}\n"
                f"   Text: {py.get('text')}\n"
            )

    return results



def search_vectors_v2(
    query_text: str,
    client: QdrantClient,
    collection_name: str = "nomic_text_vectors",
    embedding_fn=None,
    top_k: int = 3,
    filters: Optional[Dict[str, Any]] = None,
    using_vector: str = "nomic-embed-text",
    verbose: bool = True,
    use_rerank: bool = True,
):
    if embedding_fn is None:
        raise ValueError("embedding_fn is required")

    # 1. Embed query
    query_vector = embedding_fn(query_text)
    qdrant_filter = _build_filter(filters)

    initial_k = top_k * 10 if use_rerank else top_k

    try:
        qp = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=initial_k, # <--- Lấy 50 kết quả
            query_filter=qdrant_filter,
            using=using_vector,
            with_payload=True,
            with_vectors=False,
        )
        results = list(qp.points)
    except Exception as e:
        if verbose: print(f"Query error: {e}")
        raise

    if use_rerank and results:
        if verbose: print(f"🔄 Reranking {len(results)} candidates...")


        pairs = [[query_text, (r.payload or {}).get("text", "")] for r in results]

        scores = reranker.predict(pairs)


        for i, r in enumerate(results):
            r.score = float(scores[i])

        results.sort(key=lambda x: x.score, reverse=True)

        results = results[:top_k]


    cases = { (r.payload or {}).get("case") for r in results or [] }
    cases.discard(None)

    case_labels: Dict[str, Dict[str, Optional[str]]] = {}
    for case_id in cases:
        case_labels[case_id] = _gather_case_sections(client, collection_name, case_id)

    for r in results or []:
        payload = r.payload or {}
        case_id = payload.get("case")
        if case_id and case_id in case_labels:
            labels = case_labels[case_id]
            for key in ["Disease Name Short", "Final Diagnosis", "Vitals"]:
                if labels.get(key) is not None:
                    payload[key] = labels[key]
            r.payload = payload

    if verbose:
        print(f"\n🔎 Final Results (Top {len(results)}):")
        for i, r in enumerate(results, 1):
            py = r.payload
            print(f"{i}. Score: {r.score:.4f} | Case: {py.get('case')} | Section: {py.get('section')}")
            print(f"   Text: {(py.get('text') or '')}...")
            if py.get('Final Diagnosis'):
                print(f"   [Enriched] Diagnosis: {py.get('Final Diagnosis')}...")
            print("-" * 40)

    return results


def get_cases_knowledge_graph(knowledgeGraph, case_list):
    # 1. Pre-process: Extract IDs and maintain order
    # targets: [91, 102, 91] (Integer IDs in requested order)
    ids = []

    for case_str in case_list:
        match = re.search(r'\d+', case_str)
        if match:
            cid = int(match.group())
            ids.append(cid)

    if not ids:
        return []

    query = """
    MATCH (c:Case)
    WHERE c.case_id IN $case_ids
    OPTIONAL MATCH (c)-[r:HAS_INFO|HAS_SYMPTOM|LOCATED_IN|OF_DISEASE]->(neighbor)
    RETURN 
        c.case_id AS id,
        collect(
            CASE type(r)
                WHEN 'HAS_SYMPTOM' THEN { label: 'Symptom', val: neighbor.name }
                WHEN 'LOCATED_IN'  THEN { label: 'Location', val: neighbor.name }
                WHEN 'OF_DISEASE'  THEN { label: 'Diagnosis', val: neighbor.name }
                WHEN 'HAS_INFO'    THEN { label: 'Details',   val: coalesce(neighbor.name, neighbor.content) }
                ELSE { label: 'General', val: coalesce(neighbor.name, neighbor.content) }
            END
        ) AS attributes
    """

    query_v2 = """
    MATCH (c:Case)
    WHERE c.case_id IN $case_ids
    OPTIONAL MATCH (c)-[r:HAS_INFO|HAS_SYMPTOM|LOCATED_IN|OF_DISEASE|HAS_RISK_FACTOR|HAS_INVESTIGATION|HAS_EPIDEMIOLOGY]->(neighbor)
    RETURN 
        c.case_id AS id,
        collect(
            CASE type(r)
                WHEN 'HAS_SYMPTOM'      THEN { label: 'Symptoms', val: neighbor.name }
                WHEN 'LOCATED_IN'       THEN { label: 'Locations', val: neighbor.name }
                WHEN 'OF_DISEASE'       THEN { label: 'Diagnosis', val: neighbor.name }
                WHEN 'HAS_INFO'         THEN { label: 'Details', val: coalesce(neighbor.content, neighbor.name) }
                WHEN 'HAS_EPIDEMIOLOGY' THEN { label: 'Epidemiology', val: coalesce(neighbor.content, neighbor.name) }
                WHEN 'HAS_INVESTIGATION' THEN { label: 'Investigations', val: coalesce(neighbor.content, neighbor.name) }
                WHEN 'HAS_RISK_FACTOR'  THEN { label: 'Risk Factors & Patient Profile', val: coalesce(neighbor.content, neighbor.name) }
                ELSE { label: 'General', val: coalesce(neighbor.content, neighbor.name) }
            END
        ) AS attributes
    """

    query_v3 = """
    MATCH (c:Case)
    // Match against either property name
    WHERE c.case_id IN $ids OR c.file_id IN $ids

    // Include all relationship types from both datasets
    OPTIONAL MATCH (c)-[r:HAS_INFO|HAS_SYMPTOM|LOCATED_IN|OF_DISEASE|HAS_RISK_FACTOR|HAS_INVESTIGATION|HAS_EPIDEMIOLOGY]->(neighbor)

    RETURN 
        coalesce(c.case_id, c.file_id) AS id,
        collect(
            CASE type(r)
                WHEN 'HAS_SYMPTOM'      THEN { label: 'Symptoms', val: neighbor.name }
                WHEN 'LOCATED_IN'       THEN { label: 'Locations', val: neighbor.name }
                WHEN 'OF_DISEASE'       THEN { label: 'Diagnosis', val: neighbor.name }
                WHEN 'HAS_INFO'         THEN { label: 'Details', val: coalesce(neighbor.content, neighbor.name) }
                WHEN 'HAS_EPIDEMIOLOGY' THEN { label: 'Epidemiology', val: coalesce(neighbor.content, neighbor.name) }
                WHEN 'HAS_INVESTIGATION' THEN { label: 'Investigations', val: coalesce(neighbor.content, neighbor.name) }
                WHEN 'HAS_RISK_FACTOR'  THEN { label: 'Risk Factors & Patient Profile', val: coalesce(neighbor.content, neighbor.name) }
                ELSE { label: 'General', val: coalesce(neighbor.content, neighbor.name) }
            END
        ) AS attributes
    """

    # We do NOT use ORDER BY in SQL anymore because we will order by Python list logic
    # results = graph.query(query_v2, {"case_ids": ids})
    results = knowledgeGraph.query(query_v3, {"ids": ids})
    # print(f"Results: {results}")

    # 3. Create a lookup dictionary: { 91: {data...}, 102: {data...} }
    results_lookup = {rec['id']: rec for rec in results}

    output_parts = []

    # 4. Iterate through the ORIGINAL target list to preserve input order and handle missing items
    for cid in ids:
        # Check if database returned data for this ID
        record = results_lookup.get(cid)
        if record:
            symptoms = [item['val'] for item in record['attributes'] if item.get('label') == 'Symptoms']
            locations = [item['val'] for item in record['attributes'] if item.get('label') == 'Locations']
            details = [item['val'] for item in record['attributes'] if item.get('label') == 'Details']
            diagnoses = [item['val'] for item in record['attributes'] if item.get('label') == 'Diagnosis']
            epidemiology = [item['val'] for item in record['attributes'] if item.get('label') == 'Epidemiology']
            investigations = [item['val'] for item in record['attributes'] if item.get('label') == 'Investigations']
            risk_factors_patient_profile = [item['val'] for item in record['attributes'] if
                                            item.get('label') == 'Risk Factors & Patient Profile']
            general = [item['val'] for item in record['attributes'] if item.get('label') == 'General']

            info = {
                "has_info": True,
                "case_id": cid,
                "symptoms": f"Symptoms: {', '.join(symptoms)}" if symptoms else None,
                "locations": f"Locations: {', '.join(locations)}" if locations else None,
                "diagnosis": f"Diagnosis: {', '.join(diagnoses)}" if diagnoses else None,
                "details": f"Details: {', '.join(details)}" if details else None,
                "epidemiology": f"Epidemiology: {', '.join(epidemiology)}" if epidemiology else None,
                "investigations": f"Investigations: {', '.join(investigations)}" if investigations else None,
                "risk_factors_patient_profile": f"Risk Factors & Patient Profile: {', '.join(risk_factors_patient_profile)}" if risk_factors_patient_profile else None,
                "general": f"General: {', '.join(general)}" if general else None,
            }
        else:
            # Case not found in DB: Append empty structure
            info = {
                "has_info": False
            }

        output_parts.append(info)

    return output_parts


async def process_item(hits: List[dict]):
    """
    This function is used to handle and reformat the result retrieved from the
    Naive RAG (Qdrant) if there are duplicated cases.
    """
    grouped_hits = {}
    for h in (hits or []):
        payload = h.get("payload", {})
        c_id = payload.get("case", "Unknown")
        context = h.get("context", "")

        if c_id == "Unknown": continue

        if c_id not in grouped_hits:
            grouped_hits[c_id] = []

        grouped_hits[c_id].append(context)

    found_cases_list = list(grouped_hits.keys())
    context_list = []

    # Iterate through each case to create a separate context block for it
    for c_id in found_cases_list:
        # Merge all chunks belonging to THIS specific case
        case_text = "\n---\n".join(grouped_hits[c_id])

        # Add the merged text as a single item in the list
        context_list.append(case_text)

    # Note: We do NOT join with "--" anymore. We return the list directly.
    return {
        "found_cases": found_cases_list,
        "context": context_list  # This is now specific: List[str]
    }