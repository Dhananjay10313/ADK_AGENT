"""
Tool for adding a document to a Vertex AI RAG corpus and extracting a list of features.

This mirrors the style and conventions of add_data.py:
- Validates inputs (Drive/GCS), normalizes Google Docs/Drive links
- Uses rag.import_files with chunking config
- Updates the "current_corpus" in ToolContext state
- Returns a structured result dict
"""

import json
import re
import time
from typing import List, Dict, Any

import vertexai
from vertexai import rag
from vertexai.generative_models import GenerativeModel, GenerationConfig

from google.adk.tools.tool_context import ToolContext

from ..config import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_EMBEDDING_REQUESTS_PER_MIN,  # optional if you keep centralized init elsewhere
)

from .utils import check_corpus_exists, get_corpus_resource_name


def _normalize_paths(paths: List[str]) -> Dict[str, List[str]]:
    """
    Validate/normalize Drive/Docs and GCS URLs similar to add_data.py.
    Returns dict with validated_paths, invalid_paths, conversions.
    """
    validated_paths: List[str] = []
    invalid_paths: List[str] = []
    conversions: List[str] = []

    for path in paths:
        if not path or not isinstance(path, str):
            invalid_paths.append(f"{path} (Not a valid string)")
            continue

        # Convert Docs/Sheets/Slides -> Drive file URL
        docs_match = re.match(
            r"https:\/\/docs\.google\.com\/(?:document|spreadsheets|presentation)\/d\/([a-zA-Z0-9_-]+)(?:\/|$)",
            path,
        )
        if docs_match:
            file_id = docs_match.group(1)
            drive_url = f"https://drive.google.com/file/d/{file_id}/view"
            validated_paths.append(drive_url)
            conversions.append(f"{path} → {drive_url}")
            continue

        # Normalize Drive URLs to a canonical form
        drive_match = re.match(
            r"https:\/\/drive\.google\.com\/(?:file\/d\/|open\?id=)([a-zA-Z0-9_-]+)(?:\/|$)",
            path,
        )
        if drive_match:
            file_id = drive_match.group(1)
            drive_url = f"https://drive.google.com/file/d/{file_id}/view"
            validated_paths.append(drive_url)
            if drive_url != path:
                conversions.append(f"{path} → {drive_url}")
            continue

        # Allow GCS paths
        if path.startswith("gs://"):
            validated_paths.append(path)
            continue

        invalid_paths.append(f"{path} (Invalid format)")

    return {
        "validated_paths": validated_paths,
        "invalid_paths": invalid_paths,
        "conversions": conversions,
    }


def _extract_features_with_gemini(model_name: str, context_text: str) -> List[str]:
    """
    Deterministic feature extraction from provided text using Gemini.
    Forces JSON list output to simplify downstream use.
    """
    model = GenerativeModel(model_name)
    prompt = (
        "Extract a list of distinct product or system features explicitly described in the provided text.\n"
        "Return a flat JSON array of short feature names (strings), no objects, no extra keys, no explanations.\n"
        "Do not infer or hallucinate; include only features directly supported by the text.\n"
    )
    cfg = GenerationConfig(
        temperature=0.1,
        top_p=0.9,
        top_k=40,
        max_output_tokens=512,
        response_mime_type="application/json",
    )
    resp = model.generate_content([prompt, context_text], generation_config=cfg)
    raw = getattr(resp, "text", "") or ""
    # Lenient JSON parse
    try:
        data = json.loads(raw)
        if isinstance(data, list) and all(isinstance(x, str) for x in data):
            return data
    except Exception:
        pass
    # Fallback: attempt to find a JSON array
    m = re.search(r"\[(?:.|\n)*\]", raw)
    if m:
        try:
            data = json.loads(m.group(0))
            if isinstance(data, list) and all(isinstance(x, str) for x in data):
                return data
        except Exception:
            pass
    # Final fallback
    return []


def upload_and_extract(
    corpus_name: str,
    paths: List[str],
    tool_context: ToolContext,
    *,
    model_for_extraction: str = "gemini-2.0-flash",
    retrieval_top_k: int = 12,
    max_wait_seconds_for_index: int = 45,
) -> dict:
    """
    Add documents to a Vertex AI RAG corpus and extract a list of features.

    Args:
      corpus_name: Target corpus display name (required; if empty, current corpus will be used if set).
      paths: Google Drive or GCS paths to import. Supported:
             - Drive file URL (https://drive.google.com/file/d/{FILE_ID}/view)
             - Docs/Sheets/Slides URL (auto-normalized)
             - GCS path (gs://bucket/path)
      tool_context: ADK ToolContext
      model_for_extraction: Gemini model to run the feature extraction
      retrieval_top_k: Number of chunks to retrieve for extraction context
      max_wait_seconds_for_index: Best-effort wait for the index to become queryable

    Returns:
      dict with keys:
        - status: "success" | "error"
        - message: summary string
        - corpus_name: display name
        - files_added: int
        - paths: normalized/validated paths
        - invalid_paths: list of invalid inputs
        - conversions: list of URL normalizations performed
        - features: list[str] extracted features
        - debug: optional retrieval diagnostics
    """

    # Optional: initialize Vertex AI here if central init is not used elsewhere
    # try:
    #     if DEFAULT_VERTEX_PROJECT and DEFAULT_VERTEX_LOCATION:
    #         vertexai.init(project=DEFAULT_VERTEX_PROJECT, location=DEFAULT_VERTEX_LOCATION)
    # except Exception:
    #     # Safe to ignore if init is already handled by the app
    #     pass

    # Resolve corpus fallback from state
    if not corpus_name:
        corpus_name = tool_context.state.get("current_corpus") or ""

    # Existence check (same pattern as add_data.py)
    if not check_corpus_exists(corpus_name, tool_context):
        return {
            "status": "error",
            "message": (
                f"Corpus '{corpus_name}' does not exist. Create it first, "
                "or pass a valid corpus_name."
            ),
            "corpus_name": corpus_name,
            "paths": paths,
        }

    # Validate inputs
    if not paths or not all(isinstance(p, str) for p in paths):
        return {
            "status": "error",
            "message": "Invalid paths: provide a list of Drive URLs or GCS paths",
            "corpus_name": corpus_name,
            "paths": paths,
        }

    norm = _normalize_paths(paths)
    validated_paths = norm["validated_paths"]
    invalid_paths = norm["invalid_paths"]
    conversions = norm["conversions"]

    if not validated_paths:
        return {
            "status": "error",
            "message": "No valid paths provided. Please provide Google Drive URLs or GCS paths.",
            "corpus_name": corpus_name,
            "invalid_paths": invalid_paths,
        }

    try:
        # Get the corpus resource name
        corpus_resource_name = get_corpus_resource_name(corpus_name)

        # Chunking config for ingestion (same shape as add_data.py)
        transformation_config = rag.TransformationConfig(
            chunking_config=rag.ChunkingConfig(
                chunk_size=DEFAULT_CHUNK_SIZE,
                chunk_overlap=DEFAULT_CHUNK_OVERLAP,
            ),
        )

        # Import files into the corpus
        import_result = rag.import_files(
            corpus_name=corpus_resource_name,
            paths=validated_paths,
            transformation_config=transformation_config,
            max_embedding_requests_per_min=DEFAULT_EMBEDDING_REQUESTS_PER_MIN,
            # Optional sink if you want import metadata in GCS:
            # import_result_sink="gs://your-bucket/path/import_result_<unique>.ndjson",
        )

        # Maintain current corpus in state
        if not tool_context.state.get("current_corpus"):
            tool_context.state["current_corpus"] = corpus_name

        # Best-effort wait to ensure retrieval becomes available right after import
        # (In some environments, indexing/embedding may be slightly eventual.)
        waited = 0
        sleep_s = 3
        while waited < max_wait_seconds_for_index:
            try:
                # Simple no-op retrieval to warm things up; break on first success
                _ = rag.retrieval_query(
                    rag_resources=[rag.RagResource(rag_corpus=corpus_resource_name)],
                    text="Warm up retrieval",
                    rag_retrieval_config=rag.RagRetrievalConfig(top_k=1),
                )
                break
            except Exception:
                time.sleep(sleep_s)
                waited += sleep_s

        # Retrieve contexts to ground feature extraction to the newly added content
        # Note: Without per-file filters, RAG returns the most relevant chunks; keep top_k modest and prompt the extractor to use only provided context.
        retrieval = rag.retrieval_query(
            rag_resources=[rag.RagResource(rag_corpus=corpus_resource_name)],
            text="Collect the main product/system features described in the newly added documents.",
            rag_retrieval_config=rag.RagRetrievalConfig(top_k=retrieval_top_k),
        )

        contexts = []
        for ctx in getattr(retrieval, "contexts", []):
            # Each ctx typically has `text` and `source_uri`; include both for traceability
            text = getattr(ctx, "text", "") or ""
            src = getattr(ctx, "source_uri", "") or ""
            if text:
                contexts.append({"text": text, "source_uri": src})

        combined_text = "\n\n".join(c["text"] for c in contexts)[:200_000]  # safeguard upper bound
        features = _extract_features_with_gemini(model_for_extraction, combined_text) if combined_text else []

        conversion_msg = " (Converted Google Docs URLs to Drive format)" if conversions else ""
        return {
            "status": "success",
            "message": (
                f"Imported {import_result.imported_rag_files_count} file(s) to corpus "
                f"'{corpus_name}'{conversion_msg}; extracted {len(features)} feature(s)."
            ),
            "corpus_name": corpus_name,
            "files_added": getattr(import_result, "imported_rag_files_count", 0),
            "paths": validated_paths,
            "invalid_paths": invalid_paths,
            "conversions": conversions,
            "features": features,
            "debug": {
                "retrieval_top_k": retrieval_top_k,
                "context_snippets": contexts[:5],  # limited preview for debugging
            },
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Error during import or feature extraction: {str(e)}",
            "corpus_name": corpus_name,
            "paths": paths,
        }
