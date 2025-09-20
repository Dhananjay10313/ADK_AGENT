# rag_query_multi.py
"""
Multi-corpus RAG query tool for Vertex AI RAG Engine.
Drop-in compatible with your existing patterns; adds support for multiple corpora.
"""

import logging
from typing import List, Dict, Any

from google.adk.tools.tool_context import ToolContext
from vertexai import rag

from ..config import (
    DEFAULT_DISTANCE_THRESHOLD,
    DEFAULT_TOP_K,
)

from .utils import check_corpus_exists, get_corpus_resource_name


def rag_query(
    corpora: List[str],  # display names; may be empty to use current_corpus
    query: str,
    tool_context: ToolContext,
) -> Dict[str, Any]:
    """
    Query one or more Vertex AI RAG corpora and return aggregated results.

    Args:
      corpora: List of corpus display names. If empty, uses tool_context.state["current_corpus"].
      query: User query text
      tool_context: ADK ToolContext

    Returns:
      dict: status, message, corpora, results, results_count
    """
    try:
        # Resolve default from state
        if not corpora:
            current = tool_context.state.get("current_corpus")
            corpora = [current] if current else []

        if not corpora:
            return {
                "status": "error",
                "message": "No corpus specified and no current corpus set.",
                "query": query,
                "corpora": corpora,
                "results": [],
                "results_count": 0,
            }

        # Validate and resolve resource names
        valid_display_names: List[str] = []
        resources: List[rag.RagResource] = []
        invalid: List[str] = []

        for name in corpora:
            if not check_corpus_exists(name, tool_context):
                invalid.append(name)
                continue
            rn = get_corpus_resource_name(name)
            valid_display_names.append(name)
            resources.append(rag.RagResource(rag_corpus=rn))

        if not resources:
            return {
                "status": "error",
                "message": f"No valid corpora found. Invalid: {invalid}",
                "query": query,
                "corpora": corpora,
                "results": [],
                "results_count": 0,
            }

        rag_retrieval_config = rag.RagRetrievalConfig(
            top_k=DEFAULT_TOP_K,
            filter=rag.Filter(vector_distance_threshold=DEFAULT_DISTANCE_THRESHOLD),
        )

        # Single retrieval across multiple corpora
        response = rag.retrieval_query(
            rag_resources=resources,
            text=query,
            rag_retrieval_config=rag_retrieval_config,
        )

        results: List[Dict[str, Any]] = []
        # Adapt to observed response shape in your current code
        # Prefer a robust parse that works if response.contexts is either a list or an object
        ctxs = getattr(response, "contexts", None)
        if ctxs:
            # Try iterable of context objects
            iterable = getattr(ctxs, "contexts", ctxs)
            for ctx in iterable:
                results.append({
                    "source_uri": getattr(ctx, "source_uri", "") or "",
                    "source_name": getattr(ctx, "source_display_name", "") or "",
                    "text": getattr(ctx, "text", "") or "",
                    "score": getattr(ctx, "score", 0.0) or 0.0,
                })

        if not results:
            return {
                "status": "warning",
                "message": f"No results found in corpora {valid_display_names} for query.",
                "query": query,
                "corpora": valid_display_names,
                "invalid_corpora": invalid,
                "results": [],
                "results_count": 0,
            }

        return {
            "status": "success",
            "message": f"Successfully queried corpora {valid_display_names}.",
            "query": query,
            "corpora": valid_display_names,
            "invalid_corpora": invalid,
            "results": results,
            "results_count": len(results),
        }

    except Exception as e:
        logging.error("Multi-corpus query error: %s", e)
        return {
            "status": "error",
            "message": f"Error querying corpora: {str(e)}",
            "query": query,
            "corpora": corpora,
            "results": [],
            "results_count": 0,
        }
