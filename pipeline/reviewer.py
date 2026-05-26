from __future__ import annotations

from .models import (
    ChunkList,
    DraftAnswers,
    RetrievalResults,
    ReviewOverride,
    ReviewOverrides,
)

_SEPARATOR = "-" * 60


def _display_summary(
    retrieval_results: RetrievalResults,
    draft_answers: DraftAnswers,
) -> None:
    print("\n" + "=" * 60)
    print("HUMAN REVIEW CHECKPOINT — Retrieval & Draft Labels")
    print("=" * 60)

    draft_map = {d.query_id: d for d in draft_answers.answers}

    for qr in retrieval_results.results:
        draft = draft_map.get(qr.query_id)
        print(f"\n  [{qr.query_id}] {qr.question}")
        print(f"  Draft label : {draft.label if draft else 'N/A'}")
        print(f"  Retrieved   : {', '.join(rc.chunk_id for rc in qr.retrieved_chunks)}")
    print()


def run_human_review(
    retrieval_results: RetrievalResults,
    draft_answers: DraftAnswers,
    chunk_list: ChunkList,
) -> ReviewOverrides:
    _display_summary(retrieval_results, draft_answers)

    valid_chunk_ids = {c.chunk_id for c in chunk_list.chunks}
    valid_query_ids = {r.query_id for r in retrieval_results.results}
    overrides: list[ReviewOverride] = []

    print(_SEPARATOR)
    print("Do you want to override retrieved chunks for any query before audit?")
    print("Enter query_id and comma-separated chunk_ids to force as final context,")
    print("or press Enter to continue.")
    print(_SEPARATOR)

    while True:
        try:
            raw = input(
                "  Override (query_id chunk_id1,chunk_id2,...) or Enter to finish: "
            ).strip()
        except EOFError:
            # Non-interactive mode (e.g. piped input)
            break

        if not raw:
            break

        parts = raw.split(None, 1)
        if len(parts) != 2:
            print("  ERROR: expected format: Q1 chunk_id1,chunk_id2")
            continue

        query_id, chunk_ids_raw = parts
        if query_id not in valid_query_ids:
            print(
                f"  ERROR: unknown query_id '{query_id}'. Valid: {sorted(valid_query_ids)}"
            )
            continue

        chunk_ids = [cid.strip() for cid in chunk_ids_raw.split(",") if cid.strip()]
        bad_ids = [cid for cid in chunk_ids if cid not in valid_chunk_ids]
        if bad_ids:
            print(f"  ERROR: unknown chunk_ids: {bad_ids}")
            print(f"  Valid chunk_ids include: {sorted(valid_chunk_ids)[:10]} ...")
            continue

        # Replace any existing override for this query
        overrides = [o for o in overrides if o.query_id != query_id]
        overrides.append(
            ReviewOverride(query_id=query_id, override_chunk_ids=chunk_ids)
        )
        print(f"  Saved override for {query_id}: {chunk_ids}")

    if overrides:
        print(f"\n  {len(overrides)} override(s) recorded.")
    else:
        print("\n  No overrides. Proceeding with original retrieval.")

    return ReviewOverrides(overrides=overrides)
