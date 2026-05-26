.PHONY: run validate clean install test

install:
	pip install -r requirements.txt

run:
	python run_pipeline.py

validate:
	python validate.py

test:
	pytest tests/ -v

clean:
	rm -f chunks.json index_metadata.json retrieval_results.json \
	      draft_answers.json review_overrides.json answer_audit.json \
	      final_report.md llm_calls.jsonl retrieval_metrics.json \
	      revised_answers.json retrieval_error_analysis.json
