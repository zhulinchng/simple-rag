# RAG Pipeline Evaluation Report

## 1. Retrieval Summary

- **Mode**: hybrid
- **BM25 weight**: 0.5
- **Embedding weight**: 0.5
- **Embedding model**: sentence-transformers/all-MiniLM-L6-v2
- **Top-k**: 3
- **Total chunks indexed**: 4
- **Documents indexed**: billing.txt, product_overview.txt, security.txt
- **Index built at**: 2026-05-26T15:02:17.035257+00:00

## 2. Query-by-Query Results

### Q1

**Question**: How long is event data retained on the standard plan?

| Field | Value |
|-------|-------|
| Final context chunk IDs | product_overview_000000, security_000000, product_overview_000300 |
| Overridden | No |
| Draft label | `supported` ✓ |
| Draft citations | product_overview_000000 |
| Audit label | `pass` |
| Hallucination risk | **LOW** |

**Draft answer**: Event data is retained for 13 months on the standard plan.

**Final recommendation**: Answer approved.

### Q2

**Question**: Does the product support SCIM provisioning?

| Field | Value |
|-------|-------|
| Final context chunk IDs | product_overview_000300, product_overview_000000, billing_000000 |
| Overridden | No |
| Draft label | `supported` ✓ |
| Draft citations | product_overview_000300 |
| Audit label | `pass` |
| Hallucination risk | **LOW** |

**Draft answer**: Yes, the product supports SCIM provisioning on enterprise plans.

**Final recommendation**: Answer approved.

### Q3

**Question**: Can customers get refunds for unused days in a month?

| Field | Value |
|-------|-------|
| Final context chunk IDs | billing_000000, security_000000, product_overview_000300 |
| Overridden | No |
| Draft label | `supported` ✓ |
| Draft citations | billing_000000 |
| Audit label | `pass` |
| Hallucination risk | **LOW** |

**Draft answer**: Refunds are not offered for partial months, except where required by law.

**Final recommendation**: Answer approved.

### Q4

**Question**: Is the service HIPAA compliant?

| Field | Value |
|-------|-------|
| Final context chunk IDs | security_000000, product_overview_000300, billing_000000 |
| Overridden | No |
| Draft label | `supported` ✓ |
| Draft citations | security_000000 |
| Audit label | `pass` |
| Hallucination risk | **LOW** |

**Draft answer**: The service is not described as HIPAA compliant in current public documentation.

**Final recommendation**: Answer approved.

## 3. Reviewed Overrides

_No overrides applied. Audit used original retrieval results._

## 4. Audit Findings

| Query | Audit | Hallucination Risk | Citation Check |
|-------|-------|--------------------|----------------|
| Q1 | `pass` | LOW | The citation 'product_overview_000000' is valid and corresponds to the document  |
| Q2 | `pass` | LOW | The citation product_overview_000300 is valid and directly supports the claim th |
| Q3 | `pass` | LOW | The citation billing_000000 is valid and corresponds to the exact statement in t |
| Q4 | `pass` | LOW | The citation 'security_000000' is valid and directly references the document tha |

**Summary**: 4 passed, 0 failed out of 4 queries.

## 5. Failure Modes Observed

_No significant failure modes detected._

## 6. Recommended Improvements

- Pipeline performed well. No major improvements required.
