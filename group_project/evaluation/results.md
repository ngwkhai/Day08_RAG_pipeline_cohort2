# RAG Evaluation Results

Generated at: `2026-06-08T21:28:16`

Framework: **Local deterministic evaluator (DeepEval-style metrics)**

Note: this run uses deterministic local proxy metrics so the evaluation can run reproducibly without an external LLM judge. Scores are useful for A/B comparison and debugging retrieval quality, not as a final human-grade score.

## Overall Scores

| Metric | Hybrid + rerank | Hybrid no rerank | Dense-only | BM25-only |
|--------|-----------------|------------------|------------|----------|
| Faithfulness | 93.5% | 88.5% | 90.0% | 81.7% |
| Answer Relevance | 19.6% | 16.4% | 19.5% | 18.6% |
| Context Recall | 55.9% | 55.6% | 54.6% | 51.7% |
| Context Precision | 43.3% | 35.3% | 42.5% | 31.1% |
| Average | 53.1% | 48.9% | 51.6% | 45.7% |

## A/B Comparison Analysis

Best config in this run: **Config A - hybrid + RRF + rerank** (`hybrid_rerank`) with average score 53.1%.

- **Config A:** Hybrid semantic + BM25, merged by RRF, then reranked with the current bi-encoder reranker.
- **Config B:** Hybrid semantic + BM25, merged by RRF, without reranking.
- **Config C:** Dense-only retrieval from ChromaDB embeddings.
- **Config D:** BM25-only lexical retrieval from markdown chunks.

Hybrid + rerank vs dense-only delta: **+0.014**. Hybrid retrieval is currently helping or matching dense retrieval on the golden set.

## Worst Performers (Bottom 3, Config A)

| # | Question | Avg | Faithfulness | Relevance | Recall | Precision | Failure Stage | Root Cause |
|---|----------|-----|--------------|-----------|--------|-----------|---------------|------------|
| 1 | Bo luat Hinh su sua doi 2017 co lien quan gi den cac toi pham ve ma tuy? | 29.6% | 79.2% | 13.1% | 19.0% | 6.9% | Retriever precision | Expected evidence terms are weakly covered by retrieved chunks. |
| 2 | Hinh phat cho toi tang tru trai phep chat ma tuy theo Dieu 249 Bo luat Hinh su la gi? | 35.5% | 92.4% | 10.0% | 26.7% | 13.1% | Answer relevance | Expected evidence terms are weakly covered by retrieved chunks. |
| 3 | Cai nghien bat buoc khac gi voi cai nghien tu nguyen? | 36.1% | 88.3% | 6.5% | 38.6% | 11.2% | Answer relevance | Expected evidence terms are weakly covered by retrieved chunks. |

## Per-Case Scores (Config A)

| ID | Category | Avg | Sources | Question |
|----|----------|-----|---------|----------|
| legal_001 | legal | 35.5% | 5 | Hinh phat cho toi tang tru trai phep chat ma tuy theo Dieu 249 Bo luat Hinh su la gi? |
| legal_002 | legal | 41.1% | 5 | Toi van chuyen trai phep chat ma tuy duoc quy dinh tai dieu nao cua Bo luat Hinh su? |
| legal_003 | legal | 45.2% | 5 | Toi mua ban trai phep chat ma tuy nam o dieu nao va noi dung chinh la gi? |
| legal_004 | legal | 62.8% | 5 | Luat Phong chong ma tuy 2021 quy dinh nhung hinh thuc cai nghien ma tuy nao? |
| legal_005 | legal | 66.5% | 5 | Nghi dinh 105/2021/ND-CP huong dan noi dung gi lien quan den Luat Phong chong ma tuy? |
| legal_006 | legal | 39.7% | 5 | Quy trinh xac dinh tinh trang nghien ma tuy can dua tren nhung gi? |
| legal_007 | legal | 46.5% | 5 | Huong dan ap dung quy dinh ve cac toi pham ma tuy dung de lam gi? |
| legal_008 | legal | 46.5% | 5 | Nguoi su dung trai phep chat ma tuy co lien quan den bien phap quan ly va cai nghien nhu the... |
| legal_009 | legal | 29.6% | 5 | Bo luat Hinh su sua doi 2017 co lien quan gi den cac toi pham ve ma tuy? |
| legal_010 | legal | 36.1% | 5 | Cai nghien bat buoc khac gi voi cai nghien tu nguyen? |
| news_001 | news | 68.3% | 5 | Bai VnExpress nao noi ve ca si Long Nhat va Son Ngoc Minh bi bat vi lien quan ma tuy? |
| news_002 | news | 66.4% | 5 | Bai Ma tuy trong loi song showbiz phan tich van de gi? |
| news_003 | news | 65.2% | 5 | Bai ve dien vien hai Huu Tin tren VnExpress noi ly do su dung ma tuy la gi? |
| news_004 | news | 78.2% | 5 | Bai Tuoi Tre ve vu Miu Le o Cat Ba noi den viec gi? |
| news_005 | news | 68.4% | 5 | Cac bai bao ve Chi Dan, An Tay va Truc Phuong lien quan den chu de nao? |

## Recommendations

### Cai tien 1
**Action:** Clean converted markdown, remove navigation/advertisement blocks from news pages before indexing.  
**Expected impact:** Higher context precision and cleaner citations.

### Cai tien 2
**Action:** Rebuild the golden set after reviewing final markdown, especially for news items whose crawler title is generic.  
**Expected impact:** More reliable relevance and recall scores.

### Cai tien 3
**Action:** Try reranking with a real multilingual cross-encoder or Jina API, then compare against the current bi-encoder rerank.  
**Expected impact:** Better ordering for legal questions with similar articles and provisions.
