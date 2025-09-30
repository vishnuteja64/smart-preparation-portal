# smart-preparation-portal

[User Frontend] <--HTTPS--> [API (FastAPI)]
      |                            |
      |                            +--> [Auth / User DB (Postgres)]
      |                            |
      |                            +--> [Retrieval Service] --> [Cache (Redis)]
      |                                                        |
      |                                                        +--> [Vector DB (HNSW/FAISS/Milvus)]
      |                                                        |
      |                                                        +--> [Reranker Service]
      |
      +--> [Upload Endpoint] --> [Ingestion Worker] --> [OCR/Parser] --> [Chunker] --> [Embedding Worker] --> [Vector DB]
      
[API] --calls--> [LLM Service (OpenAI / HF / local)] (via prompt templates)
[API] --logs--> [Tracing / Observability]


MVP features (must-have)

Upload exam PDFs / textbooks / notes (PDF, images via OCR).

Parse + chunk + embed + index documents (metadata: year, subject, topic).

Year-wise / topic-wise query & practice generation (get past questions by topic/year).

Chat-with-document (RAG-backed QA) that cites sources (page/year).

Create practice sets from topic/year and record answers + correctness.

Simple frontend (React or Streamlit) + backend API (FastAPI) + deployed demo.

Small evaluation harness: 20–50 labeled queries to compute precision@1/3 and latency.

MVP acceptance: live URL + GitHub repo + demo video + baseline metrics logged.

Advanced features (v2+)

Hybrid retriever (BM25 + dense) and cross-encoder reranker.

Agent actions: “create custom test”, “generate flashcards”, “summarize chapter”, “schedule revision (spaced repetition)”.

Auto-topic tagging (NER/topic classification) and difficulty estimation.

Analytics dashboard (user progress, weakness-by-topic).

Offline export (PDF practice paper) and shareable links.

Multimodal support (diagrams from textbooks via OCR + image Q&A).

Social / community features (leaderboard, study groups).

Data & ingestion (what to include)

Past question PDFs (year, file metadata).

Textbooks / reference PDFs.

Student notes (text).

Images / scanned pages → OCR (Tesseract / Amazon Textract / Google Vision) → text.

Optional: video lectures → ASR transcript.

Store raw files in object storage (S3 / GCS) + parsed text & metadata in Postgres + chunks in vector DB.

Learning map — what you’ll learn while building

Python engineering (FastAPI, async I/O).

Preprocessing & OCR (text extraction, normalization).

Chunking strategies (semantic chunking, sliding windows).

Embeddings (model selection, batching).

Vector DBs & indexes (HNSW, IVF, FAISS/Milvus/Pinecone).

Retrieval types (BM25, dense, hybrid).

Reranking (cross-encoders, MMR).

RAG & prompt engineering (citation, instruction design).

Agents & tool use (LangChain agents).

Evaluation metrics (precision@k, recall, MRR, accuracy, latency).

Caching, batching & cost optimization.

Tracing & observability (OpenTelemetry / LangSmith).

CI/CD, Docker, infra-as-code, deployment (Cloud Run / ECS / Render).

Security & privacy (PII handling, ACLs).

High-Level Architecture (HLD)

Components:

Frontend (React/Next.js or Streamlit)

API Gateway / Backend (FastAPI)

Ingestion Worker(s) (OCR, parser, chunker)

Embedding Worker (batched embeddings)

Vector DB (Milvus / Pinecone / Weaviate / FAISS)

Relational DB (Postgres) for metadata, user data, results

Caching layer (Redis)

Reranker service (optional small model service)

LLM service (OpenAI or hosted LLM endpoints)

Agent orchestration (LangChain)

Tracing / Logging / Metrics (LangSmith / OpenTelemetry + Grafana)

CI/CD pipeline (GitHub Actions)

Object Storage (S3/GCS)


Low-Level Design (LLD)

Below are key schemas, API endpoints, and implementation details.

DB schemas (simplified)

users: id, email, hashed_password, created_at, prefs
documents: id, user_id, title, filename, source_type, year, subject, uploaded_at, s3_path
chunks: id, document_id, chunk_text, start_offset, end_offset, tokens, metadata (topic, difficulty)
vectors: id (chunk_id), vector (stored in vector DB), indexing_meta (shard, created_at)
queries: id, user_id, query_text, created_at, top_k_ids, reranker_scores, final_answer_id
practice_sessions: id, user_id, topic, created_at, questions[], correct_count, time_spent
flashcards: id, user_id, front, back, efactor, interval, next_review_date

Vector DB schema

Index by chunk_id with metadata fields: document_id, page, year, subject, topic, difficulty, text_snippet.

Choose HNSW initially (fast, high recall for chat).

Chunking params (starting)

chunk_size: 300–600 tokens

overlap: 20–30% (~60–150 tokens)

tokenizer = same as embedding model tokenizer

special handling: keep question/answer pairs intact if detected

Embedding pipeline

Batch size: 64–256 (depends on model)

Asynchronous worker queue (Celery / RQ / Cloud Tasks)

Version embeddings (store model_name + model_version with vectors)

Retriever flow

Normalize query (lowercase, remove punctuation)

Fingerprint query for cache check

If cache miss: query vector DB (dense top-k) + optionally BM25 top-k (lexical)

Combine scores (hybrid) → pass top-N to reranker

Reranker sorts top-K → select top-3 passages for LLM prompt

Reranker

Cross-encoder model (smaller BERT-like) running as microservice (TorchServe / FastAPI)

Re-rank top-50 from ANN to top-5 with cross-attention scoring

Prompt templates (example)
SYSTEM: You are a factual exam assistant. Use only the passages below to answer; cite passage identifiers.
USER: Question: {user_query}
Passages:
[1] {passage_1} (doc: {doc1}, page:{p1})
[2] {passage_2} ...
Instruction: Answer concisely, show short explanation, then cite sources like (doc:year:page).

Agent tools (v2)

tool: generate_test(topic, year_range, n_questions)

tool: create_flashcards(question_id)

tool: evaluate_answer(user_answer, correct_answer)

tool: search_docs(filter={topic, year})

APIs (examples)

POST /upload -> returns document_id (accepts file)

POST /parse/{document_id} -> triggers ingestion pipeline

POST /query body: {user_id, query, filters} -> returns {answer, sources, top_passages}

GET /practice/start?topic=&year= -> returns question set

POST /practice/answer -> records result, returns feedback + explanation

GET /user/progress -> returns analytics by topic

Caching strategy

Cache key: sha256(query + user_context + filters + top_k)

TTL: short for chat (30–120s), longer for frequently requested practice sets (1–24h depending on freshness)

Cache storage: Redis with LRU eviction

Authentication & Security

JWT tokens (short-lived) + refresh tokens

ACLs on document access (user docs vs shared docs)

PII redaction option before indexing

Index & model tuning (practical starting points)

Vector index: HNSW parameters — efConstruction: 128–200; efSearch start: 50 (tune to trade recall/latency).

Top-k for retrieval: initial dense_top_k = 50; rerank_top_k = 10; final_sent_to_llm = top 3.

Chunk size: 300–400 tokens with 20–30% overlap.

Embedding model: start with a mid-size semantic model (sentence-transformers or OpenAI embeddings) and benchmark recall@1/3.

Reranking: a cross-encoder fine-tuned on QA pairs for best top-1 accuracy.

Evaluation & metrics

Retrieval: precision@1, precision@3, recall@k, MRR on labeled queries.

Generation: Factuality checks (manual or automatic with exact match for MCQ), BLEU/ROUGE for summaries (if needed).

UX metrics: average response latency, 95th percentile latency, cost per query (USD).

Product metrics: daily active users, practice sessions completed, accuracy improvement per user, retention/week.

Build an evaluation harness: eval/queries.json + scripts/compute_metrics.py run in CI (fail if recall < baseline).

Cost & optimization tips

Use smaller models for reranking (cheaper) and only call big LLMs for final answer generation.

Batch embeddings to reduce API calls.

Cache frequently asked queries & practice sets.

Use on-demand vs always-on inference depending on traffic.

Monitor cost-per-answer and set alarms (Cloud billing alerts).

Deployment recommendations (MVP)

Backend: FastAPI containerized (Docker) → deploy to Cloud Run / Render / ECS Fargate.

Vector DB: managed Pinecone or Milvus (early) to avoid infra ops.

Storage: S3 or GCS for files.

CI/CD: GitHub Actions — tests → build → push image → deploy.

Observability: Prometheus + Grafana or use managed dashboards; integrate LangSmith / OpenTelemetry for traces.
