from pathlib import Path 
import re #regex

import numpy as np #to loasd and compare saved embedding vectors
import pandas as pd
import streamlit as st # for dashboard
from rank_bm25 import BM25Okapi #for ranking exact matches
from sentence_transformers import SentenceTransformer #convert comments and qs  into embeddings
import torch #used by tranforms to run lm and detect gpu
from transformers import AutoTokenizer, AutoModelForCausalLM #convert text into tokens so model can understand
#loads the Qwen language model that generates the final grounded answer.



# here is Page setup and project paths

st.set_page_config(
    page_title="TechView YouTube Intelligence Engine",
    page_icon="🔎",
    layout="wide"
)

# All files are expected to be in the same folder as app.py.
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "youtube_comments_BERTtopics.csv"
EMBEDDINGS_PATH = BASE_DIR / "comment_embeddings.npy"
MLFLOW_RUNS_PATH = BASE_DIR / "mlflow_runs_export.csv"

MODEL_NAME = "all-MiniLM-L6-v2"
RRF_K = 60
LLM_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

# Common questioon words are removed from BM25 ranking so important terms
# such as product names and feature words have more ranking importaance.
BM25_STOPWORDS = {
    "what", "do", "does", "did", "you", "your", "users", "user",
    "people", "think", "thought", "about", "say", "says",
    "the", "a", "an", "and", "or", "for", "to", "of", "in",
    "on", "with", "from", "this", "that", "these", "those",
    "is", "are", "was", "were", "be", "been", "being",
    "can", "could", "would", "should", "i", "im", "my",
    "it", "its", "they", "them", "their", "u"
}



# Data loading

@st.cache_data
def load_data():
    """
    Load and validate the final processed dataset.

    This function removes unusable comments, standardises sentiment labels,
    ensures like_count is numeric, and caches the dataframe so it is not
    reloaded whenever the user changes a Streamlit widget.
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "youtube_comments_BERTtopics.csv was not found beside app.py."
        )

    df = pd.read_csv(DATA_PATH)

    if "clean_text" not in df.columns:
        raise ValueError("The dataset must contain a clean_text column.")

    # Keep only non-empty cleaned comments for retrieval and dashboard display.
    df = df.dropna(subset=["clean_text"]).copy()
    df["clean_text"] = df["clean_text"].astype(str).str.strip()
    df = df[df["clean_text"] != ""].copy()

    # Prefer RoBERTa sentiment if available, then fallback options.
    sentiment_candidates = [
        "roberta_sentiment",
        "sentiment",
        "vader_sentiment"
    ]

    sentiment_column = next(
        (column for column in sentiment_candidates if column in df.columns),
        None
    )

    if sentiment_column:
        raw_labels = df[sentiment_column].astype(str).str.lower().str.strip()

        # Convert model labels into readable dashboard labels.
        label_map = {
            "label_0": "negative",
            "label_1": "neutral",
            "label_2": "positive",
            "negative": "negative",
            "neutral": "neutral",
            "positive": "positive"
        }

        df["sentiment"] = raw_labels.map(label_map).fillna(raw_labels)
    else:
        df["sentiment"] = "unknown"

    # Ensure like_count always exists and is numeric.
    if "like_count" not in df.columns:
        df["like_count"] = 0

    df["like_count"] = pd.to_numeric(
        df["like_count"],
        errors="coerce"
    ).fillna(0).astype(int)

    return df.reset_index(drop=True)



# Retrieval helpers

def tokenize_bm25(text):
    """
    Convert text into BM25 tokens.

    Generic question words are removed so that product names and meaningful
    feature terms receive greater importance during lexical ranking.
    """
    tokens = re.findall(r"[a-z0-9]+", str(text).lower())

    return [
        token for token in tokens
        if token not in BM25_STOPWORDS and len(token) > 1
    ]


def route_query(query):
    """
    Select the retrieval method using transparent keyword rules.

    Priority order:
    1. Exact lookup terms -> BM25
    2. Comparison terms -> Hybrid
    3. Feature/issue terms -> Semantic
    4. Broad product opinions -> Hybrid
    """
    query = query.lower()

    lookup_terms = [
        "exact model", "specification", "specifications",
        "specs", "model number", "find comments mentioning"
    ]

    comparison_terms = [
        "compare", "comparison", "vs", "versus",
        "better than", "which is better"
    ]

    issue_terms = [
        "issue", "issues", "problem", "problems",
        "comfort", "reliability", "battery",
        "camera", "fitness", "feature", "features",
        "quality", "bug", "bugs", "failure",
        "overheating", "charging", "price"
    ]

    if any(term in query for term in lookup_terms):
        return "bm25", "Exact keyword or model lookup"

    if any(term in query for term in comparison_terms):
        return "hybrid", "Comparison question"

    if any(term in query for term in issue_terms):
        return "semantic", "Feature or issue-focused question"

    return "hybrid", "General product-opinion question"


def is_usable_comment(text, min_words=7):
    """
    Remove very short and question-only comments.

    This helps ensure the final evidence set contains useful opinions rather
    than short reactions or requests for advice.
    """
    words = re.findall(r"[a-z0-9]+", text.lower())

    if len(words) < min_words:
        return False

    question_patterns = [
        r"^(what|why|how|should|would|do|does|can|could|is|are)\b",
        r"\bwould you recommend\b",
        r"\bshould i\b",
        r"\bdo yall think\b",
        r"\bwhat do you think\b"
    ]

    return not any(
        re.search(pattern, text.lower())
        for pattern in question_patterns
    )



# Qwen LLM loading and grounded generation

@st.cache_resource(show_spinner="Loading the answer-generation model...")
def load_llm():
    """
    Load Qwen once and cache it for repeated dashboard queries.

    Qwen is only called after retrieval. It receives the selected evidence
    comments rather than the full dataset, which keeps answers grounded.
    """
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )

    # Puts the model in inference mode for stable generation.
    model.eval()

    return tokenizer, model


def build_evidence_context(documents, max_comments=5, max_chars=550):
    """
    Create numbered evidence blocks for the Qwen prompt.

    Limiting both number of comments and characters reduces prompt size
    while retaining enough context for a grounded summary.
    """
    evidence_blocks = []

    for index, document in enumerate(documents[:max_comments], start=1):
        cleaned_document = " ".join(str(document).split())

        evidence_blocks.append(
            f"[{index}] {cleaned_document[:max_chars]}"
        )

    return "\n\n".join(evidence_blocks)


def generate_grounded_rag_answer(query, retrieved_documents):
    """
    Generate a concise answer from retrieved evidence only.

    The system prompt prevents unsupported product facts and tells Qwen
    to treat comments as user opinions rather than objective truth.
    Deterministic decoding makes repeated outputs more stable.
    """
    tokenizer, llm_model = load_llm()

    evidence_context = build_evidence_context(retrieved_documents)

    messages = [
        {
            "role": "system",
            "content": (
                "You are an evidence-grounded assistant for YouTube tech-review comments. "
                "Use only the retrieved comments. Do not invent facts, specifications, "
                "or opinions. Treat comments as user opinions, not objective truth."
            )
        },
        {
            "role": "user",
            "content": f"""
Question:
{query}

Retrieved evidence:
{evidence_context}

Task:
Write one concise 2-3 sentence answer based only on the evidence.

Rules:
- Start with: "In the retrieved sample,"
- Do not say that all users agree.
- Do not use headings.
- Do not use citations.
- Do not add facts outside the retrieved comments.
"""
        }
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(llm_model.device)

    with torch.no_grad():
        generated_ids = llm_model.generate(
            **model_inputs,
            max_new_tokens=140,
            do_sample=False,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id
        )

    new_tokens = generated_ids[0][
        model_inputs["input_ids"].shape[1]:
    ]

    llm_answer = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True
    ).strip()

    return llm_answer, evidence_context


# -------------------------------------------------
# Retrieval model and index loading
# -------------------------------------------------
@st.cache_resource(
    show_spinner="Loading model and preparing retrieval index..."
)
def load_retrieval_resources():
    """
    Load SentenceTransformer, cached embeddings, and BM25 index.

    Cached embeddings avoid recomputing vectors at every dashboard restart.
    The validation check ensures the cache matches the active dataset.
    """
    df = load_data()
    texts = df["clean_text"].tolist()

    model = SentenceTransformer(MODEL_NAME)

    valid_cache = False

    if EMBEDDINGS_PATH.exists():
        cached_embeddings = np.load(EMBEDDINGS_PATH)

        valid_cache = (
            cached_embeddings.ndim == 2
            and cached_embeddings.shape[0] == len(texts)
            and cached_embeddings.shape[1]
            == model.get_sentence_embedding_dimension()
        )

        if valid_cache:
            embeddings = cached_embeddings.astype(np.float32)

    # Create embeddings only if a valid saved cache is unavailable.
    if not valid_cache:
        embeddings = model.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=True
        ).astype(np.float32)

        np.save(EMBEDDINGS_PATH, embeddings)

    # Build BM25 index from tokenised cleaned comments.
    bm25_corpus = [tokenize_bm25(text) for text in texts]
    bm25 = BM25Okapi(bm25_corpus)

    return model, embeddings, bm25, texts


def semantic_rank(query, model, embeddings, candidate_k=100):
    """
    Rank comments by semantic similarity.

    Query and stored embeddings are normalised, so their dot product is
    equivalent to cosine similarity.
    """
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )[0]

    scores = embeddings @ query_embedding

    return np.argsort(scores)[::-1][:candidate_k].tolist()


def bm25_rank(query, bm25, candidate_k=100):
    """
    Rank comments using BM25 lexical relevance.

    This method is useful when exact product names, models, or keywords
    are important to the user query.
    """
    scores = bm25.get_scores(tokenize_bm25(query))

    return np.argsort(scores)[::-1][:candidate_k].tolist()


def hybrid_rank(query, model, embeddings, bm25, candidate_k=100):
    """
    Combine semantic and BM25 results using Reciprocal Rank Fusion.

    Rank positions are combined instead of raw scores because semantic
    similarity scores and BM25 scores use different scales.
    """
    semantic_indices = semantic_rank(
        query,
        model,
        embeddings,
        candidate_k
    )

    bm25_indices = bm25_rank(
        query,
        bm25,
        candidate_k
    )

    rrf_scores = {}

    for rank, index in enumerate(semantic_indices, start=1):
        rrf_scores[index] = (
            rrf_scores.get(index, 0) + 1 / (RRF_K + rank)
        )

    for rank, index in enumerate(bm25_indices, start=1):
        rrf_scores[index] = (
            rrf_scores.get(index, 0) + 1 / (RRF_K + rank)
        )

    return sorted(
        rrf_scores,
        key=rrf_scores.get,
        reverse=True
    )[:candidate_k]


def filter_results(indices, texts, product_term, final_k):
    """
    Apply product-term and usability filtering after retrieval.

    Literal product matching is transparent, but related-model comparison
    comments may still remain if the requested product is mentioned.
    """
    product_term = product_term.lower().strip()

    filtered = []

    for index in indices:
        comment = texts[index]

        if product_term and product_term not in comment.lower():
            continue

        if not is_usable_comment(comment):
            continue

        filtered.append(index)

        if len(filtered) == final_k:
            break

    return filtered


def retrieve_comments(query, product_term, method, final_k=5):
    """
    Run the selected retrieval method and return final evidence indexes.

    The system first ranks up to 100 candidate comments, then applies
    product filtering and usability filtering to get final evidence.
    """
    model, embeddings, bm25, texts = load_retrieval_resources()

    candidate_k = min(100, len(texts))

    if method == "semantic":
        ranked_indices = semantic_rank(
            query,
            model,
            embeddings,
            candidate_k
        )

    elif method == "bm25":
        ranked_indices = bm25_rank(
            query,
            bm25,
            candidate_k
        )

    else:
        ranked_indices = hybrid_rank(
            query,
            model,
            embeddings,
            bm25,
            candidate_k
        )

    return filter_results(
        ranked_indices,
        texts,
        product_term,
        final_k
    )



# Load dashboard dataset

try:
    df = load_data()

except Exception as error:
    st.error(f"Could not load the dataset: {error}")
    st.stop()



# Dashboard header and tabs

st.title("TechView YouTube Intelligence Engine")

st.caption(
    "Routed semantic, BM25, and hybrid retrieval over YouTube "
    "consumer-electronics review comments."
)

tab_search, tab_insights, tab_evaluation = st.tabs(
    ["Routed Retrieval", "Dataset Insights", "Evaluation"]
)



# Tab 1: Routed retrieval

# User flow:
# Question -> Router/manual selection -> Retrieval -> Qwen -> Evidence display
with tab_search:
    st.subheader("Ask about consumer opinions")

    left_column, right_column = st.columns([3, 1])

    with left_column:
        query = st.text_input(
            "Question",
            value=(
                "What comfort issues do users mention "
                "about Sony XM6 headphones?"
            )
        )

        product_term = st.text_input(
            "Product filter",
            value="xm6",
            help=(
                "Use a product word that appears in comments, "
                "such as xm6, pixel, garmin, or iphone."
            )
        )

    with right_column:
        mode = st.selectbox(
            "Retrieval mode",
            ["Automatic router", "Semantic", "BM25", "Hybrid"]
        )

        final_k = st.slider(
            "Evidence comments",
            min_value=3,
            max_value=10,
            value=5
        )

    run_search = st.button(
        "Run retrieval",
        type="primary",
        use_container_width=True
    )

    if run_search:
        if not query.strip():
            st.warning("Enter a question first.")

        else:
            # Automatic mode selects a method according to query intent.
            # Manual mode lets the user compare individual methods.
            if mode == "Automatic router":
                method, reason = route_query(query)
            else:
                method = mode.lower()
                reason = "Manual retrieval-method selection"

            # Retrieve evidence before generating an answer.
            with st.spinner("Retrieving relevant comments..."):
                indices = retrieve_comments(
                    query=query,
                    product_term=product_term,
                    method=method,
                    final_k=final_k
                )

            if not indices:
                st.warning(
                    "No usable comments were found. "
                    "Try a broader product filter."
                )

            else:
                # Convert retrieved row indexes into the final evidence dataframe.
                result_df = df.iloc[indices].copy()

                result_df.insert(
                    0,
                    "rank",
                    range(1, len(result_df) + 1)
                )

                st.subheader("Grounded LLM Answer")

                # Qwen receives only retrieved evidence, never the full dataset.
                with st.spinner("Generating grounded answer..."):
                    llm_answer, _ = generate_grounded_rag_answer(
                        query=query,
                        retrieved_documents=result_df["clean_text"].tolist()
                    )

                st.info(llm_answer)

                st.caption(
                    "Generated by Qwen2.5-1.5B-Instruct using only the "
                    f"{len(result_df)} retrieved YouTube comments. "
                    "Review the evidence below."
                )

                method_names = {
                    "semantic": "Semantic retrieval",
                    "bm25": "BM25 lexical retrieval",
                    "hybrid": "Hybrid retrieval"
                }

                first_metric, second_metric, third_metric = st.columns(3)

                first_metric.metric(
                    "Selected method",
                    method_names[method]
                )

                second_metric.metric(
                    "Evidence comments",
                    len(result_df)
                )

                negative_count = int(
                    (result_df["sentiment"] == "negative").sum()
                )

                third_metric.metric(
                    "Negative labels",
                    negative_count
                )

                st.caption(f"Router reason: {reason}")

                # Sentiment summary is calculated from stored RoBERTa labels.
                # The LLM does not predict sentiment again.
                sentiment_order = ["positive", "negative", "neutral"]

                sentiment_counts = (
                    result_df["sentiment"]
                    .value_counts()
                    .reindex(sentiment_order, fill_value=0)
                )

                positive_count = sentiment_counts["positive"]
                negative_count = sentiment_counts["negative"]

                if positive_count > negative_count:
                    snapshot = (
                        "Overall snapshot: the retrieved comment sample "
                        "leans positive."
                    )
                elif negative_count > positive_count:
                    snapshot = (
                        "Overall snapshot: the retrieved comment sample "
                        "leans negative."
                    )
                else:
                    snapshot = (
                        "Overall snapshot: the retrieved comment sample "
                        "shows mixed opinions."
                    )

                st.subheader("Quick Sentiment Snapshot")
                st.write(snapshot)

                st.info(
                    f"Evidence summary: {sentiment_counts['positive']} comments "
                    f"labelled positive, {sentiment_counts['negative']} labelled "
                    f"negative, and {sentiment_counts['neutral']} labelled "
                    "neutral by RoBERTa. Review the retrieved evidence, "
                    "especially for product-comparison comments."
                )

                st.subheader("Retrieved Evidence")

                # Every displayed comment can be expanded so users can inspect
                # the source text and metadata behind the generated answer.
                for _, row in result_df.iterrows():
                    likes = int(row["like_count"])
                    sentiment = str(row["sentiment"]).title()

                    with st.expander(
                        f"Rank {row['rank']} · "
                        f"{sentiment} label · "
                        f"{likes} likes"
                    ):
                        st.write(row["clean_text"])

                        metadata = [f"Likes: {likes}"]

                        if "topic" in row.index:
                            metadata.append(f"Topic: {row['topic']}")

                        if "video_id" in row.index:
                            metadata.append(f"Video: {row['video_id']}")

                        st.caption(" · ".join(metadata))

                st.subheader("Evidence Table")

                table_columns = [
                    column for column in [
                        "rank",
                        "sentiment",
                        "like_count",
                        "topic",
                        "clean_text"
                    ]
                    if column in result_df.columns
                ]

                st.dataframe(
                    result_df[table_columns],
                    use_container_width=True,
                    hide_index=True
                )

                st.caption(
                    "Limitation: literal product filtering can include "
                    "related-model comparisons when the target product "
                    "is mentioned."
                )



# Tab 2: Dataset insights

# Provides descriptive analytics from the final processed dataset.
with tab_insights:
    st.subheader("Dataset Overview")

    first_metric, second_metric, third_metric = st.columns(3)

    first_metric.metric("Comments analysed", f"{len(df):,}")
    second_metric.metric("Videos", f"{df['video_id'].nunique():,}")
    third_metric.metric("Topics", f"{df['topic'].nunique():,}")

    st.subheader("RoBERTa Sentiment-Label Distribution")

    sentiment_distribution = (
        df["sentiment"]
        .value_counts()
        .reindex(["positive", "negative", "neutral"], fill_value=0)
    )

    st.bar_chart(
        sentiment_distribution,
        color="#7C3AED"
    )

    st.subheader("Most Liked Comments")

    most_liked_columns = [
        column for column in [
            "like_count",
            "sentiment",
            "clean_text"
        ]
        if column in df.columns
    ]

    st.dataframe(
        df.sort_values("like_count", ascending=False)
        .head(10)[most_liked_columns],
        use_container_width=True,
        hide_index=True
    )



# Tab 3: Evaluation

# Displays MLflow-exported metrics without requiring users to open
# the notebook or MLflow tracking interface.
with tab_evaluation:
    st.subheader("MLflow-Tracked Retrieval Evaluation")

    if not MLFLOW_RUNS_PATH.exists():
        st.error(
            "mlflow_runs_export.csv was not found beside app.py. "
            "Run the MLflow export cell in the Colab notebook and place "
            "the CSV here."
        )

    else:
        runs_df = pd.read_csv(MLFLOW_RUNS_PATH)

        # Retrieve only the three retrieval-method experiment runs.
        retrieval_runs = runs_df[
            runs_df["tags.mlflow.runName"].str.endswith(
                "_retrieval",
                na=False
            )
            & (
                runs_df["tags.mlflow.runName"]
                != "final_routed_engine"
            )
        ].copy()

        if (
            not retrieval_runs.empty
            and "params.retrieval_method" in retrieval_runs.columns
        ):
            precision_df = retrieval_runs.set_index(
                "params.retrieval_method"
            )[["metrics.mean_precision_at_5"]]

            precision_df.columns = ["Mean Precision@5"]
            precision_df.index.name = "Method"

            st.bar_chart(
                precision_df,
                color="#7C3AED"
            )

            best_row = precision_df["Mean Precision@5"].idxmax()

            cols = st.columns(len(precision_df) + 1)

            cols[0].metric(
                "Best retriever",
                best_row.title()
            )

            for col, (method, row) in zip(
                cols[1:],
                precision_df.iterrows()
            ):
                col.metric(
                    f"{method.title()} Precision@5",
                    f"{row['Mean Precision@5']:.2f}"
                )

        else:
            st.warning("No per-method retrieval runs found in the export.")

        st.subheader("Final Routed-Engine Evaluation")

        final_run = runs_df[
            runs_df["tags.mlflow.runName"] == "final_routed_engine"
        ]

        if not final_run.empty:
            final_row = final_run.iloc[0]

            routed_metrics = pd.DataFrame({
                "Metric": [
                    "Routing accuracy",
                    "Mean evidence relevance",
                    "Label-faithful interpretation"
                ],
                "Score": [
                    final_row.get("metrics.routing_accuracy", 0),
                    final_row.get("metrics.mean_evidence_relevance", 0),
                    final_row.get(
                        "metrics.label_faithful_interpretation",
                        0
                    )
                ]
            }).set_index("Metric")

            st.bar_chart(
                routed_metrics,
                color="#7C3AED"
            )

            st.info(
                "The rule-based router selected the intended retrieval "
                "strategy for the five final evaluation queries. Metrics "
                "are logged to MLflow and exported for this dashboard."
            )

        else:
            st.warning("No final_routed_engine run found in the export.")

        # Raw log remains available for transparency and debugging.
        with st.expander("View Raw MLflow Run Log"):
            st.dataframe(
                runs_df,
                use_container_width=True,
                hide_index=True
            )

    st.subheader("Known Limitations")

    st.write(
        "- Some retrieved comments compare related products or older models.\n"
        "- RoBERTa labels can occasionally misinterpret nuanced comparisons.\n"
        "- The evidence summary reports stored labels and does not claim "
        "official product specifications.\n"
        "- Routing is rule-based keyword matching rather than an LLM-based "
        "agent decision. This is a deliberate transparency and latency "
        "trade-off."
    )