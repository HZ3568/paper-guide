"""单元测试：检索管线的纯函数（RRF 融合、元数据过滤），不触网。"""

from langchain_core.documents import Document

from opendetect_ai.tools import retriever


def _doc(title, arxiv_id="", chunk_idx=0, published="", authors=""):
    return Document(page_content=f"{title} chunk{chunk_idx}", metadata={
        "title": title, "arxiv_id": arxiv_id, "chunk_idx": chunk_idx,
        "published": published, "authors": authors,
    })


def test_rrf_prefers_docs_ranked_high_in_both_lists() -> None:
    """同时在两个列表中靠前的文档，RRF 融合后应排最前。"""
    a = _doc("A", "1")
    b = _doc("B", "2")
    c = _doc("C", "3")
    dense  = [a, b, c]
    sparse = [b, a, c]   # B 在稀疏列表更靠前，但 A 在两列表都靠前
    fused = retriever._rrf_fuse([dense, sparse])
    keys = [retriever._doc_key(d) for d in fused]
    # A 与 B 都应排在 C 前面；A 因两列表都居前，分数不低于 B
    assert keys.index(("1", 0)) < keys.index(("3", 0))
    assert keys.index(("2", 0)) < keys.index(("3", 0))


def test_apply_filters_year_range() -> None:
    plan = retriever.RetrievalPlan(semantic_query="x", year_min=2022)
    docs = [_doc("old", "1", published="2020-01-01"),
            _doc("new", "2", published="2023-05-01")]
    out = retriever._apply_filters(docs, plan)
    assert [d.metadata["title"] for d in out] == ["new"]


def test_apply_filters_title_keyword() -> None:
    plan = retriever.RetrievalPlan(semantic_query="x", title_keyword="Swin")
    docs = [_doc("Swin Transformer", "1"), _doc("Vision Transformer", "2")]
    out = retriever._apply_filters(docs, plan)
    assert [d.metadata["title"] for d in out] == ["Swin Transformer"]


def test_apply_filters_empty_result_falls_back_to_all() -> None:
    """过滤条件过严导致全空时，应放弃过滤而非返回空（宁滥毋缺）。"""
    plan = retriever.RetrievalPlan(semantic_query="x", year_min=2099)
    docs = [_doc("a", "1", published="2020-01-01")]
    out = retriever._apply_filters(docs, plan)
    assert out == docs


def test_to_dict_preserves_pdf_element_metadata() -> None:
    doc = Document(
        page_content="## Figure 2: Architecture",
        metadata={
            "paper_id": "arxiv:1234.5678",
            "document_id": "arxiv:1234.5678:pdf:abc",
            "document_hash": "abc",
            "chunk_id": "chunk:abc",
            "content_hash": "hash",
            "parser_version": 3,
            "metadata_version": 1,
            "embedding_model": "text-embedding-v4",
            "content_source": "pdf",
            "title": "Paper",
            "page": 3,
            "chunk_idx": 7,
            "element_type": "figure",
            "element_number": "2",
            "element_title": "Architecture",
            "reference_pages": "4,6",
            "reference_count": 2,
        },
    )

    result = retriever._to_dict(doc)
    assert result["paper_id"] == "arxiv:1234.5678"
    assert result["document_id"] == "arxiv:1234.5678:pdf:abc"
    assert result["document_hash"] == "abc"
    assert result["chunk_id"] == "chunk:abc"
    assert result["metadata_version"] == 1
    assert result["embedding_model"] == "text-embedding-v4"
    assert result["content_source"] == "pdf"
    assert result["element_type"] == "figure"
    assert result["element_number"] == "2"
    assert result["reference_pages"] == "4,6"


def test_doc_key_prefers_stable_chunk_id() -> None:
    first = Document(
        page_content="first",
        metadata={"chunk_id": "chunk:stable", "chunk_idx": 1},
    )
    second = Document(
        page_content="second",
        metadata={"chunk_id": "chunk:stable", "chunk_idx": 99},
    )

    assert retriever._doc_key(first) == retriever._doc_key(second) == ("chunk:stable",)


def test_llm_reranker_preserves_valid_empty_result(monkeypatch) -> None:
    class FakeRunnable:
        def invoke(self, _messages):
            return retriever._RerankResult(relevant_indices=[])

    class FakeLLM:
        def with_structured_output(self, *_args, **_kwargs):
            return FakeRunnable()

    monkeypatch.setattr(retriever, "_get_llm", lambda: FakeLLM())
    assert retriever._rerank_llm("unrelated", [_doc("A")], 5) == []
