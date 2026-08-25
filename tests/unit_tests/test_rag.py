from paper_guide.agents.rag import _format_context


def test_format_context_separates_retrieval_rank_from_element_number() -> None:
    chunks = [
        {"title": "Paper A", "content": "text", "element_type": "text"},
        {"title": "Paper A", "content": "abstract", "element_type": "abstract"},
        {
            "title": "Paper A",
            "content": "table",
            "element_type": "table",
            "element_number": "1",
        },
        {
            "title": "Paper A",
            "content": "figure",
            "element_type": "figure",
            "element_number": "2",
        },
    ]

    formatted = _format_context(chunks)

    assert "【检索块 1｜正文】" in formatted
    assert "【检索块 2｜摘要】" in formatted
    assert "【检索块 3｜表格 1】" in formatted
    assert "【检索块 4｜图片 2】" in formatted
