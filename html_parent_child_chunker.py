"""
学术文献（Markdown 内嵌 HTML）父子块拆分：按语义单元拆子块，
强制表格/图片/段落公式上下文，同级标题父块规则，相邻父块重叠。
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

from bs4 import BeautifulSoup, NavigableString, Tag

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
PARENT_MAX_CHARS = 500
OVERLAP_CHILD_COUNT = 3
OVERLAP_MAX_CHARS = 100
MATH_BLOCK_CLASS = "math-block"

# 子块文本类：按换行与中文句末标点切分（连续分隔符合并为一次切分）
_SPLIT_DELIM_PATTERN = re.compile(r"(\n+|。+|；+|！+|？+|……+)")


@dataclass
class ChildBlock:
    """子块：用于检索与父块拼装。"""

    text: str
    block_kind: str  # text | table | image | math
    is_from_heading: bool = False
    heading_elem_id: Optional[int] = None
    source_tag: str = ""


@dataclass
class _HeadingState:
    next_id: int = 0

    def new_id(self) -> int:
        i = self.next_id
        self.next_id += 1
        return i


def _normalize_ws(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    return s.strip()


def _split_mixed_html(elem: Tag) -> List[str]:
    """
    对 <p>/<h*> 内部按换行符 /。/；/！/？ 切分，保留行内标签（含行内公式）原位拼接。
    """
    chunks: List[str] = []
    cur: List[str] = []

    def flush() -> None:
        nonlocal cur
        if not cur:
            return
        merged = _normalize_ws("".join(cur))
        if merged:
            chunks.append(merged)
        cur = []

    def walk(node: Tag) -> None:
        for c in node.children:
            if isinstance(c, NavigableString):
                raw = str(c)
                if not raw:
                    continue
                pieces = _SPLIT_DELIM_PATTERN.split(raw)
                for piece in pieces:
                    if not piece:
                        continue
                    if _SPLIT_DELIM_PATTERN.fullmatch(piece):
                        flush()
                    else:
                        cur.append(piece)
            elif isinstance(c, Tag):
                if c.name in ("script", "style"):
                    continue
                cur.append(c.get_text("", separator=""))

    walk(elem)
    flush()
    return chunks


def _p_to_text_and_images_ordered(p: Tag) -> List[Union[Tuple[str, str], Tuple[str, Tag]]]:
    """按文档序分解 <p>：('text', 文本片段) 或 ('img', img 节点)。"""
    out: List[Union[Tuple[str, str], Tuple[str, Tag]]] = []
    cur: List[str] = []

    def flush_text() -> None:
        nonlocal cur
        if not cur:
            return
        merged = _normalize_ws("".join(cur))
        if merged:
            out.append(("text", merged))
        cur = []

    def walk_text_node(raw: str) -> None:
        if not raw:
            return
        pieces = _SPLIT_DELIM_PATTERN.split(raw)
        for piece in pieces:
            if not piece:
                continue
            if _SPLIT_DELIM_PATTERN.fullmatch(piece):
                flush_text()
            else:
                cur.append(piece)

    for child in p.children:
        if isinstance(child, NavigableString):
            walk_text_node(str(child))
        elif isinstance(child, Tag):
            if child.name in ("script", "style"):
                continue
            if child.name == "img":
                flush_text()
                out.append(("img", child))
                continue
            cur.append(child.get_text("", separator=""))
    flush_text()
    return out


def _is_math_block_div(tag: Tag) -> bool:
    classes = tag.get("class") or []
    if isinstance(classes, str):
        classes = classes.split()
    return MATH_BLOCK_CLASS in classes


def _wrap_fragment(html: str) -> str:
    t = html.strip()
    if not t:
        return "<div></div>"
    return f"<div id='__root__'>{html}</div>"


def _iter_block_nodes(root: Tag) -> List[Tag]:
    """文档序收集块级单元（跳过 script/style）。"""
    r = root.find(id="__root__") or root
    out: List[Tag] = []

    def recurse(el: Tag) -> None:
        for child in el.children:
            if not isinstance(child, Tag):
                continue
            if child.name in ("script", "style"):
                continue
            name = child.name.lower()
            if name in ("h1", "h2", "h3", "h4", "h5", "h6"):
                out.append(child)
            elif name == "p":
                out.append(child)
            elif name == "table":
                out.append(child)
            elif name == "div" and _is_math_block_div(child):
                out.append(child)
            elif name == "div":
                recurse(child)
            elif name in ("section", "article", "main", "body", "span"):
                recurse(child)
            else:
                out.append(child)

    recurse(r)
    return out


def parse_markdown_html_to_child_blocks(html_text: str, heading_state: Optional[_HeadingState] = None) -> List[ChildBlock]:
    """
    解析整段 HTML（可来自 .md 文件），按文档序生成子块列表。
    """
    if heading_state is None:
        heading_state = _HeadingState()

    soup = BeautifulSoup(_wrap_fragment(html_text), "html.parser")
    root = soup.find(id="__root__") or soup
    nodes = _iter_block_nodes(root)
    children: List[ChildBlock] = []

    for node in nodes:
        name = node.name.lower()
        if name in ("h1", "h2", "h3", "h4", "h5", "h6"):
            hid = heading_state.new_id()
            for piece in _split_mixed_html(node):
                children.append(
                    ChildBlock(
                        text=piece,
                        block_kind="text",
                        is_from_heading=True,
                        heading_elem_id=hid,
                        source_tag=name,
                    )
                )
        elif name == "p":
            for kind, payload in _p_to_text_and_images_ordered(node):
                if kind == "text":
                    children.append(
                        ChildBlock(
                            text=payload,
                            block_kind="text",
                            source_tag="p",
                        )
                    )
                else:
                    img = payload
                    alt = (img.get("alt") or "").strip()
                    if not alt:
                        alt = (img.get("src") or "").strip() or "[image]"
                    children.append(
                        ChildBlock(text=alt, block_kind="image", source_tag="img")
                    )
        elif name == "table":
            children.append(
                ChildBlock(text=node.get_text(" ", strip=True), block_kind="table", source_tag="table")
            )
        elif name == "div" and _is_math_block_div(node):
            children.append(
                ChildBlock(
                    text=node.get_text(" ", strip=True),
                    block_kind="math",
                    source_tag="div.math-block",
                )
            )
        else:
            # 其它未知块：整体按混排规则切分，视为正文
            for piece in _split_mixed_html(node):
                children.append(
                    ChildBlock(text=piece, block_kind="text", source_tag=name or "unknown")
                )

    return children


def _is_context_candidate(c: ChildBlock) -> bool:
    """可用于「最近 <p> 或标题」上下文的子块。"""
    if c.block_kind != "text":
        return False
    return True


def _nearest_prev_context(children: Sequence[ChildBlock], idx: int) -> Optional[int]:
    j = idx - 1
    while j >= 0:
        if _is_context_candidate(children[j]):
            return j
        j -= 1
    return None


def _nearest_next_context(children: Sequence[ChildBlock], idx: int) -> Optional[int]:
    j = idx + 1
    n = len(children)
    while j < n:
        if _is_context_candidate(children[j]):
            return j
        j += 1
    return None


def _needs_forced_context(c: ChildBlock) -> bool:
    return c.block_kind in ("table", "image", "math")


def _expand_for_specials(children: Sequence[ChildBlock], indices: Set[int]) -> Set[int]:
    out = set(indices)
    for i in sorted(indices):
        c = children[i]
        if not _needs_forced_context(c):
            continue
        pa = _nearest_prev_context(children, i)
        pb = _nearest_next_context(children, i)
        if pa is not None:
            out.add(pa)
        if pb is not None:
            out.add(pb)
    return out


def _contiguous_span(idxs: Set[int]) -> Set[int]:
    """将下标集合闭包为连续区间 [min,max]，避免表格/图片上下文之间出现空洞。"""
    if not idxs:
        return set()
    lo, hi = min(idxs), max(idxs)
    return set(range(lo, hi + 1))


def _has_rich_block(children: Sequence[ChildBlock], idxs: Set[int]) -> bool:
    return any(children[i].block_kind in ("table", "image", "math") for i in idxs)


def _clip_second_heading(children: Sequence[ChildBlock], span: Set[int]) -> Set[int]:
    """遇第二个同级标题则截断：保留至新同级标题前的连续区间。"""
    ordered = sorted(span)
    if not ordered:
        return span
    seen_levels: Set[str] = set()
    cut: Optional[int] = None
    for i in ordered:
        if children[i].heading_elem_id is None:
            continue
        level = children[i].source_tag
        if not seen_levels:
            seen_levels.add(level)
            continue
        if level not in seen_levels:
            seen_levels.add(level)
            continue
        cut = i
        break
    if cut is None:
        return span
    lo = min(span)
    return set(range(lo, cut))


def _heading_ids_in_set(children: Sequence[ChildBlock], idxs: Set[int]) -> Set[str]:
    levels: Set[str] = set()
    for i in idxs:
        if children[i].heading_elem_id is not None:
            levels.add(children[i].source_tag)
    return levels


def _char_len_for_indices(children: Sequence[ChildBlock], idxs: Sequence[int]) -> int:
    return sum(len(children[i].text) for i in idxs)


def build_parent_indices(
    children: List[ChildBlock],
    max_chars: int = PARENT_MAX_CHARS,
) -> List[List[int]]:
    """
    按文档序生成分组前的父块（子块下标列表），满足：
    - 默认不超过 max_chars；若块内含表格/图片/段落公式，可突破上限以保留强制上下文；
    - 每个父块至多一个同级标题元素（遇第二个同级标题截断）；
    - 含表格/图片/段落公式时闭包包含最近上下文本/标题子块，并取连续区间。
    """
    n = len(children)
    if n == 0:
        return []

    parents: List[List[int]] = []
    pos = 0

    while pos < n:
        cur = _contiguous_span(_expand_for_specials(children, {pos}))
        cur = _clip_second_heading(children, cur)
        j = pos + 1
        while j < n:
            trial = _contiguous_span(_expand_for_specials(children, cur | {j}))
            trial = _clip_second_heading(children, trial)
            if trial == cur:
                break
            if len(_heading_ids_in_set(children, trial)) > 1:
                break
            new_chars = _char_len_for_indices(children, sorted(trial))
            if new_chars > max_chars and not _has_rich_block(children, trial):
                break
            cur = trial
            j += 1

        ordered = sorted(cur)
        if not ordered:
            pos += 1
            continue
        parents.append(ordered)
        pos = ordered[-1] + 1

    return parents


def _build_overlap_tail(prev: List[int], children: Sequence[ChildBlock], max_chars: int) -> List[int]:
    tail: List[int] = []
    total = 0
    for idx in reversed(prev):
        length = len(children[idx].text)
        if not tail:
            tail.insert(0, idx)
            total = length
            continue
        if total + length > max_chars:
            break
        tail.insert(0, idx)
        total += length
    return tail


def apply_parent_overlap(
    parent_indices: List[List[int]],
    children: Sequence[ChildBlock],
    overlap_max_chars: int = OVERLAP_MAX_CHARS,
) -> List[List[int]]:
    """相邻父块按字符上限重叠末尾子块，至少保留一个重叠子块。"""
    if not parent_indices:
        return []
    out: List[List[int]] = [list(parent_indices[0])]
    for k in range(1, len(parent_indices)):
        prev = out[-1]
        cur = list(parent_indices[k])
        tail = _build_overlap_tail(prev, children=children, max_chars=overlap_max_chars)
        out.append(tail + cur)
    return out


def indices_to_parent_texts(children: List[ChildBlock], parent_indices: List[List[int]]) -> List[str]:
    texts: List[str] = []
    for idxs in parent_indices:
        parts = [children[i].text for i in idxs]
        texts.append("\n\n".join(parts))
    return texts


def build_child_to_parent_map(
    num_children: int,
    parent_indices: List[List[int]],
) -> List[List[int]]:
    """
    每个子块对应其出现过的所有父块 id 列表（已去重）。
    若因父块重叠同一子块出现在多个父块中，则保留所有父块 id，按父块索引升序排列。
    """
    mapping: List[List[int]] = [[] for _ in range(num_children)]
    for pid, idxs in enumerate(parent_indices):
        for i in idxs:
            if 0 <= i < num_children:
                bucket = mapping[i]
                if not bucket or bucket[-1] != pid:
                    # parent_indices 本身按父块顺序遍历，且每个父块内部下标已去重，
                    # 只需避免连续重复，即可得到有序且去重的父块 id 列表。
                    bucket.append(pid)
    return mapping


def chunk_markdown_html(
    html_text: str,
    max_parent_chars: int = PARENT_MAX_CHARS,
    overlap_children: int = OVERLAP_CHILD_COUNT,
) -> Dict[str, Any]:
    """
    完整流程：解析 → 子块 → 父块（含上下文）→ 重叠 → 输出。

    返回 dict:
      - child_blocks: List[ChildBlock]
      - parent_texts: List[str]  # 最终父块全文
      - parent_child_indices: List[List[int]]  # 每个父块包含的子块下标（重叠后）
      - child_to_parent: List[List[int]]  # 每个子块对应的父块 id 列表（可能多个）
    """
    kids = parse_markdown_html_to_child_blocks(html_text)
    raw_parents = build_parent_indices(kids, max_chars=max_parent_chars)
    overlapped = apply_parent_overlap(raw_parents, children=kids, overlap_max_chars=overlap_children)
    parent_texts = indices_to_parent_texts(kids, overlapped)
    child_to_parent = build_child_to_parent_map(len(kids), overlapped)
    return {
        "child_blocks": kids,
        "parent_texts": parent_texts,
        "parent_child_indices": overlapped,
        "child_to_parent": child_to_parent,
    }


# --- 可选：与示例工程一致的向量入库（需 sentence_transformers / faiss / numpy）---
def optional_build_faiss(
    result: Dict[str, Any],
    embed_model_path: str,
    faiss_index_path: str = "faiss_parent_child_index.bin",
    child_data_path: str = "child_data.pkl",
    parent_data_path: str = "parent_data.pkl",
    device: str = "cpu",
    batch_size: int = 32,
) -> None:
    import pickle

    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer

    kids: List[ChildBlock] = result["child_blocks"]
    child_texts = [c.text for c in kids]
    parent_texts: List[str] = result["parent_texts"]

    model = SentenceTransformer(embed_model_path, device=device)
    dim = model.get_sentence_embedding_dimension()
    index = faiss.IndexFlatIP(dim)
    # CPU 环境下：分批编码并增量写入，避免把所有向量先拼进内存
    for i in range(0, len(child_texts), batch_size):
        batch = child_texts[i : i + batch_size]
        emb = model.encode(batch, normalize_embeddings=True)
        vecs = np.asarray(emb, dtype="float32")
        index.add(vecs)
    faiss.write_index(index, faiss_index_path)

    child_data = {
        "chunks": child_texts,
        "parent_ids": result["child_to_parent"],
    }
    with open(child_data_path, "wb") as f:
        pickle.dump(child_data, f)
    with open(parent_data_path, "wb") as f:
        pickle.dump(parent_texts, f)


if __name__ == "__main__":
    import json
    import os

    sample = """
<h2>绪论与背景。第二节展开。</h2>
<p>本节说明模型假设；并给出边界条件！后续见表。</p>
<table><tr><td>A</td><td>B</td></tr></table>
<p>表后是总结？还需要验证。</p>
<div class="math-block">E = mc^2</div>
<p>公式下方段落。</p>
<p><img src="fig1.png" alt="燃烧波示意图"/></p>
"""

    out = chunk_markdown_html(sample)
    print(json.dumps({k: v for k, v in out.items() if k != "child_blocks"}, ensure_ascii=False, indent=2))
    for i, c in enumerate(out["child_blocks"]):
        print(i, c.block_kind, repr(c.text[:80]))
