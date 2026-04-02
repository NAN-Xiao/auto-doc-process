#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""文档拆分入口。"""

from pathlib import Path
from typing import Optional

from ..core.config import load_processor_config as load_config
from ..core.logger import Logger
from .contracts import DocumentInfo
from .pdf_splitter import PDFSplitter
from .splitter_base import (
    DocumentSplitter,
    generate_smart_image_name_simple,
    generate_smart_image_name_with_llm,
)
from .word_splitter import WordSplitter


def generate_output_path(input_path: Path, root_dir: str = "processed", **_kwargs) -> Path:
    """生成输出目录路径。"""
    from ..core.utils import safe_filename

    doc_name = safe_filename(input_path.stem, input_path.stem)
    base_dir = Path(root_dir) if Path(root_dir).is_absolute() else input_path.parent / root_dir
    return base_dir / doc_name


def process_document(
    input_path: Path,
    output_dir: Path = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    use_llm_naming: bool = False,
    config: dict = None,
) -> Optional[DocumentInfo]:
    """处理单个文档。"""

    if config is None:
        config = load_config()

    suffix = input_path.suffix.lower()

    if output_dir is None:
        from ..core.config import MODULE_DIR
        paths = config.get("paths", {})
        processed_dir_raw = paths.get("processed_dir", "")
        if processed_dir_raw:
            path = Path(processed_dir_raw)
            base_dir = path if path.is_absolute() else (MODULE_DIR / path).resolve()
        else:
            documents_dir = paths.get("documents_dir", input_path.parent)
            docs_path = Path(documents_dir)
            base_dir = docs_path if docs_path.is_absolute() else (MODULE_DIR / docs_path).resolve()
            base_dir = base_dir / "processed"

        output_dir = generate_output_path(input_path, str(base_dir))

    try:
        if suffix == ".pdf":
            splitter = PDFSplitter(chunk_size, chunk_overlap, use_llm_naming, config=config)
            return splitter.process(input_path, output_dir)
        if suffix in [".docx", ".doc"]:
            if suffix == ".doc":
                Logger.warning(f".doc格式需要转换为.docx，跳过: {input_path.name}")
                return None
            splitter = WordSplitter(chunk_size, chunk_overlap, use_llm_naming, config=config)
            return splitter.process(input_path, output_dir)

        Logger.warning(f"不支持的文件格式: {suffix}")
        return None
    except ValueError as exc:
        Logger.warning(f"跳过: {input_path.name} - {exc}")
        return None
    except Exception as exc:
        Logger.error(f"处理失败 {input_path.name}: {exc}")
        import traceback

        traceback.print_exc()
        return None


__all__ = [
    "DocumentSplitter",
    "PDFSplitter",
    "WordSplitter",
    "generate_smart_image_name_simple",
    "generate_smart_image_name_with_llm",
    "generate_output_path",
    "process_document",
]
