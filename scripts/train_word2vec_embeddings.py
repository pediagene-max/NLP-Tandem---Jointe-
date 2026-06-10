#!/usr/bin/env python3
"""Train Word2Vec document embeddings from the project CSV datasets.

The script keeps the output format compatible with the existing R clustering
scripts: numeric embedding columns followed by the original label column.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from gensim.models import Word2Vec


TOKEN_PATTERN = re.compile(r"[a-zA-Z][a-zA-Z']+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Word2Vec and export one averaged embedding per document."
    )
    parser.add_argument("--input", required=True, help="Input CSV path.")
    parser.add_argument("--output", required=True, help="Output embeddings CSV path.")
    parser.add_argument("--text-column", default="text", help="Text column name.")
    parser.add_argument("--label-column", default="label", help="Label column name.")
    parser.add_argument("--vector-size", type=int, default=300)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--min-count", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--save-model",
        default=None,
        help="Optional path for saving the trained Word2Vec model.",
    )
    return parser.parse_args()


def tokenize(text: object) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(str(text))]


def document_vector(model: Word2Vec, tokens: list[str], vector_size: int) -> np.ndarray:
    vectors = [model.wv[token] for token in tokens if token in model.wv]
    if not vectors:
        return np.zeros(vector_size, dtype=np.float32)
    return np.mean(vectors, axis=0)


def validate_columns(frame: pd.DataFrame, text_column: str, label_column: str) -> None:
    missing = [name for name in (text_column, label_column) if name not in frame.columns]
    if missing:
        available = ", ".join(frame.columns)
        raise ValueError(f"Missing columns {missing}. Available columns: {available}")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    frame = pd.read_csv(input_path)
    validate_columns(frame, args.text_column, args.label_column)

    tokenized_documents = [tokenize(text) for text in frame[args.text_column]]
    non_empty_documents = [tokens for tokens in tokenized_documents if tokens]
    if not non_empty_documents:
        raise ValueError(f"No tokens were found in {input_path}.")

    model = Word2Vec(
        sentences=non_empty_documents,
        vector_size=args.vector_size,
        window=args.window,
        min_count=args.min_count,
        workers=args.workers,
        seed=args.seed,
        sg=1,
        epochs=args.epochs,
    )

    vectors = np.vstack(
        [
            document_vector(model, tokens, args.vector_size)
            for tokens in tokenized_documents
        ]
    )
    output = pd.DataFrame(vectors, columns=[str(i) for i in range(args.vector_size)])
    output[args.label_column] = frame[args.label_column].to_numpy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)

    if args.save_model:
        model_path = Path(args.save_model)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(model_path))

    print(f"Wrote {len(output)} document embeddings to {output_path}")


if __name__ == "__main__":
    main()
