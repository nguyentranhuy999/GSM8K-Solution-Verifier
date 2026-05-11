#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from getpass import getpass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from graph_model import (
    Edge,
    Node,
    build_debug_payload,
    print_preview,
    push_to_neo4j,
    write_debug_json,
)
from llm_openrouter import (
    OpenRouterError,
    extract_constructed_text,
    extract_oop_problem,
    extract_structured_problem,
)
from parsers.legacy_json_parser import extract_graph_from_structured
from parsers.legacy_text_parser import extract_graph_from_constructed_text
from parsers.oop_parser import extract_graph_from_oop_model
from parsers.semantic_parser import SemanticParser


OOP_PARSERS = {"oop", "llm", "llm_oop"}
HYBRID_OOP_PARSERS = {"hybrid", "hybrid_oop"}
SEMANTIC_PARSERS = {"semantic"}
LEGACY_JSON_PARSERS = {"legacy_json", "llm_json"}
HYBRID_LEGACY_JSON_PARSERS = {"hybrid_json"}
LEGACY_TEXT_PARSERS = {"legacy_text", "llm_text"}
HYBRID_LEGACY_TEXT_PARSERS = {"hybrid_text"}


def parse_config(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Config file must be a JSON object: {path}")
    return data


def parser_choices() -> Tuple[str, ...]:
    return tuple(
        sorted(
            OOP_PARSERS
            | HYBRID_OOP_PARSERS
            | SEMANTIC_PARSERS
            | LEGACY_JSON_PARSERS
            | HYBRID_LEGACY_JSON_PARSERS
            | LEGACY_TEXT_PARSERS
            | HYBRID_LEGACY_TEXT_PARSERS
        )
    )


def parse_args() -> argparse.Namespace:
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument("--config", type=Path, default=Path("local_config.json"))
    base_args, _ = base_parser.parse_known_args()
    config = parse_config(base_args.config)

    parser = argparse.ArgumentParser(
        description="Encode a math word problem as objects/relationships and optionally write it to Neo4j."
    )
    parser.add_argument("--config", type=Path, default=base_args.config)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(str(config.get("input", "Input/problem.txt"))),
    )
    parser.add_argument(
        "--parser",
        choices=parser_choices(),
        default=str(config.get("parser", "hybrid_oop")),
        help="Use oop/hybrid_oop for the main OOP formalizer. Legacy parser modes are kept for comparison.",
    )
    parser.add_argument(
        "--document-id",
        default=str(config.get("document_id", "problem_txt")),
    )

    parser.add_argument(
        "--openrouter-model",
        default=str(config.get("openrouter_model", os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"))),
    )
    parser.add_argument(
        "--openrouter-api-key",
        default=str(config.get("openrouter_api_key", os.getenv("OPENROUTER_API_KEY", ""))).strip() or None,
    )
    parser.add_argument(
        "--openrouter-timeout",
        type=int,
        default=int(config.get("openrouter_timeout", os.getenv("OPENROUTER_TIMEOUT", "90"))),
    )
    parser.add_argument(
        "--openrouter-site-url",
        default=str(config.get("openrouter_site_url", os.getenv("OPENROUTER_SITE_URL", ""))).strip() or None,
    )
    parser.add_argument(
        "--openrouter-app-name",
        default=str(config.get("openrouter_app_name", os.getenv("OPENROUTER_APP_NAME", ""))).strip() or None,
    )

    parser.add_argument(
        "--oop-out",
        type=Path,
        default=Path(str(config.get("oop_out", "Formalizer/oop_model.json"))),
    )
    parser.add_argument(
        "--compact-out",
        type=Path,
        default=Path(str(config.get("compact_out", "Formalizer/compact_model.txt"))),
    )
    parser.add_argument(
        "--constructed-out",
        type=Path,
        default=Path(str(config.get("constructed_out", "Formalizer/constructed_summary.txt"))),
    )
    parser.add_argument(
        "--debug-out",
        type=Path,
        default=Path(str(config.get("debug_out", "Formalizer/graph_debug.json"))),
    )
    parser.add_argument("--no-debug-out", action="store_true")

    parser.add_argument(
        "--uri",
        default=str(config.get("uri", os.getenv("NEO4J_URI", "neo4j://localhost:7687"))),
    )
    parser.add_argument(
        "--user",
        default=str(config.get("user", os.getenv("NEO4J_USER", "neo4j"))),
    )
    parser.add_argument(
        "--password",
        default=str(config.get("password", os.getenv("NEO4J_PASSWORD", ""))).strip() or None,
    )
    parser.add_argument(
        "--database",
        default=str(config.get("database", os.getenv("NEO4J_DATABASE", "neo4j"))),
    )
    parser.add_argument("--clear", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    argv = set(os.sys.argv[1:])
    if "--clear" not in argv and bool(config.get("clear", False)):
        args.clear = True
    if "--dry-run" not in argv and bool(config.get("dry_run", False)):
        args.dry_run = True
    if bool(config.get("no_debug_out", False)):
        args.no_debug_out = True
    return args


def require_openrouter(args: argparse.Namespace) -> None:
    if not args.openrouter_api_key:
        raise OpenRouterError("Missing OpenRouter API key. Set OPENROUTER_API_KEY or openrouter_api_key in local_config.json.")
    if not args.openrouter_model:
        raise OpenRouterError("Missing OpenRouter model name.")


def save_json(path: Optional[Path], payload: object, label: str) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[info] {label} saved: {path}")


def save_text(path: Optional[Path], payload: str, label: str) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload.rstrip() + "\n", encoding="utf-8")
    print(f"[info] {label} saved: {path}")


def run_oop_parser(args: argparse.Namespace, content: str) -> Tuple[str, Dict[str, Node], List[Edge], List[str]]:
    require_openrouter(args)
    oop_model = extract_oop_problem(
        content,
        api_key=args.openrouter_api_key,
        model=args.openrouter_model,
        timeout_sec=args.openrouter_timeout,
        site_url=args.openrouter_site_url,
        app_name=args.openrouter_app_name,
    )
    save_json(args.oop_out, oop_model, "OOP model")
    compact = oop_model.get("compact")
    if isinstance(compact, str) and compact.strip():
        save_text(args.compact_out, compact.strip(), "Compact model")
    nodes, edges, sentences = extract_graph_from_oop_model(
        oop_model,
        problem_id=args.document_id,
        problem_text=content,
    )
    return f"openrouter-oop:{args.openrouter_model}", nodes, edges, sentences


def run_semantic_parser(args: argparse.Namespace, content: str) -> Tuple[str, Dict[str, Node], List[Edge], List[str]]:
    parser = SemanticParser()
    nodes, edges, sentences = parser.extract(content, problem_id=args.document_id)
    return parser.name, nodes, edges, sentences


def run_legacy_json_parser(args: argparse.Namespace, content: str) -> Tuple[str, Dict[str, Node], List[Edge], List[str]]:
    require_openrouter(args)
    structured = extract_structured_problem(
        content,
        api_key=args.openrouter_api_key,
        model=args.openrouter_model,
        timeout_sec=args.openrouter_timeout,
        site_url=args.openrouter_site_url,
        app_name=args.openrouter_app_name,
    )
    nodes, edges, sentences = extract_graph_from_structured(
        structured,
        problem_id=args.document_id,
        problem_text=content,
    )
    return f"openrouter-legacy-json:{args.openrouter_model}", nodes, edges, sentences


def run_legacy_text_parser(args: argparse.Namespace, content: str) -> Tuple[str, Dict[str, Node], List[Edge], List[str]]:
    require_openrouter(args)
    constructed_text = extract_constructed_text(
        content,
        api_key=args.openrouter_api_key,
        model=args.openrouter_model,
        timeout_sec=args.openrouter_timeout,
        site_url=args.openrouter_site_url,
        app_name=args.openrouter_app_name,
    )
    if args.constructed_out:
        args.constructed_out.parent.mkdir(parents=True, exist_ok=True)
        args.constructed_out.write_text(constructed_text + "\n", encoding="utf-8")
        print(f"[info] Constructed summary saved: {args.constructed_out}")
    nodes, edges, sentences = extract_graph_from_constructed_text(
        constructed_text,
        problem_id=args.document_id,
        problem_text=content,
    )
    return f"openrouter-legacy-text:{args.openrouter_model}", nodes, edges, sentences


def run_parser(args: argparse.Namespace, content: str) -> Tuple[str, Dict[str, Node], List[Edge], List[str], Optional[Exception]]:
    llm_error: Optional[Exception] = None

    if args.parser in SEMANTIC_PARSERS:
        backend, nodes, edges, sentences = run_semantic_parser(args, content)
        return backend, nodes, edges, sentences, llm_error

    try:
        if args.parser in OOP_PARSERS | HYBRID_OOP_PARSERS:
            backend, nodes, edges, sentences = run_oop_parser(args, content)
        elif args.parser in LEGACY_JSON_PARSERS | HYBRID_LEGACY_JSON_PARSERS:
            backend, nodes, edges, sentences = run_legacy_json_parser(args, content)
        elif args.parser in LEGACY_TEXT_PARSERS | HYBRID_LEGACY_TEXT_PARSERS:
            backend, nodes, edges, sentences = run_legacy_text_parser(args, content)
        else:
            raise ValueError(f"Unsupported parser: {args.parser}")
        return backend, nodes, edges, sentences, llm_error
    except Exception as exc:
        llm_error = exc
        if args.parser in OOP_PARSERS | LEGACY_JSON_PARSERS | LEGACY_TEXT_PARSERS:
            raise
        print(f"[warn] LLM parser failed, fallback to semantic parser: {exc}")
        backend, nodes, edges, sentences = run_semantic_parser(args, content)
        return backend, nodes, edges, sentences, llm_error


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    content = args.input.read_text(encoding="utf-8")
    if not content.strip():
        print(f"[warn] Input file is empty: {args.input}")
        return

    backend_name, nodes, edges, sentences, llm_error = run_parser(args, content)
    print_preview(backend_name, sentences, nodes, edges)
    write_debug_json(
        None if args.no_debug_out else args.debug_out,
        build_debug_payload(
            backend_name=backend_name,
            problem_id=args.document_id,
            problem_text=content,
            sentences=sentences,
            nodes=nodes,
            edges=edges,
            llm_error=llm_error,
        ),
    )

    if args.dry_run:
        print("[info] Dry run enabled, skip Neo4j write.")
        return

    if not args.password:
        args.password = getpass("Neo4j password: ").strip()
    if not args.password:
        raise ValueError("Missing Neo4j password. Pass --password or set it in local_config.json.")

    push_to_neo4j(
        uri=args.uri,
        user=args.user,
        password=args.password,
        database=args.database,
        nodes=nodes,
        edges=edges,
        clear_db=args.clear,
    )
    print("[done] Problem graph has been written to Neo4j.")
    print("[hint] In Neo4j Browser, run: MATCH (n)-[r]->(m) RETURN n,r,m")


if __name__ == "__main__":
    main()
