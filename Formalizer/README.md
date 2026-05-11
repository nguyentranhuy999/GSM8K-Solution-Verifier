# Formalizer - OOP Problem Graph to Neo4j

Script `build_problem_graph.py` reads text from `Input/problem.txt`, encodes the problem as OOP-style classes/objects/relationships, and optionally writes the graph into Neo4j.
The recommended flow is: `raw problem -> compact expression + OOP JSON -> debug JSON -> Neo4j graph`.

## 1) Install dependencies

```bash
python3 -m pip install -r requirements.txt
```

The semantic fallback parser is rule-based and does not require a spaCy model.

## 2) Configure once

Edit this file:

`local_config.json`

Default values are set for Neo4j Desktop local instance:
- `uri`: `neo4j://127.0.0.1:7687`
- `database`: `formalizer` or your Neo4j database name
- `clear`: `true` (rebuild graph each run)
- `parser`: `hybrid_oop` (try OpenRouter OOP JSON first, fallback to semantic rules)

If `password` is left empty, script will prompt password at runtime.

### OpenRouter settings (optional but recommended)

In `local_config.json`:
- `parser`: `llm_oop` or `hybrid_oop`
- `openrouter_model`: e.g. `openai/gpt-4o-mini`
- `openrouter_api_key`: your key (or leave empty and use env var)
- `oop_out`: where raw OOP classes/objects JSON is saved
- `compact_out`: where the short human-readable expression is saved
- `debug_out`: where parsed graph debug JSON is saved after every run

Or set env var:

```bash
export OPENROUTER_API_KEY=your_key
export OPENROUTER_MODEL=openai/gpt-4o-mini
```

## 3) Run extraction only (no Neo4j write)

```bash
python3 Formalizer/build_problem_graph.py --dry-run
```

## 4) Write graph to Neo4j

```bash
python3 Formalizer/build_problem_graph.py \
  --input Input/problem.txt \
  --uri neo4j://localhost:7687 \
  --user neo4j \
  --password your_password \
  --database neo4j \
  --clear
```

Force LLM OOP mode:

```bash
python3 Formalizer/build_problem_graph.py --parser llm_oop
```

Hybrid OOP mode (recommended):

```bash
python3 Formalizer/build_problem_graph.py --parser hybrid_oop
```

Legacy relation JSON mode, kept only for comparison:

```bash
python3 Formalizer/build_problem_graph.py --parser llm_json
```

Legacy constructed-text mode, kept only for comparison:

```bash
python3 Formalizer/build_problem_graph.py --parser hybrid_text
```

### Debug outputs

After each run, open:

```text
Formalizer/graph_debug.json
```

This file shows the parsed `classes`, `objects`, `relationships`, backend used, and any LLM fallback error.

When the LLM OOP parser succeeds, open:

```text
Formalizer/compact_model.txt
Formalizer/oop_model.json
```

`compact_model.txt` is the short view for humans, for example:

```text
target:20day(1day(Nancy(double_espresso($3),iced_coffee($2.5))))
```

`oop_model.json` is the machine-readable class/object model returned by OpenRouter before it is converted into Neo4j graph nodes.

## Parser files

The file layout is intentionally split by responsibility:

```text
Formalizer/
  build_problem_graph.py      # main runner/orchestrator
  graph_model.py              # Node, Edge, Neo4j write, debug JSON
  llm_openrouter.py           # OpenRouter calls and prompts
  parsers/
    oop_parser.py             # main parser: OOP JSON -> graph
    semantic_parser.py        # rule-based fallback
    legacy_json_parser.py     # old JSON parser for comparison
    legacy_text_parser.py     # old DSL parser for comparison
```

### Legacy constructed text DSL

`llm_text` / `hybrid_text` asks OpenRouter to output lines like:

```text
ENTITY|Person|An
ENTITY|Person|Binh
ENTITY|Item|Keo
FACT|HAS_QUANTITY|Person:An|Item:Keo|50|keo|-
FACT|LESS_THAN_BY|Person:Binh|Person:An|3|keo|-
QUESTION|ASKS_FOR|Person:Binh|Item:Keo|-|-|-
```

This text is saved to `Formalizer/constructed_summary.txt` (or path from `constructed_out`) and then parsed into graph.

You can also use environment variables:

```bash
export NEO4J_URI=neo4j://localhost:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=your_password
export NEO4J_DATABASE=neo4j
python3 Formalizer/build_problem_graph.py --clear
```

## 5) One-click Run in VS Code

This repo includes `.vscode/launch.json` with 3 profiles:
- `Formalizer: Dry Run`
- `Formalizer: Import to Neo4j`
- `Formalizer: Import via OpenRouter`

Open **Run and Debug** in VS Code, choose one profile, then press **Run**.

## 6) Visualize in Neo4j Browser

Run this Cypher query:

```cypher
MATCH (n)-[r]->(m) RETURN n, r, m
```

## Graph model

OOP mode creates:
- `(:Class)` nodes for class definitions
- one node per problem object, labeled by its class name, e.g. `(:Person)`, `(:Item)`, `(:Quantity)`
- `(:Problem)-[:CONTAINS_OBJECT]->(:Object)`
- `(:Object)-[:INSTANCE_OF]->(:Class)`
- relationships returned by the LLM, e.g. `HAS_QUANTITY`, `HAS_PRICE`, `OCCURS_EVERY`, `ASKS_FOR`

The OOP prompt is intentionally a formalizer only. It asks the LLM not to create solution steps, formulas, final answers, or computation plans.
