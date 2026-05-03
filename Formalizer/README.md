# Formalizer - Semantic Graph to Neo4j

Script `build_problem_graph.py` reads text from `Input/problem.txt`, extracts semantic relations from math word problems, and writes a graph into Neo4j.
It supports a 2-step LLM flow: `raw problem -> constructed summary text -> graph`.

## 1) Install dependencies

```bash
python3 -m pip install -r Formalizer/requirements.txt
python3 -m spacy download en_core_web_sm
```

`en_core_web_sm` is optional. If not available, the script will fallback to rule-based extraction.

## 2) Configure once

Edit this file:

`Formalizer/local_config.json`

Default values are set for Neo4j Desktop local instance:
- `uri`: `neo4j://127.0.0.1:7687`
- `database`: `neo4j`
- `clear`: `true` (rebuild graph each run)
- `parser`: `hybrid_text` (try OpenRouter constructed text first, fallback to semantic rules)

If `password` is left empty, script will prompt password at runtime.

### OpenRouter settings (optional but recommended)

In `Formalizer/local_config.json`:
- `parser`: `llm_text` or `hybrid_text`
- `openrouter_model`: e.g. `openai/gpt-4o-mini`
- `openrouter_api_key`: your key (or leave empty and use env var)
- `constructed_out`: where constructed summary text is saved

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

Force LLM JSON-only mode:

```bash
python3 Formalizer/build_problem_graph.py --parser llm
```

Force LLM constructed-text mode:

```bash
python3 Formalizer/build_problem_graph.py --parser llm_text
```

Hybrid JSON mode (LLM JSON first, fallback semantic):

```bash
python3 Formalizer/build_problem_graph.py --parser hybrid
```

Hybrid constructed-text mode (recommended):

```bash
python3 Formalizer/build_problem_graph.py --parser hybrid_text
```

### Constructed text DSL

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

Nodes:
- `(:Problem {key, name})`
- `(:Person {key, name})`
- `(:Item {key, name})`
- `(:Period {key, name})`
- `(:Quantity {key, name, amount, unit, kind})`

Main relationships:
- `(:Person)-[:HAS_QUANTITY]->(:Item)`
- `(:Person)-[:GAINS_QUANTITY]->(:Item)`
- `(:Person)-[:LOSES_QUANTITY]->(:Item)`
- `(:Person)-[:TRANSFER_TO]->(:Person)` with property `item`
- `(:Person|:Problem)-[:ASKS_FOR]->(:Item)`
- `(:Person)-[:RATE_MULTIPLIER]->(:Person)`
- `(:Person)-[:LESS_THAN_BY]->(:Person)` (difference relation from constructed text)
- `(:Person)-[:MORE_THAN_BY]->(:Person)` (difference relation from constructed text)
- `(:Person)-[:TIMES_OF]->(:Person)` (multiplicative relation from constructed text)
- `(:Item)-[:UNIT_RATE]->(:Item)` (example: `show -> minute`, value `50`)
- `(:Person)-[:OWNS_ITEM]->(:Item)`
- `(:Person)-[:HAS_VALUE]->(:Item)` (example: bought for `20000 USD`)
- `(:Item)-[:VALUE_DECREASE_PER_PERIOD]->(:Period)` (example: `1000 USD / year`)
- `(:Person|:Problem)-[:ASKS_WORTH_AFTER]->(:Item)` (example: after `6 year`)
- `(:Person|:Item)-[:HAS_QUANTITY_VALUE]->(:Quantity)-[:DESCRIBES]->(:Person|:Item)`
