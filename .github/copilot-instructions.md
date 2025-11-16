<!-- Copilot instructions for repo-specific agent behavior -->
# PokéAgent — AI Assistant Guidance

This file explains repository-specific architecture, conventions, and developer workflows so an AI coding assistant can be immediately productive.

- Project layout (key files):
  - `run.py`: CLI entry. Starts a multiprocess server (`server.app`) and a client. See flags like `--backend`, `--model-name`, `--simple`, `--agent-auto`, `--no-ocr`, and `--record`.
  - `agent/`: Four-module agent architecture. Edit these files to change agent behavior:
    - `agent/system_prompt.py` — global prompt and prompt-building helpers.
    - `agent/perception.py`, `agent/planning.py`, `agent/memory.py`, `agent/action.py` — module interfaces and in-repo examples.
    - `agent/simple.py` — lightweight mode that bypasses the four-module pipeline.
  - `utils/`: Cross-cutting helpers and VLM backends:
    - `utils/vlm.py` — VLM backend adapters. Add new model providers here.
    - `utils/state_formatter.py` — formats game state for LLM prompts (important for parsing by planning/perception).
  - `pokemon_env/` — emulator integration (mGBA). Do not modify `memory_reader.py` unless you understand emulator memory maps.
  - `server/` — FastAPI server and frame streaming used in multiprocess mode.

- Big-picture architecture and dataflow:
  - Perception -> Planning -> Memory -> Action modules coordinate via in-memory `memory` object passed between modules.
  - In multiprocess mode, `server.app` (FastAPI) runs the emulator and exposes frames/state; the client runs the agent logic and communicates via `server.client`.
  - `run.py --agent-auto` launches the server subprocess and a lightweight frame server for the web UI (`/stream.html`).

- Important runtime conventions the assistant must respect:
  - Actions are tokenized as uppercase single words: `A`, `B`, `UP`, `DOWN`, `LEFT`, `RIGHT`, `WAIT`, `START`.
  - Movement decisions must be validated against the `MOVEMENT PREVIEW:` output from `utils/state_formatter.py`. See `agent/planning.py` for an example of parsing `MOVEMENT PREVIEW:` and `--- MAP:` sections.
  - Memory module API (expected methods/fields):
    - `get_active_goal()` -> dict or None
    - `update_goal_status(observation)`
    - `enqueue_actions(list_of_actions)`
    - `pop_next_action()` -> action or None
    - `get_movement_memory(coords)` -> string (human-readable movement history)
    - `state` dict containing `long_term_plan` when present
  - Prompts are assembled in module files (e.g., `agent/planning.py` uses `format_state_for_llm` and then calls `vlm.get_text_query(prompt, module_name="planning")`). When changing prompt format, update both prompt text and any parsing / regex logic that extracts actions.

- VLM/backends and environment variables:
  - Supported backends: `openai`, `openrouter`, `gemini`, `local`. Backend adapters live in `utils/vlm.py` and are selected via `run.py --backend`.
  - Common environment variables used by the project:
    - `OPENAI_API_KEY`, `OPENROUTER_API_KEY`, `GEMINI_API_KEY`, `GOOGLE_API_KEY`
  - Local models require additional dependencies (Torch, Transformers, bitsandbytes). See README instructions.

- Developer workflows and commands (copy-paste):
  - Create environment and install deps (project uses `uv` + `.venv`):
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    uv sync
    source .venv/bin/activate
    ```
  - Start agent (multiprocess, auto-start server):
    ```bash
    python run.py --agent-auto --backend gemini --model-name gemini-2.5-flash
    ```
  - Quick dev: simple, no OCR, record
    ```bash
    python run.py --simple --agent-auto --no-ocr --record
    ```
  - Start server only (manual client):
    ```bash
    python -m server.app --port 8000
    ```

- Tests and validation:
  - Unit / integration tests live in `tests/`. Run the test runner via:
    ```bash
    python tests/run_tests.py
    ```
  - `tests/states/` and `tests/ground_truth/` store example save states and expected observations.

- Patterns and gotchas observed in the codebase (useful for editing / refactors):
  - Prompts and parsing are tightly coupled. If you change prompt wording in `agent/*`, also update regexes in the same module (see action extraction in `agent/planning.py`).
  - `format_state_for_llm` produces labeled sections like `--- MAP:` and `MOVEMENT PREVIEW:`; many modules parse these exact markers.
  - `pokemon_env/memory_reader.py` exposes low-level game state. Avoid changing it unless mission-critical—the rest of the code assumes its output shape.
  - The Planning module demonstrates defensive fallback behavior: it validates LLM choices against `walkable` directions and falls back to the first safe move or `A`. Preserve similar safety checks when adding new planners.
  - There is a minor typo in `agent/planning.py`: a `hasattr(memory, "pop_nexction")` check appears to be mistyped; prefer `pop_next_action` — be careful when modifying memory-related code.

- When adding features or models:
  - Add VLM adapters in `utils/vlm.py` and expose the config via `run.py` flags.
  - Update `agent/system_prompt.py` for global prompt changes and review `agent/*.py` for module-specific prompt usage.

If anything here is unclear or you'd like me to expand examples (e.g., sample prompt edits, a skeleton VLM adapter, or unit-test templates), say what section to expand and I'll iterate.
