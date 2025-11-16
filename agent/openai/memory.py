#!/usr/bin/env python3
"""
memory.py
----------
Lightweight episodic memory system for the Pokémon Emerald agent.

Responsibilities:
- Maintain compact state deltas (location, progress, blockers, team)
- Provide context summaries for planning
- Avoid unnecessary data growth during long runs
"""

import time
import json
import os
import logging
from .system_prompt import MEMORY_RULES, SYSTEM_PROMPT

logger = logging.getLogger(__name__)

_GLOBAL_PENDING: list[str] = []
_BOOT_BURST_DONE: bool = False

_ALLOWED_BTNS = {"UP", "DOWN", "LEFT", "RIGHT", "A", "B", "START"}

def _normalize_primitives(actions) -> list[str]:
    if not actions:
        return []
    if not isinstance(actions, list):
        actions = [actions]
    out = []
    for a in actions:
        s = str(a).strip().upper()
        if s in _ALLOWED_BTNS:
            out.append(s)
    return out

class MemoryModule:
    """Compact memory for milestone-focused Pokémon Emerald play."""

    def __init__(self, save_path: str = ".pokeagent_cache/memory_state.json", max_entries: int = 50):
        self.save_path = save_path
        self.max_entries = max_entries
        self.entries = []  # chronological list of (timestamp, data)
        self.state = {
            "location": None,
            "badges": 0,
            "team": {},
            "progress": [],
            "blockers": [],
            "last_action": None,
            "last_seen_scene": None,
            "long_term_plan": [],
            "short_term_plan": [],
            "current_goal": None,
            "exploration": {},
            "pending_actions": [],
            "last_ascii_map": None,
            "last_location": None,
            "last_move_dir": None,
            "pending_actions": [],
            "boot_burst_done": False,
        }
        self._naming_handled = False
        self.map_cache = {}
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self.load()

    def complete_goal(self, goal_id: str):
        """
        Mark a goal in long_term_plan as completed and clear current_goal if it matches.
        Safe to call even if the goal_id does not exist.
        """
        lt_plan = self.state.get("long_term_plan", [])
        for g in lt_plan:
            if g.get("id") == goal_id:
                g["status"] = "completed"
                break

        # if current_goal is this one, clear it
        current = self.state.get("current_goal")
        if current and current.get("id") == goal_id:
            self.state["current_goal"] = None

    def _normalize_action_string(self, s: str) -> str:
        """Normalize inputs like 'A', 'PRESS A', 'MOVE UP 1' into parser-friendly directives."""
        s = (s or "").strip().upper()
        if not s:
            return ""
        # Already a MOVE directive
        if s.startswith("MOVE "):
            return s
        # Already PRESS/INTERACT/MENU
        if s.startswith("PRESS ") or s.startswith("INTERACT") or s.startswith("MENU"):
            return s
        # Single button token -> convert to PRESS <BTN>
        btns = {"A","B","START","SELECT","UP","DOWN","LEFT","RIGHT"}
        if s in btns:
            return f"PRESS {s}"
        # Fallback, keep as-is
        return s
    
    def enqueue_actions(self, actions):
        """Global primitive queue: persists across steps even if MemoryModule is recreated."""
        global _GLOBAL_PENDING
        items = _normalize_primitives(actions)
        if not items:
            print(f"[MEMQ] enqueue ignored (no allowed): {actions}")
            return
        before = len(_GLOBAL_PENDING)
        _GLOBAL_PENDING.extend(items)
        print(f"[MEMQ] enqueue GLOBAL added={items} size {before}->{len(_GLOBAL_PENDING)} queue={_GLOBAL_PENDING}")

    def pop_next_action(self):
        """Pop from the global queue first; falls back to instance queue if you keep one."""
        global _GLOBAL_PENDING
        if _GLOBAL_PENDING:
            act = _GLOBAL_PENDING.pop(0)
            # optional mirror to local last_action/dir, useful for logs
            self.state["last_action"] = act
            if act in {"UP","DOWN","LEFT","RIGHT"}:
                self.state["last_move_dir"] = act
            print(f"[MEMQ] pop GLOBAL -> {act} (remaining={len(_GLOBAL_PENDING)}) queue={_GLOBAL_PENDING}")
            return act
        # (optional) legacy per-instance queue support if you still have it:
        q = self.state.get("pending_actions") or []
        if q:
            act = q.pop(0)
            self.state["pending_actions"] = q
            self.state["last_action"] = act
            if act in {"UP","DOWN","LEFT","RIGHT"}:
                self.state["last_move_dir"] = act
            print(f"[MEMQ] pop LOCAL -> {act} (remaining={len(q)}) queue={q}")
            return act
        return None

    def has_done_boot_burst(self) -> bool:
        global _BOOT_BURST_DONE
        return _BOOT_BURST_DONE

    def set_boot_burst_done(self):
        global _BOOT_BURST_DONE
        _BOOT_BURST_DONE = True
        # keep an instance mirror too (handy for debugging)
        self.state["boot_burst_done"] = True
        print("[MEMQ] set_boot_burst_done -> True (GLOBAL)")

    def _normalize_location_name(self, name: str) -> str:
        if not name:
            return "unknown"
        name = name.strip().lower()
        # optional: replace spaces with _
        name = name.replace(" ", "_")
        return name

    def store_map_snapshot(self, location_name: str, map_payload: dict):
        """
        Save a map/stitcher snapshot under a normalized location name.
        map_payload can contain: visual_map, walkable, player_x, player_y, raw_dims, etc.
        """
        if not map_payload:
            return

        key = self._normalize_location_name(location_name)
        if key == "unknown":
            # we can still store under 'unknown', but it's less useful
            key = "unknown_area"

        # we can be smart and merge, but for now just overwrite
        self.map_cache[key] = map_payload

    def get_best_map_for_observation(self, observation: dict) -> dict | None:
        """
        Try, in order:
        1) map for this observation's location (if present)
        2) map for last seen location in state
        3) fall back to latest entry in cache
        """
        # 1) from observation
        ke = observation.get("key_elements") or {}
        obs_loc = ke.get("Location") or ke.get("location") or ""
        key = self._normalize_location_name(obs_loc)
        if key in self.map_cache:
            return self.map_cache[key]

        # 2) from memory.state
        last_loc = self.state.get("location") or self.state.get("last_seen_location") or ""
        key2 = self._normalize_location_name(last_loc)
        if key2 in self.map_cache:
            return self.map_cache[key2]

        # 3) fallback: latest map we saw at all
        if self.map_cache:
            # return the most recently inserted
            # python 3.7+ dicts keep insertion order
            last_key = list(self.map_cache.keys())[-1]
            return self.map_cache[last_key]

        return None

    def record_position(self, location_name: str, x: int, y: int):
        """Mark (x,y) in this location as visited."""
        loc_key = self._normalize_location_name(location_name)
        if "exploration" not in self.state or not isinstance(self.state["exploration"], dict):
            self.state["exploration"] = {}

        loc_entry = self.state["exploration"].setdefault(loc_key, {"visited": []})
        coord_str = f"{x},{y}"
        if coord_str not in loc_entry["visited"]:
            loc_entry["visited"].append(coord_str)
            # optional: also record event if you want to see it in the jsonl
            self._record_event("visited_tile", {"location": loc_key, "coord": coord_str})

    def get_visited_positions(self, location_name: str) -> set[str]:
        """Return visited coords for this location as a set of 'x,y' strings."""
        loc_key = self._normalize_location_name(location_name)
        exp = self.state.get("exploration", {})
        loc_entry = exp.get(loc_key) or {}
        return set(loc_entry.get("visited", []))
    
    # --- map assist for planner -------------------------------------------------
    def set_last_ascii_map(self, ascii_map: str, location: str | None = None):
        """Remember the last ASCII map we gave to the LLM."""
        self.state["last_ascii_map"] = ascii_map
        if location:
            self.state["last_location"] = location

    def set_last_planned_move_dir(self, move_dir: str):
        """Remember the last MOVE direction the planner asked for (UP/DOWN/LEFT/RIGHT)."""
        self.state["last_move_dir"] = move_dir.upper()

    def get_last_ascii_map(self) -> str | None:
        return self.state.get("last_ascii_map")

    def predict_ascii_after_move(self) -> str | None:
        """
        If we didn't get a fresh stitch this step, try to 'move' the player
        in the last known ASCII map according to the last planned move.
        Works with maps that use P or A for player and X/# for walls.
        """
        ascii_map = self.state.get("last_ascii_map")
        move_dir = self.state.get("last_move_dir")
        if not ascii_map or not move_dir:
            return ascii_map  # nothing to do

        # parse map into grid
        lines = ascii_map.strip("\n").split("\n")
        grid = [list(line) for line in lines]

        # find player char
        player_char_candidates = ("P", "A")
        px = py = None
        for y, row in enumerate(grid):
            for x, ch in enumerate(row):
                if ch in player_char_candidates:
                    px, py = x, y
                    break
            if px is not None:
                break

        if px is None:
            # no player in map, just return original
            return ascii_map

        # movement delta
        dx = dy = 0
        if move_dir == "UP":
            dy = -1
        elif move_dir == "DOWN":
            dy = 1
        elif move_dir == "LEFT":
            dx = -1
        elif move_dir == "RIGHT":
            dx = 1

        nx = px + dx
        ny = py + dy

        # clamp to bounds
        if 0 <= ny < len(grid) and 0 <= nx < len(grid[ny]):
            # move player even if target is X/# — per your example
            old_char = grid[py][px]
            # pick what to leave behind
            grid[py][px] = "O" if old_char in ("P", "A") else old_char
            grid[ny][nx] = "P"
        # else: out of bounds → just keep original

        # back to string
        new_lines = ["".join(row) for row in grid]
        return "\n".join(new_lines)


    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, observation: dict, last_action: str = None):
        """Update memory with the latest scene and optional action."""
        try:
            scene = observation.get("scene_type")
            summary = observation.get("summary", "")
            key_elements = observation.get("key_elements", {})

            # Update high-level location
            if "Location" in key_elements:
                new_location = key_elements["Location"]
                if new_location != self.state.get("location"):
                    logger.info(f"📍 Location changed: {self.state.get('location')} → {new_location}")
                    self.state["location"] = new_location
                    self._record_event("location", new_location)

            # Detect badge or progress keywords
            if any(word in summary.lower() for word in ["badge", "gym leader", "victory", "defeated"]):
                self.state["badges"] += 1
                self._record_event("badge", self.state["badges"])

            # Detect blockers
            if "Blocker" in key_elements:
                blocker = key_elements["Blocker"]
                if blocker not in self.state["blockers"]:
                    self.state["blockers"].append(blocker)
                    self._record_event("blocker", blocker)

            # Track team summary if available
            if "PlayerHP" in key_elements or "Lead" in key_elements:
                self.state["team"]["lead_status"] = key_elements.get("PlayerHP", "?")

            # Update last known scene and action
            self.state["last_seen_scene"] = scene
            if last_action:
                self.state["last_action"] = last_action

            # Trim memory
            if len(self.entries) > self.max_entries:
                self.entries = self.entries[-self.max_entries :]

            # Auto-save every 5 updates
            if len(self.entries) % 5 == 0:
                self.save()

        except Exception as e:
            logger.warning(f"Memory update error: {e}")

    def summarize(self) -> dict:
        """Return a compact summary used by the planner."""
        return {
            "location": self.state.get("location"),
            "badges": self.state.get("badges"),
            "team": self.state.get("team"),
            "blockers": self.state.get("blockers"),
            "progress": len(self.state.get("progress", [])),
            "last_action": self.state.get("last_action"),
        }

    def last_context_signature(self) -> str:
        """Compact signature string for context hashing."""
        loc = self.state.get("location", "Unknown")
        badges = self.state.get("badges", 0)
        blockers = ",".join(sorted(self.state.get("blockers", [])))
        return f"{loc}|{badges}|{blockers}"

    def clear(self):
        """Reset memory (rarely used)."""
        self.entries.clear()
        self.state = {
            "location": None,
            "badges": 0,
            "team": {},
            "progress": [],
            "blockers": [],
            "last_action": None,
            "last_seen_scene": None,
        }
        self.save()
        logger.info("🧹 Memory cleared.")

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _record_event(self, key: str, value):
        """Record a timestamped memory delta."""
        event = {"time": time.time(), "type": key, "value": value}
        self.entries.append(event)
        logger.debug(f"Memory event: {event}")

    def set_long_term_plan(self, goals: list[str]):
        """Store or update global goals (badge progression, captures, etc.)."""
        self.state["long_term_plan"] = goals
        self._record_event("long_term_plan", goals)

    def set_short_term_plan(self, steps: list[str]):
        """Store the active tactical plan (e.g., reach PokéCenter)."""
        self.state["short_term_plan"] = steps
        if steps:
            self.state["current_goal"] = steps[0]
        self._record_event("short_term_plan", steps)

    def advance_short_term_goal(self):
        """Mark current goal as done and pop next."""
        if self.state["short_term_plan"]:
            done = self.state["short_term_plan"].pop(0)
            self._record_event("goal_completed", done)
            self.state["current_goal"] = (
                self.state["short_term_plan"][0]
                if self.state["short_term_plan"] else None
            )

    def save(self):
        """Save memory state to disk (non-blocking-safe)."""
        try:
            data = {"entries": self.entries, "state": self.state}
            with open(self.save_path, "w") as f:
                json.dump(data, f, indent=2)
            logger.debug(f"Memory saved to {self.save_path}")
        except Exception as e:
            logger.warning(f"Memory save failed: {e}")

    def ensure_long_term_plan(self):
        """Ensure there's a placeholder for LLM-generated plan."""
        if "long_term_plan" not in self.state:
            self.state["long_term_plan"] = []
            self._record_event("lt_plan_init", [])
            logger.info("🧩 Initialized empty long-term plan placeholder.")

    def load(self):
        """Load previous memory from disk if present."""
        try:
            if os.path.exists(self.save_path):
                with open(self.save_path, "r") as f:
                    data = json.load(f)
                self.entries = data.get("entries", [])
                self.state.update(data.get("state", {}))
                logger.info(f"Memory loaded from {self.save_path} ({len(self.entries)} entries)")
        except Exception as e:
            logger.warning(f"Memory load failed: {e}")

def initialize_long_term_plan(self):
    """Seed default long-term plan if missing."""
    if not self.state.get("long_term_plan"):
        self.state["long_term_plan"] = DEFAULT_LT_PLAN
        self._record_event("long_term_plan_init", DEFAULT_LT_PLAN)
        logger.info("🎯 Initialized long-term goals.")

def get_active_goal(self):
    """Return the next incomplete goal."""
    for g in self.state.get("long_term_plan", []):
        if g.get("status") != "completed":
            return g
    return None

def mark_in_lt_plan(self, goal_id: str):
    """Helper to keep long_term_plan in sync."""
    lt = self.state.get("long_term_plan", [])
    for g in lt:
        if g.get("id") == goal_id:
            g["status"] = "completed"
            break

def update_goal_status(self, observation):
    """
    Update the currently active goal using GAME STATE first (location, party, map),
    then fall back to dialogue / OCR text.
    """
    # 0) normalize
    if not isinstance(observation, dict):
        observation = {"summary": str(observation), "key_elements": ""}

    # text we can still use as a fallback
    text = (
        (observation.get("summary", "") + " " + str(observation.get("key_elements", "")))
        .lower()
    )

    # game state from memory reader
    loc = (self.state.get("location") or "").lower()
    # game state from perception
    raw_ke = observation.get("key_elements")
    if isinstance(raw_ke, dict):
        mem_ke = raw_ke
    else:
        mem_ke = {}
    obs_loc = (mem_ke.get("Location") or "").lower()

    # party info (0 vs >=1)
    party = self.state.get("party")
    if isinstance(party, list):
        party_count = len(party)
    else:
        party_count = int(self.state.get("party_count") or 0)

    # map id if your reader puts it in
    map_id = self.state.get("map_id") or mem_ke.get("map_id")

    # get current goal
    current = self.get_active_goal()
    if not current:
        return

    goal_id = current.get("id")
    scene_type = observation.get("scene_type", "")

    # ---------- STATE-BASED RULES ----------

    # FIX: start_game completion - only complete when we see END of intro
    if goal_id == "start_game":
        scene_type = observation.get("scene_type", "")
        
        # Check if we're on MAP (past cutscenes)
        if scene_type == "MAP":
            # Look for specific end-of-intro phrases
            end_phrases = [
                "your very own adventure",
                "well, i'll be expecting you later",
                "come see me"
            ]
            
            # Only complete if we saw the end dialogue
            if any(phrase in text for phrase in end_phrases):
                print(f"[MEMORY] start_game completed: saw end-intro text")
                current["status"] = "completed"
                self.mark_in_lt_plan(goal_id)
                self.state["current_goal"] = None
                return
            
            # Otherwise, if we're in truck/boxes scene WITHOUT end text,
            # it means we skipped ahead somehow - DON'T complete yet
            if "box" in text or "truck" in text or "storage" in text:
                print(f"[MEMORY] start_game: In truck moving to next goal active")
                current["status"] = "completed"
                self.mark_in_lt_plan(goal_id)
                self.state["current_goal"] = None
                return
        
        # Still in cutscene/dialogue - keep goal active
        return

    # FIX: leave_the_truck completion
    if goal_id == "leave_the_truck":
        # Completed when we see house interior cues
        house_cues = [
            "your room is upstairs",
            "player's house",
            "mom",
            "welcome home",
            "arrived at your new home",
            "downstairs"
        ]
        
        if any(cue in text for cue in house_cues):
            print(f"[MEMORY] leave_the_truck completed: house interior detected")
            current["status"] = "completed"
            self.mark_in_lt_plan(goal_id)
            self.state["current_goal"] = None
            return

    # 1) boot_intro (legacy, may not be used with new start_game)
    if goal_id == "boot_intro":
        if "house" in loc or "room is upstairs" in text or "your room is upstairs" in text:
            current["status"] = "completed"
            self.mark_in_lt_plan(goal_id)
            self.state["current_goal"] = None
            return

    # 2) house_downstairs_free
    if goal_id == "house_downstairs_free":
        if "house" in loc or "1f" in loc or "downstairs" in text:
            current["status"] = "completed"
            self.mark_in_lt_plan(goal_id)
            self.state["current_goal"] = None
            return

    # 3) go_upstairs
    if goal_id == "go_upstairs":
        # Only complete when LOCATION indicates we are actually upstairs,
        # not just when dialogue mentions "upstairs".
        upstairs_loc_cues = [
            "2f",
            "player_house_2f",
            "player house 2f",
            "player's house 2f",
        ]

        # Use both emulator location and observation-inferred location
        is_upstairs_loc = any(cue in loc for cue in upstairs_loc_cues) or \
                          any(cue in obs_loc for cue in upstairs_loc_cues)

        if is_upstairs_loc:
            print("[MEMORY] go_upstairs completed: location indicates 2F")
            current["status"] = "completed"
            self.mark_in_lt_plan(goal_id)
            self.state["current_goal"] = None
            return

        # IMPORTANT: do not fall through to generic cue-based completion,
        # otherwise "upstairs" in dialogue will complete this goal too early.
        return

    # 4) player_bedroom_entered
    if goal_id == "player_bedroom_entered":
        if ("this is your room" in text
            or "this is your bedroom" in text
            or "nintendo gamecube" in text):
            current["status"] = "completed"
            self.mark_in_lt_plan(goal_id)
            self.state["current_goal"] = None
            return

    # 5) set_bedroom_clock
    if goal_id == "set_bedroom_clock":
        if ("set the time" in text or "set the clock" in text or "better set it" in text or "clock is set" in text):
            current["status"] = "completed"
            self.mark_in_lt_plan(goal_id)
            self.state["current_goal"] = None
            return

    # 6) return_downstairs
    if goal_id == "return_downstairs":
        if "1f" in loc or "downstairs" in text or ("house" in loc and "2f" not in loc):
            current["status"] = "completed"
            self.mark_in_lt_plan(goal_id)
            self.state["current_goal"] = None
            return

    # 7) exit_player_house
    if goal_id == "exit_player_house":
        if "littleroot" in loc or "outside" in text or "route 101" in text:
            current["status"] = "completed"
            self.mark_in_lt_plan(goal_id)
            self.state["current_goal"] = None
            return

    # 10) choose_starter
    if goal_id == "choose_starter":
        if party_count >= 1:
            current["status"] = "completed"
            self.mark_in_lt_plan(goal_id)
            self.state["current_goal"] = None
            return

    # 8) littleroot_town_explore
    if goal_id == "littleroot_town_explore":
        if ("littleroot_town", "outside_littleroot", "littleroot") in loc or "littleroot" in obs_loc:
            current["status"] = "completed"
            self.mark_in_lt_plan(goal_id)
            self.state["current_goal"] = None
            return

    # 9) route_101
    if goal_id == "route_101":
        if "route 101" in loc or "route 101" in obs_loc or str(map_id) == "3":
            current["status"] = "completed"
            self.mark_in_lt_plan(goal_id)
            self.state["current_goal"] = None
            return

    # ---------- FALLBACK: dialogue-based rules ----------
    cues = current.get("completion_cues") or []
    for cue in cues:
        if cue and cue.lower() in text:
            current["status"] = "completed"
            self.mark_in_lt_plan(goal_id)
            self.state["current_goal"] = None
            break

# Bind helper methods to the class
MemoryModule.initialize_long_term_plan = initialize_long_term_plan
MemoryModule.get_active_goal = get_active_goal
MemoryModule.update_goal_status = update_goal_status
MemoryModule.mark_in_lt_plan = mark_in_lt_plan


def memory_step(perception_output, planning_output, memory):
    """
    Legacy wrapper so __init__.py can call memory_step().
    """
    from .memory import MemoryModule
    module = MemoryModule()
    module.update(perception_output, last_action=planning_output.get("action"))
    return module.summarize()