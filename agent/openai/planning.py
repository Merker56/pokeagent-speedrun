import json
import re
import time
import hashlib
import logging
import random
from utils.helpers import add_text_update
from utils.llm_logger import log_llm_interaction, log_llm_error
from utils.state_formatter import _format_map_info
from .system_prompt import SYSTEM_PROMPT, PLANNING_PROMPT, PLANNING_FORMAT_HINT
try:
    from utils.vlm import VLMBackend
except Exception as e:
    logging.getLogger(__name__).warning(f"Planning: could not import VLMBackend ({e})")
    VLMBackend = None

logger = logging.getLogger(__name__)

# FIXED: More specific completion cues for start_game, no generic "boxes"
DEFAULT_EMERALD_LT_PLAN = [
    {
        "id": "start_game", 
        "goal": "Start a New Game and complete the initial Professor Birch introduction dialogue.", 
        "status": "pending", 
        "completion_cues": [
            "your very own adventure",
            "well, i'll be expecting you later",
            "come see me",
        ]
    },
    {
        "id": "leave_the_truck",
        "goal": "Exit the moving truck by pressing RIGHT to move to the door.",
        "status": "pending",
        "completion_cues": [
            "your room is upstairs", 
            "player's house", 
            "you've arrived at your new home",
            "mom",
            "downstairs",
        ],
    },
    {
        "id": "go_upstairs",
        "goal": "Go upstairs to the second floor of your house by walking straight ahead.",
        "status": "pending",
        "completion_cues": [
            "2f", 
            "second floor",
        ],
    },
    {
        "id": "player_bedroom_entered",
        "goal": "Find the clock in the bedroom and interact with it. It should be along an upper wall to the left of the stairs that you came up.",
        "status": "pending",
        "completion_cues": [
            "it's a nintendo gamecube",
            "this is your room",
            "this is your bedroom",
        ],
    },
    {
        "id": "set_bedroom_clock",
        "goal": "Set the bedroom clock by progressing through the dialogue.",
        "status": "pending",
        "completion_cues": [
            "better set it",
            "set the time",
            "set the clock",
            "it's a pokémon brand clock",
            "clock is stopped",
        ],
    },
    {
        "id": "return_downstairs",
        "goal": "Go downstairs to the first floor using the stairs to your right. It will be along the top wall.",
        "status": "pending",
        "completion_cues": [
            "downstairs",
            "1f",
        ],
    },
    {
        "id": "exit_player_house",
        "goal": "Leave the player's house and go outside.",
        "status": "pending",
        "completion_cues": [
            "littleroot",
            "outside", 
            "route 101",
        ],
    },
    {"id": "littleroot_town_explore", "goal": "Find Professor Birch's lab which is next door to your house.", "status": "pending", "completion_cues": ["birch's lab", "go visit prof. birch",]},
    {"id": "route_101", "goal": "Proceed to Route 101 to encounter Professor Birch.", "status": "pending", "completion_cues": ["route 101", "oldale town", "birch's bag", "map_id=3"]},
    {"id": "choose_starter", "goal": "Help Professor Birch and choose a Mudkip.", "status": "pending", "completion_cues": ["chose torchic", "chose treecko", "chose mudkip", "received pokemon", "party_count=1"]},
    {"id": "first_rival_battle", "goal": "Battle the rival on Route 103 and win.", "status": "pending", "completion_cues": ["defeated may", "route 103", "winner", "battle_result=win"]},
    {"id": "oldale_town_visit", "goal": "Visit Oldale Town, heal up, and get any useful items.", "status": "pending", "completion_cues": ["oldale town", "pokecenter", "healed", "map_id=2"]},
    {"id": "petalburg_city_reach", "goal": "Reach Petalburg City and visit the Petalburg Gym.", "status": "pending", "completion_cues": ["petalburg city", "norman's gym", "map_id=4"]},
    {"id": "petalb_woods_navigate", "goal": "Navigate through Petalburg Woods to Rustboro City.", "status": "pending", "completion_cues": ["petalburg woods", "devon corp", "map_id=5"]},
    {"id": "rustboro_city_reach", "goal": "Arrive in Rustboro City and interact with the Devon Researcher.", "status": "pending", "completion_cues": ["rustboro city", "devon goods", "map_id=6"]},
    {"id": "rustboro_gym_challenge", "goal": "Challenge and defeat Gym Leader Roxanne for the Stone Badge.", "status": "pending", "completion_cues": ["roxanne", "stone badge", "gym_id=1", "battle_result=win"]},
    {"id": "route_116_explore", "goal": "Explore Route 116 and help the old man find his Peeko.", "status": "pending", "completion_cues": ["route 116", "rusturf tunnel", "map_id=7"]},
    {"id": "rusturf_tunnel_clear", "goal": "Clear Rusturf Tunnel and recover the Devon Goods.", "status": "pending", "completion_cues": ["rusturf tunnel", "team aqua", "devon goods recovered", "map_id=8"]},
]


class PlanningModule:
   """Goal-aware planning module that cooperates with Memory and Perception."""

   def __init__(self):
      self.context_cache = {}
      self.initial_long_term_plan = DEFAULT_EMERALD_LT_PLAN
      self._PENDING_SHADOW: list[str] = []

   def _hash_context(self, observation, memory):
      """Compact hash of the high-level state to avoid redundant calls."""
      if isinstance(observation, dict):
         scene_type = observation.get("scene_type", "")
         summary = observation.get("summary", "")
      else:
         print(f"[PLANNING DEBUG] observation not dict in _hash_context, got {type(observation)}")
         scene_type = ""
         summary = str(observation)

      active_goal_id = "no_goal"
      try:
         if hasattr(memory, "get_active_goal"):
               ag = memory.get_active_goal()
               print(f"[PLANNING DEBUG] _hash_context got active goal: {ag!r} (type={type(ag)})")
               if isinstance(ag, dict):
                  active_goal_id = ag.get("id", "no_goal")
               elif ag:
                  active_goal_id = str(ag)
         else:
               print("[PLANNING DEBUG] memory has no get_active_goal in _hash_context")
      except Exception as e:
         print(f"[PLANNING DEBUG] ERROR reading active goal in _hash_context: {e!r}")
         active_goal_id = "goal_error"

      mem_keys = []
      try:
         if hasattr(memory, "state") and isinstance(memory.state, dict):
               mem_keys = sorted(memory.state.keys())
         else:
               print("[PLANNING DEBUG] memory.state missing or not dict in _hash_context")
      except Exception as e:
         print(f"[PLANNING DEBUG] ERROR reading memory.state in _hash_context: {e!r}")

      key = f"{scene_type}|{summary}|{active_goal_id}|{mem_keys}"
      print(f"[PLANNING DEBUG] _hash_context key before hash: {key}")
      return hashlib.sha1(key.encode()).hexdigest()

   def _shadow_enqueue(self, actions):
      if not actions:
         return
      if not isinstance(actions, list):
         actions = [actions]
      allowed = {"UP","DOWN","LEFT","RIGHT","A","B"}
      for a in actions:
         a = str(a).strip().upper()
         if a in allowed:
               self._PENDING_SHADOW.append(a)

   def _shadow_pop(self):
      if self._PENDING_SHADOW:
         return self._PENDING_SHADOW.pop(0)
      return 
   
   def _build_navigation_context(self, obs_state, key_elements, memory, goal_name, goal_id):
      """
      Build navigation/map hints for the planning prompt from state + memory.
      Returns (state_hints, ascii_map).
      """
      state_hints = []
      ascii_map = None
      walkable = None
      location_name = None
      px = None
      py = None

      # 1) Pull player + location + map info from obs_state
      if isinstance(obs_state, dict):
         player_info = obs_state.get("player") or {}
         if isinstance(player_info, dict):
               loc_from_player = player_info.get("location")
               pos = (
                  player_info.get("position")
                  or player_info.get("coordinates")
                  or player_info.get("tile_coordinates")
               )
               if isinstance(pos, dict):
                  px = pos.get("x")
                  py = pos.get("y")
               if isinstance(loc_from_player, str):
                  location_name = loc_from_player

         map_info = obs_state.get("map") or {}
         if isinstance(map_info, dict):
               # Debug: see what map keys we actually have
               try:
                  print(f"[PLAN] NAV map_info keys: {list(map_info.keys())}")
               except Exception:
                  pass

               # Prefer richer map forms but fall back gracefully.
               # NOTE: enhanced_text_map is produced by the enhanced map pipeline
               ascii_map = (
                  map_info.get("visual_map")
                  or map_info.get("ascii_map")
                  or map_info.get("ascii")
                  or map_info.get("enhanced_text_map")  # <-- NEW: use enhanced text map
               )

               # Walkability / traversability
               # Some paths use `walkable`, the enhanced one uses `traversability`
               walkable = map_info.get("walkable") or map_info.get("traversability")

      # 2) Fallback for player position from key_elements if needed
      try:
         if px is None:
               px = key_elements.get("player_x")
         if py is None:
               py = key_elements.get("player_y")
      except Exception:
         pass

      # 3) If we still don't have ascii_map but do have walkable, synthesize a local map
      def _ascii_from_walkable_local(w, px_val=None, py_val=None):
         if not isinstance(w, dict) or not w:
               return None

         # Compute bounds around player; slightly larger window than before
         xs = [x for (x, _) in w.keys()]
         ys = [y for (_, y) in w.keys()]
         min_x, max_x = min(xs), max(xs)
         min_y, max_y = min(ys), max(ys)

         # If we know player position, center around them; otherwise bounding box
         if px_val is not None and py_val is not None:
               radius = 4  # was effectively ~1 before; give more context
               min_x = px_val - radius
               max_x = px_val + radius
               min_y = py_val - radius
               max_y = py_val + radius

         lines = []
         for y in range(min_y, max_y + 1):
               row = []
               for x in range(min_x, max_x + 1):
                  if px_val is not None and py_val is not None and x == px_val and y == py_val:
                     row.append("P")
                  else:
                     v = w.get((x, y), None)
                     if v is True:
                           row.append("O")  # walkable
                     elif v is False:
                           row.append("X")  # blocked
                     else:
                           row.append("?")  # unknown
               lines.append("".join(row))
         return "\n".join(lines)

      if not ascii_map and walkable:
         ascii_map = _ascii_from_walkable_local(walkable, px, py)

      if not ascii_map:
         ascii_map = "Map not available"

      # 4) Build navigation hints text
      if location_name:
         state_hints.append(f"LOCATION: {location_name}")
      if px is not None and py is not None:
         state_hints.append(f"PLAYER_TILE: (x={px}, y={py})")

      if walkable and px is not None and py is not None:
         dirs = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}
         lines = []
         for d, (dx, dy) in dirs.items():
               nx, ny = px + dx, py + dy
               status = "unknown"
               try:
                  if isinstance(walkable, dict) and (nx, ny) in walkable:
                     status = "walkable" if walkable.get((nx, ny)) else "blocked"
               except Exception:
                  status = "unknown"
               lines.append(f"{d}: {status}")
         state_hints.append("LOCAL_MOVEMENT_OPTIONS:\n  " + "\n  ".join(lines))

      # Always include GOAL + MAP at the end
      state_hints.append(f"GOAL: {goal_name} (ID={goal_id})")
      state_hints.append(
         "MAP (ASCII):\n"
         "Legend: X=blocked, O=walkable, ~=visited, P=you, ?=unknown\n"
         f"{ascii_map}"
      )

      # Optional debug output so you can see what the LLM sees
      try:
         print("[PLAN] NAV CTX location=", location_name, " pos=", (px, py))
         print("[PLAN] NAV CTX map preview:\n" + str(ascii_map)[:400])
      except Exception:
         pass

      return state_hints, ascii_map

   def decide(self, vlm, observation, memory):
      # -------- 0) Read observation safely --------
      obs = observation or {}
      scene_type   = obs.get("scene_type", "UNKNOWN")
      summary      = obs.get("summary") or ""
      ctx_hash = self._hash_context(observation, memory)
      cached = self.context_cache.get(ctx_hash)
      if cached:
         return cached

      # FIX 4: Update goal status BEFORE planning decision
      if hasattr(memory, 'update_goal_status'):
          try:
              memory.update_goal_status(observation)
              print(f"[PLAN] Updated goal status before planning")
          except Exception as e:
              print(f"[PLAN] Goal status update failed: {e}")

      summary_lower = summary.lower()
      obs_state     = obs.get("state") or {}
      key_elements  = obs.get("key_elements") or {}

      if not isinstance(key_elements, dict):
         if isinstance(key_elements, str):
               try:
                  key_elements = eval(key_elements)
               except:
                  key_elements = {}
         else:
               key_elements = {}

      print(f"[PLAN] scene={scene_type}, summary={summary[:80]}")

      # -------- A) Check if memory has queued actions --------
      if hasattr(memory, "pop_next_action"):
         nxt = memory.pop_next_action()
         if nxt:
               return {"action": nxt, "reason": "queued"}

      if self._PENDING_SHADOW:
         a = self._shadow_pop()
         if a:
               return {"action": a, "reason": "shadow_queue"}

      # -------- B) Title screen detection (press A to start) --------
      if scene_type == "TITLE":
         memory.enqueue_actions(["A", "A", "A", "A", "A", "A", "A", "A", "A"])
         return {"action": "A", "reason": "title_screen"}

      # -------- C) BATTLE → always A (no LLM) --------
      if scene_type == "BATTLE":
         return {"action": "A", "reason": "autoA_battle"}

      # -------- D) MENU/NAMING fastpath (ENHANCED) --------
      try:
         # Initialize flag on memory if not present
         if not hasattr(memory, '_naming_handled'):
            memory._naming_handled = False
         
         # Check for naming screen
         if scene_type == "MENU" and not memory._naming_handled:
            # Broad detection - any indication of name entry
            summary_has_name = any(keyword in summary_lower for keyword in [
                  "name", "enter", "keyboard", "character", "nickname", "input"
            ])
            
            # Check key elements
            has_naming_keys = False
            if isinstance(key_elements, dict):
                  ke_str = str(key_elements).lower()
                  has_naming_keys = any(keyword in ke_str for keyword in [
                     "name", "keyboard", "input", "naming", "nickname", "character"
                  ])
            
            if summary_has_name or has_naming_keys:
                  # Set flag IMMEDIATELY
                  memory._naming_handled = True
                  
                  # Queue multiple As with START - NOTE: 3 As causes issue with memory reader
                  if hasattr(memory, "enqueue_actions"):
                     memory.enqueue_actions(["A", "A", "A", "START"])
                     print("[PLAN] NAMING DETECTED -> enqueued 3xA + Start")
                  
                  # Return START immediately
                  print("[PLAN] NAMING -> return START (flag now True)")
                  return {"action": "START", "reason": "naming_completion"}
      except Exception as e:
         print(f"[PLAN][EXC] naming block: {e}")

      # -------- E) Dialogue / Cutscene → always A (no LLM, summary-aware) --------
      is_dialog_like = False

      # Normalize scene_type
      try:
         st_upper = (scene_type or "UNKNOWN").upper()
      except Exception:
         st_upper = "UNKNOWN"

      # 1) Direct scene_type match
      if st_upper in ("DIALOGUE", "CUTSCENE"):
         is_dialog_like = True
      else:
         # 2) Summary-based detection (for MAP + dialogue box overlays)
         dialog_keywords = [
            "dialogue box",
            "dialog box",
            "speech bubble",
            "text box",
            "message box",
            "conversation",
            "talking",
            "is speaking",
            "is saying",
         ]
         if any(kw in summary_lower for kw in dialog_keywords):
            is_dialog_like = True

         # 3) Optional: key_elements-based detection if present
         try:
            if isinstance(key_elements, dict):
               ke_str = str(key_elements).lower()
               if any(kw in ke_str for kw in ["dialogue_box", "speech_bubble", "text_box"]):
                  is_dialog_like = True
         except Exception:
            pass

      if is_dialog_like:
         print(
            f"[PLAN] DIALOGUE/CUTSCENE fastpath -> A "
            f"(scene_type={scene_type}, summary≈{summary[:80]!r})"
         )
         return {"action": "A", "reason": "autoA_dialogue_or_cutscene"}


      # ===================== MAP / NAVIGATION =====================

      # -------- F) Recover GOAL (for prompt) --------
      active_goal = memory.get_active_goal() if hasattr(memory, "get_active_goal") else None
      if isinstance(active_goal, dict):
         goal_name = active_goal.get("goal") or active_goal.get("name") or "Progress the game"
         goal_id   = active_goal.get("id")   or "no_goal"
      else:
         goal_name = "Progress the game"
         goal_id   = "no_goal"

      # -------- G) Build navigation + map hints --------
      state_hints, ascii_map = self._build_navigation_context(
          obs_state=obs_state,
          key_elements=key_elements,
          memory=memory,
          goal_name=goal_name,
          goal_id=goal_id,
      )

      # -------- H) FIX 3: Enhanced truck/storage exit heuristic --------
      if "truck" in summary_lower or ("box" in summary_lower and "storage" in summary_lower):
         # In truck/storage area - prioritize moving RIGHT to exit
         if goal_id in ["leave_the_truck", "start_game"]:
               print("[PLAN] TRUCK detected, prioritizing RIGHT to exit")
               return {"action": "RIGHT", "reason": "truck_exit"}
         # Otherwise explore with bias toward RIGHT
         return {"action": random.choice(["RIGHT", "RIGHT", "DOWN", "LEFT"]), "reason": "truck_explore"}

      # -------- I) Compact LLM prompt with GOAL + MAP --------
      extra = "\n".join(state_hints)
      goal_text = ""
      active_goal = memory.get_active_goal()
      if active_goal:
         goal_text = f"CURRENT GOAL: {active_goal['goal']}"

      prompt = (
         f"{goal_text}\n"
         f"Observation:\n{json.dumps(observation, indent=2)}\n"
         f"State Hints:\n{extra}\n"
         f"{PLANNING_PROMPT}\n"
      )

      try:
         out = vlm.get_text_query(prompt, module_name="planning")
      except Exception as e:
         print(f"[PLAN][EXC] VLM call: {e}")
         out = "A"

      if not isinstance(out, str):
         out = str(out)
      txt = out.upper().strip()
      print(f"[PLAN] LLM -> {txt[:120]}")

      m = re.search(r"ACTION:\s*([A-Z0-9\s,]+)", txt)
      if m:
         txt = m.group(1).strip()

      # Handle comma-separated action sequences
      # Example: "MOVE UP 1, MOVE RIGHT 1, A, MOVE DOWN 2, A"
      if "," in txt:
         parts = [p.strip() for p in txt.split(",")]
         print(f"[PLAN] Multi-action detected: {len(parts)} parts")
         
         # Parse all parts, queue all but first
         if len(parts) > 1 and hasattr(memory, "enqueue_actions"):
               queue_actions = []
               for p in parts[1:]:  # Skip first, we'll return that
                  # Parse each part
                  move_m = re.search(r"MOVE\s+(UP|DOWN|LEFT|RIGHT)\s+(\d+)", p)
                  if move_m:
                     d = move_m.group(1)
                     n = max(1, min(3, int(move_m.group(2))))
                     queue_actions.extend([d] * n)
                  else:
                     # Single button
                     for t in re.findall(r"[A-Z]+", p):
                           if t in {"UP","DOWN","LEFT","RIGHT","A","B","START"}:
                              queue_actions.append(t)
                              break
               
               if queue_actions:
                  memory.enqueue_actions(queue_actions)
                  print(f"[PLAN] QUEUE multi-action: {queue_actions}")
         
         # Process first part as the return action
         txt = parts[0]
         print(f"[PLAN] Processing first action: {txt}")

      # Process the first/only action
      mm = re.search(r"MOVE\s+(UP|DOWN|LEFT|RIGHT)\s+(\d+)", txt)
      if mm:
         d = mm.group(1)
         n = max(1, min(3, int(mm.group(2))))
         if n > 1 and hasattr(memory, "enqueue_actions"):
               memory.enqueue_actions([d] * (n - 1))
               print(f"[PLAN] QUEUE tail after MOVE -> {[d]*(n-1)}")
         return {"action": d}

      allowed = {"UP","DOWN","LEFT","RIGHT","A","B","START"}
      for t in re.findall(r"[A-Z]+", txt):
         if t in allowed:
               # bias away from A on MAP if we can move
               if scene_type == "MAP" and t == "A":
                  return {"action": random.choice(["UP","DOWN","LEFT","RIGHT"])}
               return {"action": t}

      # final fallback: move (don't A-mash) on MAP; else A
      return {"action": random.choice(["UP","DOWN","LEFT","RIGHT"]) if scene_type == "MAP" else "A"}

# ------------------------------------------------------------------
# Legacy wrapper for compatibility
# ------------------------------------------------------------------
def planning_step(vlm, perception_output, memory):
   """Compatibility wrapper."""
   from agent.memory import MemoryModule

   if not hasattr(memory, 'state') or not hasattr(memory, 'get_active_goal'):
      if isinstance(memory, (list, dict)):
         temp_mem = MemoryModule()
         if isinstance(memory, list):
               temp_mem.state["entries"] = memory
         elif isinstance(memory, dict):
               temp_mem.state.update(memory)
               if "long_term_plan" not in temp_mem.state or not temp_mem.state["long_term_plan"]:
                  temp_mem.state["long_term_plan"] = DEFAULT_EMERALD_LT_PLAN
         memory = temp_mem
      else:
         logger.error(f"Received unexpected memory type: {type(memory)}. Creating a default MemoryModule.")
         memory = MemoryModule()

   module = PlanningModule()
   return module.decide(vlm, perception_output, memory)