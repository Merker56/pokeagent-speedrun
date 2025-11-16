#!/usr/bin/env python3
"""memory.py - Memory and goal tracking for Pokemon Emerald"""

import json
import os
import logging

logger = logging.getLogger(__name__)

_GLOBAL_PENDING = []
_ALLOWED_BTNS = {"UP", "DOWN", "LEFT", "RIGHT", "A", "B", "START"}

def _normalize_primitives(actions):
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
    def __init__(self, save_path: str = ".pokeagent_cache/memory_state.json"):
        from agent.planning import DEFAULT_EMERALD_LT_PLAN
        self.save_path = save_path
        self.state = {
            "location": None,
            "long_term_plan": [],
            "current_goal": None,
            "last_action": None,
            "naming_handled": False,  # Track if we've already processed naming screen
            "explored_map": {},  # {"map_name": {"tiles": {"x_y": "symbol"}, "width": w, "height": h}}
        }
        # Track where we entered each map (runtime only, not persisted)
        self._entry_coords_by_map = {}  # {map_name: (x, y)}
        # Runtime-only flags (not persisted to disk)
        self._runtime_flags = {
            "clock_sequence_queued": False,
            "set_clock_sequence_queued": False,
            "downstairs_sequence_queued": False,
        }
        self.failed_movements = {}  # {coord_key: [directions]}
        self.npc_interactions = {}  # {coord_key: notes}
        self._last_coords = None  # Track for movement detection
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self.load()
        # Ensure long_term_plan is always initialized
        if not self.state.get("long_term_plan"):
            self.state["long_term_plan"] = DEFAULT_EMERALD_LT_PLAN.copy()
            self.state["current_goal"] = None
    
    def enqueue_actions(self, actions):
        global _GLOBAL_PENDING
        items = _normalize_primitives(actions)
        if not items:
            return
        before = len(_GLOBAL_PENDING)
        _GLOBAL_PENDING.extend(items)
        print(f"[MEM] Enqueued {len(items)}: {items}. Queue: {before}->{len(_GLOBAL_PENDING)}")
    
    def pop_next_action(self):
        global _GLOBAL_PENDING
        if _GLOBAL_PENDING:
            act = _GLOBAL_PENDING.pop(0)
            self.state["last_action"] = act
            print(f"[MEM] Popped: {act}. Remaining: {len(_GLOBAL_PENDING)}")
            return act
        return None
    
    def peek_next_action(self):
        """Peek at the next action without removing it from queue."""
        global _GLOBAL_PENDING
        if _GLOBAL_PENDING:
            return _GLOBAL_PENDING[0]
        return None
    
    def skip_next_action(self):
        """Remove and discard the next action from queue (used when validating rejects an action)."""
        global _GLOBAL_PENDING
        if _GLOBAL_PENDING:
            skipped = _GLOBAL_PENDING.pop(0)
            print(f"[MEM] Skipped rejected action: {skipped}. Remaining: {len(_GLOBAL_PENDING)}")
            return skipped
        return None
    
    def update_explored_map(self, observation):
        """Update the explored map from observation state."""
        if not isinstance(observation, dict):
            return
        
        state = observation.get("state", {})
        if not isinstance(state, dict):
            return
        
        map_data = state.get("map", {})
        if not map_data:
            return
        
        map_name = map_data.get("name") or map_data.get("current_map")
        if not map_name:
            return
        
        # Initialize map entry if new
        if "explored_map" not in self.state:
            self.state["explored_map"] = {}
        
        if map_name not in self.state["explored_map"]:
            self.state["explored_map"][map_name] = {
                "tiles": {},
                "width": map_data.get("width", 0),
                "height": map_data.get("height", 0)
            }
        
        # Update tiles from current view
        tiles = map_data.get("tiles", [])
        width = map_data.get("width", 0)
        height = map_data.get("height", 0)
        
        if tiles and width and height:
            stored = self.state["explored_map"][map_name]
            stored["width"] = width
            stored["height"] = height
            
            # Store tiles as x_y: symbol dict
            for y in range(height):
                for x in range(width):
                    idx = y * width + x
                    if idx < len(tiles):
                        tile = tiles[idx]
                        key = f"{x}_{y}"
                        # Only update if we have actual tile data
                        if tile:
                            stored["tiles"][key] = str(tile)
    
    def get_explored_map(self, map_name):
        """Get the explored map for a location."""
        if "explored_map" not in self.state:
            return None
        return self.state["explored_map"].get(map_name)
    
    def find_tiles_of_type(self, map_name, tile_types):
        """Find all coordinates of specific tile types in explored map.
        
        Args:
            map_name: Name of the map to search
            tile_types: List of tile symbols to find (e.g., ['D', 'S'])
        
        Returns:
            List of (x, y) tuples
        """
        explored = self.get_explored_map(map_name)
        if not explored:
            return []
        
        tiles = explored.get("tiles", {})
        results = []
        
        for coord_key, symbol in tiles.items():
            if symbol in tile_types:
                x, y = coord_key.split("_")
                results.append((int(x), int(y)))
        
        return results
    
    def record_failed_movement(self, coords, direction, reason):
        """Record a movement that didn't work"""
        if not coords:
            return
        
        key = f"{coords[0]}_{coords[1]}"
        if key not in self.failed_movements:
            self.failed_movements[key] = []
        
        if direction not in self.failed_movements[key]:
            self.failed_movements[key].append(direction)
            logger.info(f"⚠️  Recorded failed {direction} at {coords}: {reason}")
    
    def get_movement_memory(self, coords):
        """Get movement memory for location"""
        if not coords:
            return ""
        
        key = f"{coords[0]}_{coords[1]}"
        failed = self.failed_movements.get(key, [])
        
        if failed:
            return f"⚠️  MOVEMENT MEMORY: At ({coords[0]},{coords[1]}) previously failed: {', '.join(failed)}"
        return ""
    
    def initialize_long_term_plan(self, plan_list):
        self.state["long_term_plan"] = plan_list
        # Activate first pending goal
        for goal in plan_list:
            if goal.get("status") == "pending":
                self.state["current_goal"] = goal
                print(f"[MEM] Activated goal: {goal['id']}")
                break
    
    def get_active_goal(self):
        current = self.state.get("current_goal")
        if current and isinstance(current, dict):
            return current
        
        # Auto-activate next pending goal SEQUENTIALLY
        plan = self.state.get("long_term_plan", [])
        for goal in plan:
            if goal.get("status") == "pending":
                self.state["current_goal"] = goal
                print(f"[MEM] Auto-activated: {goal['id']}")
                return goal
        
        return None
    
    def complete_current_goal(self, goal_id):
        """Complete the current goal and activate next sequential goal."""
        # CRITICAL: Only allow completing the currently active goal
        current = self.get_active_goal()
        if not current or current.get("id") != goal_id:
            print(f"[MEM] ⚠️ Attempted to complete non-active goal {goal_id}, current is {current.get('id') if current else 'None'}")
            return
        
        plan = self.state.get("long_term_plan", [])
        
        # Find and mark current goal completed
        for i, g in enumerate(plan):
            if g.get("id") == goal_id:
                g["status"] = "completed"
                print(f"[MEM] ✅ Completed: {goal_id}")
                
                # Activate next pending goal in sequence
                for j in range(i + 1, len(plan)):
                    next_goal = plan[j]
                    if next_goal.get("status") == "pending":
                        self.state["current_goal"] = next_goal
                        print(f"[MEM] ➡️ Next goal: {next_goal['id']}")
                        return
                
                # No more goals
                self.state["current_goal"] = None
                print(f"[MEM] All goals complete!")
                return
    
    def update_goal_status(self, observation):
        """Check if current goal is completed."""
        # Convert non-dict observations to dict
        if not isinstance(observation, dict):
            if isinstance(observation, str):
                observation = {"summary": observation, "key_elements": {}, "state": {}}
            else:
                return
        
        current = self.get_active_goal()
        if not current or not isinstance(current, dict):
            return
        
        goal_id = current.get("id")
        if not goal_id:
            return
        
        # Get data safely
        summary = observation.get("summary", "")
        key_elements = observation.get("key_elements", {})
        state = observation.get("state", {})
        
        # Convert to text for matching
        text = summary.lower()
        if isinstance(key_elements, dict):
            text += " " + " ".join(str(v).lower() for v in key_elements.values())
        elif key_elements:
            text += " " + str(key_elements).lower()
        
        scene_type = observation.get("scene_type", "")
        
        # Get state info safely
        player = state.get("player", {}) if isinstance(state, dict) else {}
        loc = ""
        party_count = 0
        
        if isinstance(player, dict):
            loc = str(player.get("location", "")).lower()
            party = player.get("party", [])
            party_count = len(party) if isinstance(party, list) else 0
        
        # Goal completion checks
        if goal_id == "start_game":
            # Only complete when we're actually IN the truck on the map, not during dialogue
            if scene_type == "MAP" and ("truck" in text or "moving box" in text):
                self.complete_current_goal(goal_id)
                return
        
        if goal_id == "leave_the_truck":
            if any(c in text for c in ["your room is upstairs", "player's house", "mom", "arrived"]):
                self.complete_current_goal(goal_id)
                return
        
        if goal_id == "go_upstairs":
            # Only complete when location actually shows 2F
            if "2f" in loc:
                self.complete_current_goal(goal_id)
                return
        
        if goal_id == "player_bedroom_entered":
            # Only complete when on 2F AND seeing bedroom-related text
            if "2f" in loc and any(c in text for c in ["this is your room", "your bedroom", "nintendo gamecube", "clock"]):
                self.complete_current_goal(goal_id)
                return
        
        if goal_id == "set_bedroom_clock":
            # Complete after dialogue following clock setting
            if scene_type == "DIALOGUE" and hasattr(self, "_runtime_flags") and self._runtime_flags.get("set_clock_sequence_queued", False):
                logger.info("[MEM] Clock set (in dialogue), completing goal")
                self.complete_current_goal(goal_id)
                return
            # Also handle if back on map
            if scene_type == "MAP" and hasattr(self, "_runtime_flags") and self._runtime_flags.get("set_clock_sequence_queued", False):
                logger.info("[MEM] Clock set (on map), completing goal")
                self.complete_current_goal(goal_id)
                return
        
        if goal_id == "return_downstairs":
            # Only complete when actually on 1F AND clock was already set
            if "1f" in loc and hasattr(self, "_runtime_flags") and self._runtime_flags.get("set_clock_sequence_queued", False):
                self.complete_current_goal(goal_id)
                return
        
        if goal_id == "exit_player_house":
            # Only complete when location shows Littleroot Town (not just house 1F)
            if "littleroot town" in loc and "house" not in loc:
                self.complete_current_goal(goal_id)
                return
        
        if goal_id == "go_to_birch_lab":
            # Only complete when location actually shows the lab
            if "lab" in loc and "birch" in loc:
                self.complete_current_goal(goal_id)
                return
        
        if goal_id == "find_route_101":
            if "route 101" in loc or "route101" in loc:
                self.complete_current_goal(goal_id)
                return
        
        if goal_id == "find_professor_birch":
            if "professor" in text and ("help" in text or "pokemon" in text or "birch" in text):
                self.complete_current_goal(goal_id)
                return
        
        if goal_id == "choose_starter":
            if party_count >= 1:
                self.complete_current_goal(goal_id)
                return
    
    def update(self, observation, last_action=None):
        if last_action:
            self.state["last_action"] = last_action
        
        if isinstance(observation, dict):
            state = observation.get("state", {})
            if isinstance(state, dict):
                player = state.get("player", {})
                if isinstance(player, dict):
                    loc = player.get("location")
                    if loc:
                        # Check if map changed - if so, clear the queue and record entry coords
                        prev_loc = self.state.get("location")
                        if prev_loc and prev_loc != loc:
                            global _GLOBAL_PENDING
                            if _GLOBAL_PENDING:
                                print(f"[MEM] 🗺️ Map changed: {prev_loc} → {loc}. Clearing {len(_GLOBAL_PENDING)} queued actions")
                                _GLOBAL_PENDING.clear()
                            # Record entry position for the new map
                            coords = self._get_coords(observation)
                            if coords:
                                self._entry_coords_by_map[str(loc)] = coords
                                print(f"[MEM] ↪ Recorded entry position for {loc}: {coords}")
                        self.state["location"] = loc

        # Track coordinate changes to detect failed movements
        if last_action and last_action in ['UP', 'DOWN', 'LEFT', 'RIGHT']:
            current_coords = self._get_coords(observation)
            
            if self._last_coords and current_coords:
                if current_coords == self._last_coords:
                    # Movement failed! Coordinates didn't change
                    self.record_failed_movement(current_coords, last_action, "blocked")
            
            self._last_coords = current_coords

    def _get_coords(self, observation):
        """Extract coordinates from observation"""
        try:
            state = observation.get("state", {})
            player = state.get("player", {})
            pos = player.get("position", {})
            x, y = pos.get('x'), pos.get('y')
            if x is not None and y is not None:
                return (x, y)
        except:
            pass
        return None

    def get_entry_coords(self, map_name: str):
        """Return the coordinates where we entered the given map, if known."""
        try:
            return self._entry_coords_by_map.get(str(map_name))
        except Exception:
            return None
     
    def save(self):
        try:
            with open(self.save_path, 'w') as f:
                json.dump(self.state, f, indent=2)
        except Exception as e:
            logger.warning(f"Save failed: {e}")
    
    def load(self):
        if os.path.exists(self.save_path):
            try:
                with open(self.save_path, 'r') as f:
                    loaded = json.load(f)
                    self.state.update(loaded)
            except Exception as e:
                logger.warning(f"Load failed: {e}")
    
    def summarize(self):
        current_goal = self.get_active_goal()
        goal_text = current_goal.get("goal", "None") if current_goal else "None"
        
        return {
            "location": self.state.get("location", "Unknown"),
            "current_goal": goal_text,
            "last_action": self.state.get("last_action", "None"),
            "queue_size": len(_GLOBAL_PENDING)
        }


def memory_step(perception_output, planning_output, memory):
    # Prefer using the passed-in memory object when available (preserve state)
    module = None
    if isinstance(memory, MemoryModule):
        module = memory
    else:
        module = MemoryModule()
        # If caller passed a plain dict, merge it into the new module state
        if isinstance(memory, dict):
            try:
                module.state.update(memory)
            except Exception:
                pass

    # Safely extract last_action from planning_output (which may be None or a string)
    last_action = None
    try:
        if isinstance(planning_output, dict):
            last_action = planning_output.get("action")
        elif isinstance(planning_output, (list, tuple)) and planning_output:
            last_action = planning_output[0]
        elif isinstance(planning_output, str):
            last_action = planning_output
    except Exception:
        last_action = None

    module.update(perception_output, last_action=last_action)
    return module.summarize()
