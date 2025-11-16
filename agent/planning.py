import json
import re
import logging
import random
from utils.state_formatter import format_state_for_llm
from .system_prompt import SYSTEM_PROMPT, PLANNING_PROMPT

logger = logging.getLogger(__name__)

# Speedrun goals
DEFAULT_EMERALD_LT_PLAN = [
    {"id": "start_game", "goal": "Start game and complete intro", "status": "pending", "completion_cues": ["your very own adventure", "well, i'll be expecting you"]},
    {"id": "leave_the_truck", "goal": "Exit the truck by moving RIGHT", "status": "pending", "completion_cues": ["your room is upstairs", "player's house", "mom"], "movement_only": True},
    {"id": "go_upstairs", "goal": "Go upstairs to second floor", "status": "pending", "completion_cues": ["2f", "second floor"]},
    {"id": "player_bedroom_entered", "goal": "Find clock in bedroom", "status": "pending", "completion_cues": ["this is your room", "nintendo gamecube"]},
    {"id": "set_bedroom_clock", "goal": "Set the clock", "status": "pending", "completion_cues": ["set the time", "set the clock", "better set it"]},
    {"id": "return_downstairs", "goal": "Go back downstairs", "status": "pending", "completion_cues": ["1f"]},
    {"id": "exit_player_house", "goal": "Leave house to Littleroot Town", "status": "pending", "completion_cues": ["littleroot"], "movement_only": True},
    {"id": "go_to_birch_lab", "goal": "Go to Professor Birch's lab", "status": "pending", "completion_cues": ["professor", "lab", "birch"]},
    {"id": "find_route_101", "goal": "Find Route 101", "status": "pending", "completion_cues": ["route 101", "route101"]},
    {"id": "find_professor_birch", "goal": "Find Professor Birch being chased", "status": "pending", "completion_cues": ["professor", "help", "birch"]},
    {"id": "choose_starter", "goal": "Choose Mudkip as starter", "status": "pending", "completion_cues": ["chose mudkip", "received pokemon"]},
    {"id": "rustboro_gym_challenge", "goal": "Defeat Roxanne for badge", "status": "pending", "completion_cues": ["roxanne", "stone badge"]},
]
class PlanningModule:
    """Planning with proper scene priority."""
    def __init__(self):
        self.initial_long_term_plan = DEFAULT_EMERALD_LT_PLAN
        # No hard-coded movement-only set here; read per-goal flags from LT_PLAN
    
    def _build_map_context(self, observation):
        """Build map context from observation state."""
        context_lines = []
        
        state = observation.get("state") or {}
        if not state:
            return []
        
        try:
            # Get full formatted state (includes MapStitcher data)
            full_context = format_state_for_llm(state, include_debug_info=False, include_npcs=False)
            
            # Extract map section
            lines = full_context.split('\n')
            in_map = False
            for line in lines:
                if '--- MAP:' in line:
                    in_map = True
                elif in_map and line.startswith('---'):
                    break
                if in_map:
                    context_lines.append(line)
            
            if context_lines:
                # Add position info
                player = state.get("player", {})
                pos = player.get("position", {})
                if pos:
                    x, y = pos.get('x', '?'), pos.get('y', '?')
                    context_lines.append(f"\nYou are at ({x}, {y})")
                    
        except Exception as e:
            logger.warning(f"Map context failed: {e}")
        
        return context_lines
    
    def _analyze_movement_preview(self, observation):
        """
        Analyze movement preview from game state.
        Returns walkable/blocked directions and special tiles.
        """
        walkable = []
        blocked = []
        special = {}
        
        state = observation.get("state", {})
        if not state:
            return {'walkable': [], 'blocked': [], 'special': {}}
        
        # FIRST: Try to extract from state.walkable dict (most reliable)
        walkable_dict = state.get("walkable", {})
        if isinstance(walkable_dict, dict) and walkable_dict:
            for direction, tile_info in walkable_dict.items():
                direction_upper = direction.upper()
                if direction_upper in ['UP', 'DOWN', 'LEFT', 'RIGHT']:
                    walkable.append(direction_upper)
                    # Check for special tile types
                    if isinstance(tile_info, dict):
                        behavior = (tile_info.get('behavior') or '').upper()
                        if 'STAIRS' in behavior or 'WARP' in behavior:
                            special[direction_upper] = 'stairs'
                        elif 'DOOR' in behavior:
                            special[direction_upper] = 'door'
                        elif 'GRASS' in behavior:
                            special[direction_upper] = 'grass'
            if walkable:
                return {'walkable': walkable, 'blocked': blocked, 'special': special}
        
        # SECOND: Try calculating from map grid + player position
        try:
            walkable_calc, blocked_calc = self._calculate_walkable_from_map(observation)
            if walkable_calc or blocked_calc:
                print(f"[PLAN] Calculated from map: Walkable={walkable_calc}, Blocked={blocked_calc}")
                return {'walkable': walkable_calc, 'blocked': blocked_calc, 'special': special}
        except Exception as e:
            logger.warning(f"Map grid calculation failed: {e}")
        
        # THIRD: Try parsing formatted state for MOVEMENT PREVIEW section
        try:
            formatted_state = format_state_for_llm(state)
            lines = formatted_state.split('\n')
            
            in_preview = False
            for line in lines:
                if 'MOVEMENT PREVIEW:' in line:
                    in_preview = True
                    continue
                
                if in_preview:
                    if not line.strip() or line.startswith('---'):
                        break
                    
                    if ':' in line:
                        parts = line.strip().split(':')
                        if len(parts) >= 2:
                            direction = parts[0].strip()
                            rest = parts[1].strip()
                            
                            if direction in ['UP', 'DOWN', 'LEFT', 'RIGHT']:
                                if 'WALKABLE' in rest:
                                    walkable.append(direction)
                                    # Check for special tiles
                                    if 'Stairs/Warp' in rest:
                                        special[direction] = 'stairs'
                                    elif 'Door/Entrance' in rest:
                                        special[direction] = 'door'
                                    elif 'Tall grass' in rest:
                                        special[direction] = 'grass'
                                    elif 'Jump ledge' in rest and 'can jump' in rest:
                                        special[direction] = 'ledge'
                                elif 'BLOCKED' in rest:
                                    blocked.append(direction)
        except Exception as e:
            logger.warning(f"Movement preview analysis failed: {e}")

        # FOURTH: As a last resort, derive from ASCII map context (P-centered grid)
        # This uses the '=== YOUR MAP ===' section produced by state formatter.
        if not walkable and not blocked:
            try:
                map_lines = self._build_map_context(observation) or []
                grid_rows = []
                reading_grid = False
                for ln in map_lines:
                    if ln.strip().startswith('--- MAP:'):
                        reading_grid = True
                        continue
                    if reading_grid:
                        if not ln.strip():
                            break
                        if ln.strip().startswith('Legend:'):
                            break
                        # Convert a visual row into token cells; keep single-char tokens (# . D L T N P ?)
                        # Many rows have spaces for alignment; split on spaces and drop empties
                        tokens = [tok for tok in ln.strip().split(' ') if tok != '']
                        if tokens:
                            grid_rows.append(tokens)
                # Find P
                p_y = p_x = None
                for ri, row in enumerate(grid_rows):
                    for ci, tok in enumerate(row):
                        if tok == 'P':
                            p_y, p_x = ri, ci
                            break
                    if p_y is not None:
                        break
                # Neighbor lookup
                def tok_at(dy, dx):
                    y = (p_y + dy) if p_y is not None else None
                    x = (p_x + dx) if p_x is not None else None
                    if y is None or x is None:
                        return None
                    if y < 0 or y >= len(grid_rows):
                        return None
                    row = grid_rows[y]
                    if x < 0 or x >= len(row):
                        return None
                    return row[x]
                if p_y is not None:
                    ascii_rules_walkable = {'.', 'D'}  # doors treated as enterable
                    ascii_rules_blocked = {'#', 'N', 'T'}
                    neighbors = {
                        'UP': tok_at(-1, 0),
                        'DOWN': tok_at(1, 0),
                        'LEFT': tok_at(0, -1),
                        'RIGHT': tok_at(0, 1),
                    }
                    for d, tok in neighbors.items():
                        if tok in ascii_rules_walkable:
                            walkable.append(d)
                            if tok == 'D':
                                special[d] = 'door'
                        elif tok in ascii_rules_blocked or tok is None:
                            blocked.append(d)
                        else:
                            # Unknown tokens (e.g., '?', 'L') → conservative: treat as blocked
                            blocked.append(d)
                    if walkable or blocked:
                        print(f"[PLAN] Derived from ASCII map: Walkable={walkable}, Blocked={blocked}, Special={special}")
            except Exception as e:
                logger.warning(f"ASCII map walkable derivation failed: {e}")
        
        return {
            'walkable': walkable,
            'blocked': blocked,
            'special': special
        }
    
    def _get_coords(self, observation):
        """Extract player coordinates from observation (check both state and top-level)"""
        try:
            # Try top-level player keys first (from client)
            player = observation.get("player", {})
            x = player.get("x")
            y = player.get("y")
            if x is not None and y is not None:
                return (x, y)
            
            # Fallback to state.player.position
            state = observation.get("state", {})
            player = state.get("player", {})
            pos = player.get("position", {})
            x, y = pos.get('x'), pos.get('y')
            if x is not None and y is not None:
                return (x, y)
        except:
            pass
        return None
    
    def _extract_map_grid(self, observation):
        """Extract 2D map grid from state to determine walkable tiles."""
        try:
            state = observation.get("state", {})
            map_data = state.get("map", {})
            
            # Try to get tiles from map data
            tiles = map_data.get("tiles", [])
            width = map_data.get("width", 0)
            height = map_data.get("height", 0)
            
            if tiles and width > 0 and height > 0:
                return tiles, width, height
        except:
            pass
        return None, 0, 0
    
    def _calculate_walkable_from_map(self, observation):
        """Calculate walkable directions from map grid and player position."""
        walkable = []
        blocked = []
        
        coords = self._get_coords(observation)
        if not coords:
            return walkable, blocked
        
        px, py = coords
        tiles, width, height = self._extract_map_grid(observation)
        
        if not tiles or width == 0 or height == 0:
            return walkable, blocked
        
        # Define walkable tile behaviors
        WALKABLE_BEHAVIORS = {
            'NORMAL', 'DOOR', 'STAIRS', 'WARP', 'LEDGE',
            'TALL_GRASS', 'WATER', 'ICE', 'SAND'
        }
        BLOCKED_BEHAVIORS = {'WALL', 'BARRIER', 'IMPASSABLE'}
        
        # Check each direction
        directions = [
            ('UP', 0, -1),
            ('DOWN', 0, 1),
            ('LEFT', -1, 0),
            ('RIGHT', 1, 0),
        ]
        
        for dir_name, dx, dy in directions:
            nx, ny = px + dx, py + dy
            
            # Check bounds
            if nx < 0 or nx >= width or ny < 0 or ny >= height:
                blocked.append(dir_name)
                continue
            
            # Get tile at (nx, ny)
            tile_idx = ny * width + nx
            if tile_idx >= len(tiles):
                blocked.append(dir_name)
                continue
            
            tile = tiles[tile_idx]
            
            # Determine if walkable
            is_walkable = False
            if isinstance(tile, dict):
                # tile is {tile_id, behavior, collision, elevation}
                behavior = tile.get('behavior', '')
                collision = tile.get('collision', 0)
                
                # Walkable if behavior indicates it or collision is 0
                if behavior and any(b in str(behavior).upper() for b in WALKABLE_BEHAVIORS):
                    is_walkable = True
                elif collision == 0 and not any(b in str(behavior).upper() for b in BLOCKED_BEHAVIORS):
                    is_walkable = True
            elif isinstance(tile, (list, tuple)):
                # tile is [tile_id, behavior, collision, elevation] or similar
                if len(tile) >= 3:
                    behavior = tile[1] if len(tile) > 1 else ''
                    collision = tile[2] if len(tile) > 2 else 0
                    
                    if behavior and any(b in str(behavior).upper() for b in WALKABLE_BEHAVIORS):
                        is_walkable = True
                    elif collision == 0 and not any(b in str(behavior).upper() for b in BLOCKED_BEHAVIORS):
                        is_walkable = True
            
            if is_walkable:
                walkable.append(dir_name)
            else:
                blocked.append(dir_name)
        
        return walkable, blocked
        
    
    def decide(self, vlm, observation, memory):
        """Make planning decision with proper priority."""
        
        # Normalize
        if not isinstance(observation, dict):
            observation = {"scene_type": "UNKNOWN", "summary": str(observation), "key_elements": {}}
        
        scene_type = observation.get("scene_type", "UNKNOWN")
        summary = observation.get("summary", "")
        summary_lower = summary.lower()
        
        # Extract map name early (available for all scenes)
        map_info = observation.get("map", {}) or {}
        map_name = map_info.get("name") or map_info.get("current_map") or "Unknown"
        
        print(f"[PLAN] scene={scene_type}, summary={summary[:80]}")
        
        # Initialize runtime flags for dialogue control and scene tracking
        try:
            if not hasattr(memory, "_runtime_flags"):
                memory._runtime_flags = {}
            rf = memory._runtime_flags
        except Exception:
            rf = {}
        rf["step_counter"] = rf.get("step_counter", 0) + 1
        prev_scene = rf.get("last_scene")
        
        # Update goals
        if hasattr(memory, 'update_goal_status'):
            try:
                memory.update_goal_status(observation)
            except Exception as e:
                logger.warning(f"Goal update failed: {e}")

        # Determine active goal early so we can apply goal-level restrictions (e.g., movement_only)
        active_goal_early = None
        try:
            if hasattr(memory, 'get_active_goal'):
                active_goal_early = memory.get_active_goal()
        except Exception:
            active_goal_early = None
        restrict_queue = False
        goal_id_early = None
        if isinstance(active_goal_early, dict):
            restrict_queue = bool(active_goal_early.get('movement_only', False))
            goal_id_early = active_goal_early.get('id')
        
        # HARDCODED SEQUENCES for tutorial goals (skip LLM for speed)
        # Queue once, complete goal immediately, then fall through to queue processing
        if goal_id_early == "player_bedroom_entered" and scene_type == "MAP":
            # Find clock: move to clock and interact (only queue once)
            # CRITICAL: Only trigger if at correct starting position (7, 2) on 2F
            coords = self._get_coords(observation)
            clock_queued = memory._runtime_flags.get("clock_sequence_queued", False) if hasattr(memory, "_runtime_flags") else False
            if not clock_queued and coords and coords == (7, 2) and hasattr(memory, "enqueue_actions"):
                memory.enqueue_actions(["LEFT", "LEFT", "UP", "A", "A"])
                if hasattr(memory, "_runtime_flags"):
                    memory._runtime_flags["clock_sequence_queued"] = True
                # Complete goal immediately after queueing
                if hasattr(memory, "complete_current_goal"):
                    memory.complete_current_goal(goal_id_early)
                print(f"[PLAN] 🎯 HARDCODED: player_bedroom_entered at {coords} → queued [LEFT, LEFT, UP, A, A] and completed goal")
        
        if goal_id_early == "set_bedroom_clock" and scene_type == "MAP":
            # Set clock: confirm time (only queue once)
            coords = self._get_coords(observation)
            set_clock_queued = memory._runtime_flags.get("set_clock_sequence_queued", False) if hasattr(memory, "_runtime_flags") else False
            if not set_clock_queued and coords and coords == (5, 2) and hasattr(memory, "enqueue_actions"):
                memory.enqueue_actions(["A", "UP", "A", "UP", "A", "UP", "A", "UP"])
                if hasattr(memory, "_runtime_flags"):
                    memory._runtime_flags["set_clock_sequence_queued"] = True
                # Complete goal immediately after queueing
                if hasattr(memory, "complete_current_goal"):
                    memory.complete_current_goal(goal_id_early)
                print("[PLAN] 🎯 HARDCODED: set_bedroom_clock → queued [UP, A, UP, A, UP] and completed goal")
        
        if goal_id_early == "return_downstairs" and scene_type == "MAP":
            # Go downstairs: move to stairs and go down
            downstairs_queued = memory._runtime_flags.get("downstairs_sequence_queued", False) if hasattr(memory, "_runtime_flags") else False
            if not downstairs_queued and hasattr(memory, "enqueue_actions"):
                memory.enqueue_actions(["RIGHT", "DOWN", "DOWN", "DOWN", "DOWN"])
                if hasattr(memory, "_runtime_flags"):
                    memory._runtime_flags["downstairs_sequence_queued"] = True
                # Complete goal immediately after queueing
                if hasattr(memory, "complete_current_goal"):
                    memory.complete_current_goal(goal_id_early)
                print("[PLAN] 🎯 HARDCODED: return_downstairs → queued [RIGHT, DOWN*4] and completed goal")
        
        # PRIORITY 1: Scene-based overrides (BEFORE queue)
        # These scenes MUST use specific actions, ignore queue
        
        if scene_type == "TITLE":
            print("[PLAN] Title screen detected - pressing A")
            rf["last_scene"] = scene_type
            return {"action": "A", "reason": "title"}
        
        # Handle UNKNOWN as title screen if we see title screen indicators
        if scene_type == "UNKNOWN":
            if any(kw in summary_lower for kw in ["press start", "main menu", "pokemon emerald", "new game", "continue"]):
                print("[PLAN] UNKNOWN scene with title keywords - pressing A")
                rf["last_scene"] = scene_type
                return {"action": "A", "reason": "title_fallback"}
        
        if scene_type == "BATTLE":
            rf["last_scene"] = scene_type
            return {"action": "A", "reason": "battle"}
        
        # CRITICAL: Dialogue and cutscenes ONLY use A
        # Enhanced handling: mash A, escalate if stuck, then drain queued A's
        if scene_type in ("DIALOGUE", "CUTSCENE"):
            try:
                rf["dialogue_streak"] = rf.get("dialogue_streak", 0) + 1
                # On first entry into dialogue, enqueue an A burst
                if prev_scene not in ("DIALOGUE", "CUTSCENE") and hasattr(memory, "enqueue_actions"):
                    memory.enqueue_actions(["A", "A", "A", "A"])
                    print("[PLAN] ✳ Dialogue entered — enqueued A burst")
                # Escalate if stuck for too long
                if rf["dialogue_streak"] >= 60:
                    rf["last_scene"] = scene_type
                    print("[PLAN] ⏫ Dialogue escalation: START")
                    return {"action": "START", "reason": "dialogue_escalation_start"}
                if rf["dialogue_streak"] >= 40:
                    rf["last_scene"] = scene_type
                    print("[PLAN] ⏫ Dialogue escalation: B")
                    return {"action": "B", "reason": "dialogue_escalation_b"}
                # Drain queued A during dialogue if present
                if hasattr(memory, "peek_next_action") and hasattr(memory, "pop_next_action"):
                    nxt = memory.peek_next_action()
                    if nxt == "A":
                        memory.pop_next_action()
                        rf["last_scene"] = scene_type
                        return {"action": "A", "reason": "dialogue_queue"}
            except Exception as e:
                logger.warning(f"Dialogue handler error: {e}")
            rf["last_scene"] = scene_type
            return {"action": "A", "reason": "dialogue"}

        # Reset dialogue tracking when we leave dialogue/cutscene
        if prev_scene in ("DIALOGUE", "CUTSCENE") and scene_type not in ("DIALOGUE", "CUTSCENE"):
            rf["dialogue_streak"] = 0
            # Allow a one-time escape step on entering MAP to avoid re-triggering
            rf["post_dialogue_escape_done"] = False
        
        # Post-dialogue escape: if we just left dialogue and are now on MAP,
        # enqueue a single safe step to avoid re-triggering the same NPC/trigger.
        if scene_type == "MAP" and prev_scene in ("DIALOGUE", "CUTSCENE") and not rf.get("post_dialogue_escape_done", False):
            try:
                movement_info_pd = self._analyze_movement_preview(observation)
                walkable_pd = movement_info_pd.get('walkable') or []
                if walkable_pd and hasattr(memory, "enqueue_actions"):
                    preferred = ["RIGHT", "DOWN", "LEFT", "UP"]
                    escape_dir = next((d for d in preferred if d in walkable_pd), walkable_pd[0])
                    memory.enqueue_actions([escape_dir])
                    rf["post_dialogue_escape_done"] = True
                    print(f"[PLAN] ↪ Post-dialogue escape enqueued: {escape_dir}")
            except Exception as e:
                logger.warning(f"Post-dialogue escape failed: {e}")

        # PRIORITY 2: Queue - HIGHEST PRIORITY for MAP/MENU
        # If there's anything in the queue, use it before generating new plans
        if hasattr(memory, "peek_next_action"):
            nxt_peek = memory.peek_next_action()
            if nxt_peek:
                # Pop and use the queued action
                if hasattr(memory, "pop_next_action"):
                    nxt = memory.pop_next_action()
                    print(f"[PLAN] ✓ Using queued action: {nxt}")
                    rf["last_scene"] = scene_type
                    return {"action": nxt, "reason": "queued"}
        
        # PRIORITY 3: Menu handling
        if scene_type == "MENU":
            # Check if we've already processed the naming screen (persisted in memory)
            naming_handled = memory.state.get("naming_handled", False) if hasattr(memory, "state") else False
            
            if not naming_handled:
                is_naming = any(kw in summary_lower for kw in ["name", "keyboard", "character", "enter"])
                if is_naming:
                    # Mark as handled in memory state so it doesn't trigger again
                    if hasattr(memory, "state"):
                        memory.state["naming_handled"] = True
                    if hasattr(memory, "enqueue_actions"):
                        memory.enqueue_actions(["A", "A", "A", "A", "START"])
                    print("[PLAN] Naming screen - completing")
                    rf["last_scene"] = scene_type
                    return {"action": "START", "reason": "naming"}
        
        # Handle TRANSITION scene
        if scene_type == "TRANSITION":
            rf["last_scene"] = scene_type
            return {"action": "WAIT", "reason": "transition"}
        
        # PRIORITY 4: MAP navigation
        if scene_type == "MAP":
            # Get goal
            active_goal = None
            goal_text = "Progress the game"
            goal_id = "no_goal"
            
            try:
                if hasattr(memory, "get_active_goal"):
                    active_goal = memory.get_active_goal()
                    if active_goal and isinstance(active_goal, dict):
                        goal_text = active_goal.get("goal", goal_text)
                        goal_id = active_goal.get("id", goal_id)
            except:
                pass

            # Determine if this is a navigation-style goal early
            is_navigation_goal = any(
                kw in goal_text.lower() for kw in [
                    "go", "upstairs", "downstairs", "exit", "leave", "enter", "find", "reach"
                ]
            )
            
            # Special: truck exit
            if goal_id in ["leave_the_truck", "start_game"]:
                if "truck" in summary_lower or "box" in summary_lower:
                    print("[PLAN] In truck - exiting RIGHT")
                    return {"action": "RIGHT", "reason": "truck_exit"}
            
            # CRITICAL: Analyze movement options
            movement_info = self._analyze_movement_preview(observation)
            walkable = movement_info['walkable']
            blocked = movement_info['blocked']
            special = movement_info['special']
            
            # DEBUG: Log walkable/blocked directions
            print(f"[PLAN] Walkable: {walkable}, Blocked: {blocked}, Special: {special}")
            
            # Get movement memory from memory module
            coords = self._get_coords(observation)
            movement_memory = ""
            if coords and hasattr(memory, 'get_movement_memory'):
                movement_memory = memory.get_movement_memory(coords)
            
            # Update explored map in memory
            if hasattr(memory, 'update_explored_map'):
                memory.update_explored_map(observation)
            
            # For navigation goals, identify target tiles (doors/stairs)
            target_info = ""
            if is_navigation_goal and hasattr(memory, 'find_tiles_of_type'):
                # Find stairs and doors in explored map
                stairs = memory.find_tiles_of_type(map_name or "current", ['S'])
                doors = memory.find_tiles_of_type(map_name or "current", ['D'])
                
                # Get entry coords for current map to avoid targeting the entry door/stairs
                entry_coords = None
                try:
                    if hasattr(memory, 'get_entry_coords'):
                        entry_coords = memory.get_entry_coords(map_name or "current")
                except Exception:
                    entry_coords = None

                def filter_away_entry(targets_list):
                    if not targets_list:
                        return targets_list
                    if not entry_coords:
                        return targets_list
                    ex, ey = entry_coords
                    # Exclude the tile we likely came from (exact match or adjacent within 1 tile)
                    return [t for t in targets_list if abs(t[0]-ex) + abs(t[1]-ey) > 1]

                # For upstairs/downstairs goals, target stairs or doors
                if 'upstairs' in goal_text.lower() or 'downstairs' in goal_text.lower():
                    targets = stairs if stairs else doors
                    targets = filter_away_entry(targets)
                    if targets and coords:
                        # Find closest target
                        closest = min(targets, key=lambda t: abs(t[0]-coords[0]) + abs(t[1]-coords[1]))
                        dx = closest[0] - coords[0]
                        dy = closest[1] - coords[1]
                        direction = ""
                        if abs(dx) > abs(dy):
                            direction = "RIGHT" if dx > 0 else "LEFT"
                        else:
                            direction = "DOWN" if dy > 0 else "UP"
                        
                        target_type = "stairs" if closest in stairs else "door"
                        target_info = f"\n🎯 TARGET: {target_type.upper()} at ({closest[0]},{closest[1]}) - move {direction} (distance: {abs(dx)+abs(dy)} tiles)"
                        print(f"[PLAN] Target (memory): {target_type.upper()} at ({closest[0]},{closest[1]}) → {direction}")
                # For exit/leave goals, prefer DOORS only and avoid entry tile
                elif 'exit' in goal_text.lower() or 'leave' in goal_text.lower():
                    targets = filter_away_entry(doors)
                    if targets and coords:
                        closest = min(targets, key=lambda t: abs(t[0]-coords[0]) + abs(t[1]-coords[1]))
                        dx = closest[0] - coords[0]
                        dy = closest[1] - coords[1]
                        if abs(dx) > abs(dy):
                            direction = "RIGHT" if dx > 0 else "LEFT"
                        else:
                            direction = "DOWN" if dy > 0 else "UP"
                        target_info = f"\n🎯 TARGET: DOOR at ({closest[0]},{closest[1]}) - move {direction} (distance: {abs(dx)+abs(dy)} tiles)"
                        print(f"[PLAN] Target (memory): DOOR at ({closest[0]},{closest[1]}) → {direction}")
            
            # Build map context
            map_context = self._build_map_context(observation)
            # Print the user's current formatted map while running (compact view)
            if map_context:
                try:
                    print("[PLAN] MAP CONTEXT:")
                    for ln in map_context:
                        print(ln)
                except Exception:
                    pass
            # If the active goal requires movement-only, read from the active goal flag
            restrict_to_movement = False
            if isinstance(active_goal, dict):
                restrict_to_movement = bool(active_goal.get('movement_only', False))
            if restrict_to_movement:
                print(f"[PLAN] Goal {goal_id} requires movement-only actions")
            
            # Log player location and map info
            print(f"[PLAN] Player location: {map_name}, Coords: {coords}")
            print(f"[PLAN] Active goal: {goal_id} - {goal_text}")
            
            # Build enhanced prompt with movement info
            prompt_parts = [
                f"GOAL: {goal_text}",
                "",
                f"SCENE: {summary}",
                "",
            ]
            
            # Add location and map info prominently
            coords = self._get_coords(observation)
            if coords:
                x, y = coords
                prompt_parts.append(f"CURRENT LOCATION: {map_name or 'Unknown map'} at ({x}, {y})")
            else:
                prompt_parts.append(f"CURRENT LOCATION: {map_name or 'Unknown map'}")
            prompt_parts.append("")
            
            # Add movement analysis
            if walkable or blocked:
                prompt_parts.append("=== MOVEMENT ANALYSIS ===")
                if walkable:
                    prompt_parts.append(f"WALKABLE: {', '.join(walkable)}")
                    if special:
                        special_desc = [f"{d}={t}" for d, t in special.items()]
                        prompt_parts.append(f"SPECIAL: {', '.join(special_desc)}")
                if blocked:
                    prompt_parts.append(f"BLOCKED: {', '.join(blocked)}")
                prompt_parts.append("")
            
            # Add movement memory if available
            if movement_memory:
                prompt_parts.append(movement_memory)
                prompt_parts.append("")
            
            # Add target info if found
            if target_info:
                prompt_parts.append(target_info)
                # Help the model relate moves to coordinates when a target exists
                if coords:
                    prompt_parts.append("COORDINATE HINTS: (x, y) — RIGHT:+x, LEFT:-x, DOWN:+y, UP:-y.")
                    prompt_parts.append("Compare target (tx, ty) to current (x, y): if tx>x move RIGHT; if tx<x move LEFT; if ty>y move DOWN; if ty<y move UP.")
                prompt_parts.append("")
            
            if map_context:
                prompt_parts.append("=== YOUR MAP ===")
                prompt_parts.extend(map_context)
                prompt_parts.append("")

            # Fallback: derive target from ASCII map if navigation goal and no target_info yet
            if is_navigation_goal and not target_info and map_context:
                try:
                    # Build grid from map_context (lines after header until blank/Legend)
                    grid_rows = []
                    for ln in map_context:
                        stripped = ln.rstrip()
                        if not stripped:
                            break
                        if stripped.startswith('Legend:'):
                            break
                        if '--- MAP:' in stripped:
                            continue
                        # Split while preserving single-character tokens and filter to valid grid tokens
                        raw_tokens = [t for t in stripped.split(' ') if t]
                        valid_set = {'.', '#', 'D', 'L', 'T', 'N', 'P', '?'}
                        tokens = [t for t in raw_tokens if len(t) == 1 and t in valid_set]
                        if tokens and len(tokens) == len(raw_tokens):
                            grid_rows.append(tokens)
                    # Find player P position
                    p_y = p_x = None
                    for yi, row in enumerate(grid_rows):
                        for xi, tok in enumerate(row):
                            if tok == 'P':
                                p_y, p_x = yi, xi
                                break
                        if p_y is not None:
                            break
                    # Collect doors/stairs
                    targets = []
                    for yi, row in enumerate(grid_rows):
                        for xi, tok in enumerate(row):
                            if tok in ('D','S'):
                                targets.append((xi, yi, tok))
                    if p_y is not None and targets:
                        # Choose nearest by Manhattan
                        nearest = min(targets, key=lambda t: abs(t[0]-p_x) + abs(t[1]-p_y))
                        tx, ty, sym = nearest
                        dx = tx - p_x
                        dy = ty - p_y
                        # Primary direction toward target
                        if abs(dx) >= abs(dy):
                            primary_dir = 'RIGHT' if dx > 0 else ('LEFT' if dx < 0 else ('DOWN' if dy>0 else 'UP'))
                        else:
                            primary_dir = 'DOWN' if dy > 0 else ('UP' if dy < 0 else ('RIGHT' if dx>0 else 'LEFT'))
                        # Only add if movement direction is actually walkable
                        if not walkable or primary_dir in walkable:
                            target_desc = 'stairs' if sym == 'S' else 'door'
                            ascii_target_line = f"🎯 TARGET (ASCII): {target_desc.upper()} approx {abs(dx)+abs(dy)} tiles away → move {primary_dir}"
                            prompt_parts.append(ascii_target_line)
                            print(f"[PLAN] Target (ASCII): {target_desc.upper()} approx {abs(dx)+abs(dy)} tiles → {primary_dir}")
                            prompt_parts.append("")
                except Exception as e:
                    print(f"[PLAN] ASCII target derivation failed: {e}")
            
            prompt_parts.extend([
                "Choose ONE action to progress toward your goal:",
                "",
                "CRITICAL RULES:",
            ])
            
            # If this goal should only use movement, make it explicit
            if restrict_to_movement:
                prompt_parts.append("- GOAL REQUIRES MOVEMENT ONLY: Do NOT use A, INTERACT, or menu actions. Use only MOVE <DIR> <1-3>.")
            
            # Navigation goals: prioritize movement over interaction
            is_navigation_goal = any(kw in goal_text.lower() for kw in ["go", "upstairs", "downstairs", "exit", "leave", "enter", "find", "reach"])
            if is_navigation_goal and walkable:
                prompt_parts.append("- NAVIGATION GOAL: Prioritize MOVEMENT to reach your destination. Only use A if you must interact with doors/stairs/objects to proceed.")
            # Special explicit hint for exiting the player house
            if (goal_id == 'exit_player_house') or ('exit player house' in goal_text.lower()):
                prompt_parts.append("- OBJECTIVE: Reach the Littleroot Town exterior door/portal (NOT the entry stairs). Move toward the exterior door leading outside.")
            
            if walkable:
                prompt_parts.append(f"- Movement (ONLY use these): {', '.join(walkable)}")
            else:
                prompt_parts.append("- Movement: No walkable directions detected, use A to interact")
            
            if blocked:
                prompt_parts.append(f"- NEVER use these (BLOCKED): {', '.join(blocked)}")
            
            prompt_parts.append("- A: Interact with doors (D), NPCs, objects, stairs")
            prompt_parts.append("- If stuck, try a different walkable direction")
            prompt_parts.append("")
            
            # Enable multi-action plans after early goals to speed up execution
            allow_multi_action = False
            if goal_id not in ["start_game", "leave_the_truck", "player_bedroom_entered", "set_bedroom_clock"]:
                allow_multi_action = True
            
            prompt_parts.extend([
                "",
                "Study the map carefully:",
                "- Stairs (S) lead to other floors",
                "- Doors (D) connect rooms and can usually be entered with A",
                "- Unknown areas (?) should be explored to find your destination",
                "- Plan your route to minimize backtracking",
                ""
            ])
            
            if is_navigation_goal:
                if allow_multi_action:
                    prompt_parts.extend([
                        "Create a MOVEMENT PLAN to reach your destination:",
                        "1. Identify where stairs/doors/target likely are (check unexplored '?' areas)",
                        "2. Plot shortest walkable path",
                        "3. Return 2-3 movement actions as a JSON array: [\"UP\", \"RIGHT\", \"UP\"]",
                        "",
                        "PLAN:"
                    ])
                else:
                    prompt_parts.extend([
                        "Take ONE step toward your destination.",
                        "If you see stairs (S) or relevant doors (D), move toward them.",
                        "Explore unknown areas (?) to find your target.",
                        "",
                        "ACTION:"
                    ])
            elif allow_multi_action:
                prompt_parts.extend([
                    "You may provide 2-3 sequential actions as JSON: [\"RIGHT\", \"A\", \"UP\"]",
                    "Or single action: RIGHT",
                    "",
                    "ACTION(S):"
                ])
            else:
                prompt_parts.append("ACTION:")
            
            prompt = "\n".join(prompt_parts)
            
            # Get LLM decision
            response = vlm.get_text_query(prompt, module_name="planning")
            print(f"[PLAN] LLM response: {response}")
            
            # Parse action(s) from response - may be single or array
            actions = self._parse_actions(response, walkable, allow_multi=allow_multi_action)
            
            # Track recommended target direction for soft override
            recommended_direction = None
            if target_info:
                # Extract direction from target_info string
                for d in ['UP', 'DOWN', 'LEFT', 'RIGHT']:
                    if f"move {d}" in target_info or f"→ {d}" in target_info:
                        recommended_direction = d
                        break
            
            # If multiple actions, queue all but first
            if isinstance(actions, list) and len(actions) > 1:
                if hasattr(memory, "enqueue_actions"):
                    memory.enqueue_actions(actions[1:])
                    print(f"[PLAN] Queued {len(actions)-1} additional actions: {actions[1:]}")
                action = actions[0]
            elif isinstance(actions, list) and len(actions) == 1:
                action = actions[0]
            else:
                action = actions
            
            # Validate and return
            try:
                # HARD CONSTRAINT: go_upstairs goal ONLY allows UP action
                if active_goal and active_goal.get("id") == "go_upstairs":
                    if action in ["DOWN", "A"]:
                        print(f"[PLAN] 🚫 HARD OVERRIDE for go_upstairs: {action} → UP (only UP allowed)")
                        action = "UP"
                
                # SOFT OVERRIDE: If we have a target direction and LLM chose something else
                elif is_navigation_goal and recommended_direction and action != recommended_direction:
                    # Only override if LLM picked A or a non-optimal move
                    if action == "A" or (action in ["UP", "DOWN", "LEFT", "RIGHT"] and action in walkable):
                        # Check if recommended direction is walkable
                        if not walkable or recommended_direction in walkable:
                            print(f"[PLAN] 🎯 Soft override: {action} → {recommended_direction} (toward target)")
                            action = recommended_direction
                
                # CRITICAL: Validate movement against walkable directions
                if action in ["UP", "DOWN", "LEFT", "RIGHT"]:
                    if walkable and action not in walkable:
                        print(f"[PLAN] ❌ LLM chose BLOCKED direction {action}!")
                        if walkable:
                            action = walkable[0]
                            print(f"[PLAN] ✓ Using first walkable direction instead: {action}")
                        else:
                            if restrict_to_movement:
                                print(f"[PLAN] ✋ No walkable directions and goal requires movement — returning WAIT")
                                return {"action": "WAIT", "reason": "no_walkable_movement_only_goal"}
                            else:
                                action = "A"
                                print(f"[PLAN] ✓ No walkable directions, using A to interact")
                else:
                    # Check if action is a non-movement button press
                    action_upper = str(action).upper().strip()
                    # Extract base direction if present
                    base_action = action_upper.split()[0] if ' ' in action_upper else action_upper
                    
                    if base_action in ["A", "B", "START", "SELECT"]:
                        # Non-move actions
                        if restrict_to_movement:
                            if walkable:
                                action = walkable[0]
                                print(f"[PLAN] Overriding non-move action to movement-only: {action}")
                            else:
                                print(f"[PLAN] ✋ Goal requires movement but no walkable directions — WAIT")
                                return {"action": "WAIT", "reason": "movement_required_but_blocked"}
                
                print(f"[PLAN] Final action: {action}")
                return {"action": action, "reason": "validated_navigation"}
                    
            except Exception as e:
                logger.warning(f"LLM failed: {e}")
            
            # Fallback: Use first walkable direction or A
            if walkable:
                action = walkable[0]
                print(f"[PLAN] Fallback - using first walkable: {action}")
                return {"action": action, "reason": "fallback_walkable"}
            else:
                print(f"[PLAN] Fallback - no walkable directions, using A")
                return {"action": "A", "reason": "fallback_interact"}
    
    def _parse_actions(self, response, walkable_dirs, allow_multi=False):
        """Parse single or multiple actions from LLM response."""
        if not response:
            return ["A"]
        
        # Try to parse as JSON array first if multi-action allowed
        if allow_multi:
            try:
                # Clean response - sometimes LLM wraps in markdown
                clean = response.strip()
                if clean.startswith("```") and clean.endswith("```"):
                    lines = clean.split("\n")
                    clean = "\n".join(lines[1:-1])
                
                parsed = json.loads(clean.strip())
                if isinstance(parsed, list) and len(parsed) > 0:
                    # Validate and normalize each action
                    validated = []
                    for act in parsed[:3]:  # Max 3 actions
                        validated_act = self._normalize_action(str(act), walkable_dirs)
                        # _normalize_action may return a list (e.g., ["RIGHT", "RIGHT", "RIGHT"])
                        if isinstance(validated_act, list):
                            validated.extend(validated_act)
                        else:
                            validated.append(validated_act)
                    return validated[:3]  # Cap total at 3
            except (json.JSONDecodeError, ValueError):
                pass
        
        # Fall back to single action parsing
        result = self._normalize_action(response, walkable_dirs)
        # If normalize returned a list (e.g., "RIGHT 3" -> ["RIGHT", "RIGHT", "RIGHT"]), return it
        if isinstance(result, list):
            return result
        return [result]
    
    def _normalize_action(self, action_str, walkable_dirs):
        """Normalize a single action string to standard format.
        Returns either a single action string OR a list of actions (for DIR N expansion).
        """
        action_str = action_str.upper().strip()
        
        # Check for "DIR N" pattern (e.g., "RIGHT 3" -> ["RIGHT", "RIGHT", "RIGHT"])
        dir_step_match = re.match(r"(UP|DOWN|LEFT|RIGHT)\s+(\d+)", action_str)
        if dir_step_match:
            direction = dir_step_match.group(1)
            steps = int(dir_step_match.group(2))
            # Cap at 3 steps max for safety
            steps = min(steps, 3)
            return [direction] * steps
        
        # Simple direction -> normalize
        if action_str in ["UP", "DOWN", "LEFT", "RIGHT"]:
            return action_str
        if action_str in ["A", "B", "START", "SELECT"]:
            return action_str
        
        # MOVE X N -> [X, X, ...]
        match = re.match(r"MOVE\s+(UP|DOWN|LEFT|RIGHT)(?:\s+(\d+))?", action_str)
        if match:
            direction = match.group(1)
            steps = int(match.group(2)) if match.group(2) else 1
            steps = min(steps, 3)
            if steps > 1:
                return [direction] * steps
            return direction
        
        # PRESS X -> X
        match = re.match(r"PRESS\s+(A|B|START|SELECT|UP|DOWN|LEFT|RIGHT)", action_str)
        if match:
            return match.group(1)
        
        # Default
        return "A"

def planning_step(vlm, perception_output, memory):
    """Compatibility wrapper."""
    from agent.memory import MemoryModule
    
    if not hasattr(memory, 'state') or not hasattr(memory, 'get_active_goal'):
        if isinstance(memory, dict):
            temp_mem = MemoryModule()
            temp_mem.state.update(memory)
            if "long_term_plan" not in temp_mem.state:
                temp_mem.state["long_term_plan"] = DEFAULT_EMERALD_LT_PLAN
            memory = temp_mem
        else:
            memory = MemoryModule()
    
    module = PlanningModule()
    return module.decide(vlm, perception_output, memory)
