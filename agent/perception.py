#!/usr/bin/env python3
"""perception.py - Scene classification"""

import logging
from PIL import Image
import numpy as np
from utils.state_formatter import format_state_for_llm, detect_dialogue_on_frame
import json
import time
import os

logger = logging.getLogger(__name__)

# Add file handler for perception debug logs
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
LLM_LOG_DIR = os.path.join(REPO_ROOT, 'llm_logs')
perc_debug_log_path = os.path.join(LLM_LOG_DIR, 'perception_debug.log')
os.makedirs(LLM_LOG_DIR, exist_ok=True)
perc_file_handler = logging.FileHandler(perc_debug_log_path, mode='a')
perc_file_handler.setLevel(logging.DEBUG)
perc_formatter = logging.Formatter('[%(asctime)s] %(message)s')
perc_file_handler.setFormatter(perc_formatter)
logger.addHandler(perc_file_handler)
logger.setLevel(logging.DEBUG)

# Ensure full debug file exists so the user can tail it
perception_full_debug_path = os.path.join(LLM_LOG_DIR, "perception_full_debug.jsonl")
try:
    if not os.path.exists(perception_full_debug_path):
        open(perception_full_debug_path, "a").close()
except Exception:
    pass


class PerceptionModule:
    """Manages visual perception with redundancy checks and structured output."""
    def __init__(self, vlm, cache_ttl: float = 3.0):
        self.vlm = vlm
        self.last_hash = None
        self.last_result = None
        self.last_timestamp = 0
        self.cache_ttl = cache_ttl
        self.call_count = 0

    def is_black_frame(self, frame) -> bool:
        """Check if frame is transition/loading (black screen)"""
        try:
            if hasattr(frame, 'convert'):
                img = frame
            elif hasattr(frame, 'shape'):
                img = Image.fromarray(frame)
            else:
                return False
            
            img_array = np.array(img)
            mean_brightness = np.mean(img_array)
            std_dev = np.std(img_array)
            
            # Black if very dark OR very uniform
            return mean_brightness < 10 or std_dev < 5
        except:
            return False
    
    def _extract_frame(self, observation):
        try:
            from PIL import Image
            import numpy as np
        except Exception:
            return observation

        if hasattr(observation, "convert"):
            return observation
        if hasattr(observation, "shape"):
            try:
                return Image.fromarray(observation)
            except Exception:
                pass
        if isinstance(observation, dict):
            img = (
                observation.get("frame")
                or observation.get("image")
                or observation.get("screenshot")
            )
            if img is None:
                return None
            if hasattr(img, "convert"):
                return img
            if hasattr(img, "shape"):
                try:
                    return Image.fromarray(img)
                except Exception:
                    return None
        return None
    

    def _normalize_key_elements(self, key_elements):
        if isinstance(key_elements, dict):
            return key_elements
        if not key_elements:
            return {}
        raw = str(key_elements)
        parts = [p.strip() for p in raw.replace(";", ",").split(",") if p.strip()]
        out = {}
        for p in parts:
            if "=" in p:
                k, v = p.split("=", 1)
                out[k.strip()] = v.strip()
            else:
                out[p] = True
        return out

    def _infer_scene_from_state(self, obs: dict, mem) -> dict:
        obs = obs or {}
        state = obs.get("state", {}) if isinstance(obs.get("state"), dict) else {}

        is_title = bool(
            obs.get("title_screen")
            or state.get("is_title")
            or state.get("title")
        )
        in_battle = bool(
            obs.get("battle", {}).get("in_battle")
            or state.get("in_battle")
            or state.get("battle")
        )
        dialogue_open = bool(
            obs.get("dialogue", {}).get("open")
            or state.get("dialogue_open")
            or obs.get("textbox_open")
            or state.get("textbox_open")
        )
        menu_open = bool(
            obs.get("menu", {}).get("open")
            or state.get("menu_open")
            or state.get("in_menu")
        )
        cutscene = bool(
            obs.get("cutscene")
            or state.get("cutscene")
            or state.get("script_active")
        )

        map_name = (
            (obs.get("map") or {}).get("name")
            or state.get("map_name")
            or obs.get("location")
        )
        px = (obs.get("player") or {}).get("x", (state.get("player_x")))
        py = (obs.get("player") or {}).get("y", (state.get("player_y")))
        party = obs.get("party") or state.get("party") or []
        party_count = len(party) if isinstance(party, list) else 0
        opp_name = None
        if isinstance(obs.get("battle"), dict):
            opp_name = obs["battle"].get("opponent_name")
        if not opp_name and isinstance(state.get("battle_state"), dict):
            opp_name = state["battle_state"].get("opponent_name")
        
        # DEBUG
        if dialogue_open:
            print(f"[PERC._infer] dialogue_open=True: textbox_open={state.get('textbox_open')}, dialogue_open={state.get('dialogue_open')}")

        if is_title:
            return {
                "scene_type": "TITLE",
                "summary": "Main menu visible.",
                "key_elements": "Screen=Title"
            }
        if in_battle:
            summ = f"Battle in progress"
            if opp_name:
                summ += f" vs {opp_name}"
            return {
                "scene_type": "BATTLE",
                "summary": summ,
                "key_elements": f"Party={party_count}"
            }
        if dialogue_open:
            return {
                "scene_type": "DIALOGUE",
                "summary": "Dialogue textbox visible.",
                "key_elements": f"Party={party_count}"
            }
        if menu_open:
            return {
                "scene_type": "MENU",
                "summary": "Menu open.",
                "key_elements": f"Party={party_count}"
            }
        if cutscene:
            return {
                "scene_type": "CUTSCENE",
                "summary": "Scripted scene.",
                "key_elements": f"Party={party_count}"
            }
        if not dialogue_open and not menu_open and not cutscene:
            if map_name or (px is not None and py is not None):
                loc = map_name or "Overworld"
                pos = f"({px},{py})" if (px is not None and py is not None) else ""
                return {
                    "scene_type": "MAP",
                    "summary": f"{loc} {pos}".strip(),
                    "key_elements": f"Party={party_count}"
                }
        return {
            "scene_type": "UNKNOWN",
            "summary": "",
            "key_elements": ""
        }

    def _classify_scene_consensus(self, observation, frame=None):
        """Combine multiple signals (memory, state, OCR, frame heuristics) to pick a scene_type.

        Returns the same result dict as `_infer_from_state` but with a more robust decision.
        """
        obs = observation or {}
        game = obs.get('game', {}) or {}
        player = obs.get('player', {}) or {}
        map_info = obs.get('map', {}) or {}
        visual = obs.get('visual', {}) or {}

        # Direct high-confidence signals
        if game.get('is_in_battle') or game.get('in_battle'):
            return {"scene_type": "BATTLE", "summary": "In battle (memory)", "key_elements": {}, "state": obs.get('state', {})}

        gs = (game.get('game_state') or '').lower()
        if gs == 'menu':
            return {"scene_type": "MENU", "summary": "In menu (memory)", "key_elements": {}, "state": obs.get('state', {})}
        if gs == 'title':
            return {"scene_type": "TITLE", "summary": "Title screen (memory)", "key_elements": {}, "state": obs.get('state', {})}

        # Memory-level dialogue detection (from memory_reader)
        dialogue_meta = game.get('dialogue_detected', {}) or {}
        mem_has_dialogue = bool(dialogue_meta.get('has_dialogue'))
        mem_conf = float(dialogue_meta.get('confidence') or 0.0)

        # OCR / textual signals
        dialog_text = visual.get('dialogue_text') or game.get('dialog_text')

        # Frame-based signals
        frame_dialog_conf = 0.0
        try:
            # try state_formatter's detector first (works with base64 or array)
            if visual.get('screenshot_base64'):
                r = detect_dialogue_on_frame(screenshot_base64=visual.get('screenshot_base64'))
                frame_dialog_conf = max(frame_dialog_conf, float(r.get('confidence', 0.0)))
            if frame is not None:
                # If frame is PIL Image, convert to numpy array for state_formatter
                try:
                    import numpy as _np
                    if hasattr(frame, 'convert'):
                        arr = _np.array(frame)
                    else:
                        arr = frame
                    r2 = detect_dialogue_on_frame(frame_array=arr)
                    frame_dialog_conf = max(frame_dialog_conf, float(r2.get('confidence', 0.0)))
                except Exception:
                    pass
        except Exception:
            frame_dialog_conf = 0.0

        # Perception's own frame detector (more conservative) if frame present
        perc_frame_conf = 0.0
        try:
            if frame is not None:
                vals = self._detect_dialogue_from_frame(frame, obs, debug=True)
                if isinstance(vals, tuple):
                    perc_frame_conf = float(vals[0])
                else:
                    perc_frame_conf = float(vals)
        except Exception:
            perc_frame_conf = 0.0

        # OCR detector result (if available)
        ocr_text = None
        ocr_conf = 0.0
        if getattr(self, 'ocr_detector', None) is not None and getattr(self, '_ocr_available', False) and frame is not None:
            try:
                ocr_text = self.ocr_detector.detect_dialogue_from_screenshot(frame)
                if ocr_text:
                    ocr_conf = 1.0
            except Exception:
                ocr_text = None

        # Consensus logic: prioritize explicit memory signals and OCR, then frame detectors
        # If memory strongly indicates dialogue, and OCR/frame don't contradict strongly, choose DIALOGUE
        if mem_has_dialogue and mem_conf >= 0.7:
            # if frame evidence strongly contradicts (very low), and we have map/player info, prefer MAP
            if (perc_frame_conf < 0.2 and frame_dialog_conf < 0.2) and (map_info.get('name') or (player.get('x') is not None and player.get('y') is not None)):
                return {"scene_type": "MAP", "summary": "Map (memory/dialogue contradicted by frame)", "key_elements": {}, "state": obs.get('state', {})}
            return {"scene_type": "DIALOGUE", "summary": "In dialogue (memory)", "key_elements": {}, "state": obs.get('state', {})}

        # If OCR found readable text, pick DIALOGUE
        if dialog_text or ocr_text:
            return {"scene_type": "DIALOGUE", "summary": "In dialogue (OCR/text present)", "key_elements": {}, "state": obs.get('state', {})}

        # If any frame-based detector indicates dialogue with good confidence, pick DIALOGUE
        if perc_frame_conf >= 0.8 or frame_dialog_conf >= 0.6:
            return {"scene_type": "DIALOGUE", "summary": "In dialogue (frame-based)", "key_elements": {}, "state": obs.get('state', {})}

        # If map info / coordinates exist, prefer MAP
        if (map_info and (map_info.get('tiles') or map_info.get('tile_names') or map_info.get('current_map') or map_info.get('name'))) or (player.get('x') is not None and player.get('y') is not None):
            return {"scene_type": "MAP", "summary": f"On map: {map_info.get('name') or map_info.get('current_map') or 'Unknown'}", "key_elements": {"Location": map_info.get('name')}, "state": obs.get('state', {})}

        # Fallback to _infer_from_state
        return self._infer_from_state(observation)
    
    def analyze(self, observation, memory=None, frame=None):
        img = self._extract_frame(observation)
        if img is None and isinstance(observation, dict):
            pass
        inferred = self._infer_scene_from_state(
            observation if isinstance(observation, dict) else {},
            memory,
        )
        
        # CRITICAL: Override DIALOGUE if visual evidence suggests MAP
        # (e.g., game_state='dialog' but we have map tiles and player position, indicating player is in world not in dialogue)
        if inferred["scene_type"] == "DIALOGUE":
            try:
                state = observation.get("state", {}) if isinstance(observation, dict) else {}
                # If we have map data (tiles) and player position, player is in the world, not in an active dialogue
                map_data = state.get("map", {})
                player = state.get("player", {})
                pos = player.get("position")
                
                has_map = bool(map_data and map_data.get("tiles"))
                has_position = bool(pos and (pos.get("x") is not None and pos.get("y") is not None))
                
                if has_map and has_position:
                    print(f"[PERC] Override: DIALOGUE -> MAP (has map tiles and player position at {pos.get('x')},{pos.get('y')})")
                    inferred["scene_type"] = "MAP"
                    inferred["summary"] = f"Player in map at ({pos.get('x')},{pos.get('y')})"
            except Exception as e:
                print(f"[PERC] Override check failed: {e}")
        
        if inferred["scene_type"] != "UNKNOWN":
            normalized_ke = self._normalize_key_elements(inferred.get("key_elements"))
            state = observation.get("state", {}) if isinstance(observation, dict) else {}
            player = state.get("player") or {}
            pos = player.get("position") or player.get("coordinates")
            if isinstance(pos, dict):
                normalized_ke["player_x"] = pos.get("x")
                normalized_ke["player_y"] = pos.get("y")
            location = (
                state.get("game", {}).get("location_name")
                or state.get("map", {}).get("name")
            )
            if location:
                normalized_ke["Location"] = location
            if "clock" in (inferred.get("summary") or "").lower():
                normalized_ke["clock"] = True
            
            # CRITICAL: Include state and map for planning module
            result = inferred.copy()
            result["key_elements"] = normalized_ke
            result["state"] = state  # Pass full state for planning
            result["map"] = state.get("map", {})  # Pass map explicitly
            
            # Store perception scene type in observation for state formatter consistency
            if isinstance(observation, dict) and "game" in observation:
                observation["game"]["_perception_scene_type"] = inferred["scene_type"]
            
            # CRITICAL: Also copy into state.game so format_state_for_llm can see it
            if state and isinstance(state, dict) and "game" in state:
                state["game"]["_perception_scene_type"] = inferred["scene_type"]

            # Write a structured debug line for this frame
            try:
                self._write_full_debug(observation, result)
            except Exception:
                pass
            return result
        if img is None:
            return {"scene_type": "CUTSCENE", "summary": "Scripted scene.", "key_elements": ""}
        try:
            prompt = (
                "Briefly classify the game screen:\n"
                "Return exactly:\n"
                "SceneType: <MAP|BATTLE|MENU|DIALOGUE|CUTSCENE|TITLE>\n"
                "Summary: <≤2 sentences>\n"
                "KeyElements: <comma-separated key=value pairs or short list>\n"
            )
            text = self.vlm.get_query(img, prompt, module_name="perception") or ""
        except Exception as e:
            return {"scene_type": "CUTSCENE", "summary": "Scripted scene.", "key_elements": ""}
        scene_type, summary, key_elems = "UNKNOWN", "", ""
        try:
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            for ln in lines:
                if ln.lower().startswith("scenetype:"):
                    scene_type = ln.split(":", 1)[1].strip().upper()
                elif ln.lower().startswith("summary:"):
                    summary = ln.split(":", 1)[1].strip()
                elif ln.lower().startswith("keyelements:"):
                    key_elems = ln.split(":", 1)[1].strip()
        except Exception:
            pass
        valid = {"MAP", "BATTLE", "MENU", "DIALOGUE", "CUTSCENE", "TITLE"}
        if scene_type not in valid:
            scene_type = "CUTSCENE"
            if not summary:
                summary = "Scripted scene."
        normalized_ke = self._normalize_key_elements(key_elems)
        state = observation.get("state", {})
        player = (state.get("player") or {})
        pos = player.get("position") or player.get("coordinates")
        walk = state.get("walkable", {}) or {}
        if walk:
            normalized_ke["walkable"] = walk
        else:
            normalized_ke.setdefault("walkable", {})
        if isinstance(pos, dict):
            normalized_ke["player_x"] = pos.get("x")
            normalized_ke["player_y"] = pos.get("y")
        location = (
            state.get("game", {}).get("location_name")
            or state.get("map", {}).get("name")
        )
        if location:
            normalized_ke["Location"] = location
        if "clock" in (summary or "").lower():
            normalized_ke["clock"] = True
        
        # CRITICAL: Include state and map for planning module
        result = {
            "scene_type": scene_type,
            "summary": summary,
            "key_elements": normalized_ke,
            "state": state,  # Pass full state for planning to extract map/player info
            "map": state.get("map", {}),  # Pass map explicitly
        }
        try:
            self._write_full_debug(observation, result)
        except Exception:
            pass
        return result
    
    def _detect_dialogue_from_frame(self, frame, obs=None, debug=False):
        """Detect dialogue boxes or naming keyboard in frame (GBA resolution 240x160).
        Returns confidence score 0-1. Uses both bottom-region variance (dialogue box)
        and a high-frequency edge detector (useful for keyboard/naming screens).
        If `obs` is provided, the player's name is consulted to bias detection for
        naming screens (e.g. name is '????????' or empty).
        """
        try:
            # Convert frame to numpy array if needed
            if hasattr(frame, 'convert'):
                arr = np.array(frame)
            elif hasattr(frame, 'shape'):
                arr = frame
            else:
                return 0.0

            def _write_full_debug(self, observation, inferred):
                """Write full JSONL debug containing scene inference and selected state fields.

                Each line is a JSON object, one per frame. This is used for offline debugging and
                should be append-only to avoid spamming the console.
                """
                try:
                    os.makedirs("llm_logs", exist_ok=True)
                    out_path = os.path.join("llm_logs", "perception_full_debug.jsonl")
                    data = {
                        "ts": time.time(),
                        "scene_type": inferred.get("scene_type"),
                        "summary": inferred.get("summary"),
                        "key_elements": inferred.get("key_elements"),
                        "map_name": (observation.get("map") or {}).get("name") if isinstance(observation, dict) else None,
                        "player": (observation.get("player") or {}).get("x") if isinstance(observation, dict) else None,
                    }
                    # Add compact state info: game_state, dialog flags, walkable if present
                    state = (observation.get("state") or {}) if isinstance(observation, dict) else {}
                    data["game_state"] = state.get("game", {}).get("game_state") or state.get("game_state")
                    data["dialogue_open"] = bool(state.get("dialogue_open") or state.get("textbox_open"))
                    data["menu_open"] = bool(state.get("menu_open") or state.get("in_menu"))
                    data["walkable"] = state.get("walkable") if state else None
                    # Write compact map meta if available
                    map_info = state.get("map", {}) if state else {}
                    if map_info:
                        data["map_summary"] = {
                            "name": map_info.get("name"),
                            "width": map_info.get("width"),
                            "height": map_info.get("height"),
                        }

                    with open(out_path, "a") as f:
                        f.write(json.dumps(data, default=str) + "\n")
                except Exception:
                    # Do not raise; debug write should be best-effort
                    pass

            if arr is None or getattr(arr, 'size', 0) == 0:
                return 0.0

            h, w = arr.shape[:2]
            # Bottom region for dialogue boxes
            dialogue_region_height = min(60, max(24, h // 4))
            bottom_region = arr[-dialogue_region_height:, :, :3] if arr.ndim == 3 else arr[-dialogue_region_height:, :]

            # Basic variance-based dialogue score (text box presence)
            if bottom_region.size == 0:
                base_score = 0.0
            else:
                if bottom_region.ndim == 3:
                    variance = np.var(bottom_region, axis=(0, 1))
                    mean_variance = float(np.mean(variance))
                    # Dialogue boxes tend to be low-variance regions (uniform background).
                    # Invert and normalize variance so that LOWER variance => HIGHER base_score.
                    var_scale = 400.0
                    normalized = min(1.0, mean_variance / var_scale)
                    base_score = max(0.0, 1.0 - normalized)
                else:
                    std = float(np.std(bottom_region))
                    std_scale = 20.0
                    normalized = min(1.0, std / std_scale)
                    base_score = max(0.0, 1.0 - normalized)

            # Edge/keyboard detector (useful for naming screens): measure high-frequency
            # content in the central/top area where the keyboard is typically shown.
            # Compute simple gradient magnitude via differences (avoids cv2 dependency).
            center_h = min(h // 2, 120)
            top_region = arr[:center_h, :, :3] if arr.ndim == 3 else arr[:center_h, :]
            if top_region.size == 0:
                edge_score = 0.0
                top_edge_score = 0.0
            else:
                # Convert to grayscale for gradient computation
                if top_region.ndim == 3:
                    gray = np.dot(top_region[...,:3], [0.299, 0.587, 0.114])
                else:
                    gray = top_region
                # Horizontal and vertical gradients
                gx = np.abs(np.diff(gray, axis=1)).mean()
                gy = np.abs(np.diff(gray, axis=0)).mean()
                mean_grad = float((gx + gy) / 2.0)
                # Make edge detector less sensitive (require stronger gradients)
                edge_score = min(1.0, mean_grad / 40.0)

                # Top-edge detection in the bottom_region: dialogue boxes typically have a
                # distinct horizontal border near the top of the dialogue area. Compute vertical
                # gradient in the bottom_region and inspect the top rows for a strong horizontal edge.
                try:
                    if bottom_region.ndim == 3:
                        bottom_gray = np.dot(bottom_region[...,:3], [0.299, 0.587, 0.114])
                    else:
                        bottom_gray = bottom_region
                    # Vertical gradient (differences between rows)
                    vgrad = np.abs(np.diff(bottom_gray, axis=0))
                    # Look at the mean gradient in the top few rows of the bottom region
                    # (dialogue borders are typically narrow). Use fewer rows and make
                    # the scale larger so the detector is less likely to trigger on
                    # incidental texture in the scene.
                    top_rows = min(4, vgrad.shape[0])
                    top_grad_mean = float(vgrad[:top_rows, :].mean()) if top_rows > 0 else 0.0
                    # Scale to 0-1 but require a larger mean gradient to reach 1.0
                    top_edge_score = min(1.0, top_grad_mean / 40.0)
                except Exception:
                    top_edge_score = 0.0

            # If observation suggests the player name is unset, bias toward keyboard detection
            name_hint = None
            try:
                if isinstance(obs, dict):
                    player = obs.get('player', {}) or {}
                    name_hint = player.get('name')
            except Exception:
                name_hint = None

            name_unset = (not name_hint) or (isinstance(name_hint, str) and name_hint.strip().startswith('?'))

            # Color-based heuristics: many Pokemon dialogue boxes have a cyan/white
            # border and a light background. Detect presence of cyan and white
            # pixels in the bottom region to boost confidence.
            try:
                cyan_mask = None
                white_mask = None
                if bottom_region.ndim == 3:
                    r = bottom_region[:,:,0].astype(int)
                    g = bottom_region[:,:,1].astype(int)
                    b = bottom_region[:,:,2].astype(int)
                    # Cyan-ish: G and B significantly higher than R
                    cyan_mask = (b > 120) & (g > 110) & (r < 120)
                    white_mask = (r > 200) & (g > 200) & (b > 200)
                    cyan_pct = float(cyan_mask.sum()) / float(cyan_mask.size)
                    white_pct = float(white_mask.sum()) / float(white_mask.size)
                else:
                    cyan_pct = 0.0
                    white_pct = 0.0
            except Exception:
                cyan_pct = 0.0
                white_pct = 0.0

            # Combine scores: require BOTH a strong bottom-region signal AND keyboard/edge
            # signal if name is unset. Also prefer presence of a top-edge (dialogue border)
            # to avoid static scenes like the truck which may have low variance but no border.
            if name_unset:
                combined = min(base_score, edge_score * 1.2, top_edge_score * 1.2)
            else:
                # For normal dialogue, prefer at least two supporting signals (base, edge, top-edge).
                sig_base = 1 if base_score > 0.45 else 0
                sig_edge = 1 if edge_score > 0.45 else 0
                sig_top = 1 if top_edge_score > 0.45 else 0
                sig_count = sig_base + sig_edge + sig_top
                if sig_count >= 2:
                    # Strong consensus
                    combined = max(base_score, edge_score, top_edge_score)
                else:
                    # Weak single-signal case: downweight to avoid false positives
                    combined = max(base_score, edge_score, top_edge_score) * 0.45

            # Clamp
            combined = max(0.0, min(1.0, combined))
            if debug:
                return combined, base_score, edge_score, top_edge_score
            return combined
        except Exception as e:
            logger.debug(f"Frame dialogue detection failed: {e}")
            return 0.0

    def _write_full_debug(self, observation, inferred):
        """Write full JSONL debug containing scene inference and selected state fields.

        Each line is a JSON object, one per frame. This is used for offline debugging and
        should be append-only to avoid spamming the console.
        """
        try:
            os.makedirs(LLM_LOG_DIR, exist_ok=True)
            out_path = perception_full_debug_path
            data = {
                "ts": time.time(),
                "scene_type": inferred.get("scene_type"),
                "summary": inferred.get("summary"),
                "key_elements": inferred.get("key_elements"),
                "map_name": (observation.get("map") or {}).get("name") if isinstance(observation, dict) else None,
                "player": (observation.get("player") or {}).get("x") if isinstance(observation, dict) else None,
            }
            # Add compact state info: game_state, dialog flags, walkable if present
            state = (observation.get("state") or {}) if isinstance(observation, dict) else {}
            data["game_state"] = state.get("game", {}).get("game_state") or state.get("game_state")
            data["dialogue_open"] = bool(state.get("dialogue_open") or state.get("textbox_open"))
            data["menu_open"] = bool(state.get("menu_open") or state.get("in_menu"))
            data["walkable"] = state.get("walkable") if state else None
            # Write compact map meta if available
            map_info = state.get("map", {}) if state else {}
            if map_info:
                data["map_summary"] = {
                    "name": map_info.get("name"),
                    "width": map_info.get("width"),
                    "height": map_info.get("height"),
                }

            with open(out_path, "a") as f:
                f.write(json.dumps(data, default=str) + "\n")
            # Also write a brief debug line to the perception log
            logger.debug(f"WROTE PERCEPTION DEBUG: scene={data.get('scene_type')} map={data.get('map_summary', {}).get('name') if data.get('map_summary') else 'None'} ts={data.get('ts')}")
        except Exception:
            # Do not raise; debug write should be best-effort
            pass


def perception_step(vlm, observation, memory):
    from agent.memory import MemoryModule
    
    if isinstance(memory, dict):
        temp_mem = MemoryModule()
        temp_mem.state.update(memory)
        memory = temp_mem
    
    module = PerceptionModule(vlm)
    # Extract the frame (if present in the observation) and pass it to analyze
    frame = module._extract_frame(observation)
    return module.analyze(observation, memory, frame=frame)
