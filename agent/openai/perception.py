#!/usr/bin/env python3
"""
perception.py
------------
Visual scene understanding for the Pokémon Emerald agent.

Responsibilities:
- Capture emulator frame + state snapshot
- Run minimal structured scene classification via VLM
- Avoid redundant calls via visual hashing / caching
- Return compact dict: {scene_type, summary, key_elements}
"""

import time
import hashlib
import json
from PIL import Image
from utils.helpers import frame_to_base64, add_text_update
# from utils.llm_logger import log_llm_interaction, log_llm_error # unused currently
# from .system_prompt import SYSTEM_PROMPT, PERCEPTION_PROMPT, PERCEPTION_FORMAT_HINT #unused currently
from utils.vlm import VLMBackend
import logging

logger = logging.getLogger(__name__)


class PerceptionModule:
    """Manages visual perception with redundancy checks and structured output."""

    def __init__(self, vlm: VLMBackend, cache_ttl: float = 3.0):
        """
        Args:
            vlm: Vision-language backend implementing get_query()
            cache_ttl: Seconds to keep last frame hash before forcing re-check
        """
        self.vlm = vlm
        self.last_hash = None
        self.last_result = None
        self.last_timestamp = 0
        self.cache_ttl = cache_ttl
        self.call_count = 0

    def _hash_frame(self, frame: Image.Image) -> str:
        """Compute SHA1 hash of the downscaled frame for redundancy detection."""
        resized = frame.resize((64, 48)).convert("L")
        return hashlib.sha1(resized.tobytes()).hexdigest()
    
    def _extract_frame_from_observation(self, observation):
        """Return a PIL.Image from observation (dict or image/ndarray)."""
        try:
            from PIL import Image
            import numpy as np
        except Exception:
            return observation

        # Already PIL?
        if hasattr(observation, "convert"):
            return observation

        # ndarray -> PIL
        if hasattr(observation, "shape"):
            try:
                return Image.fromarray(observation)
            except Exception:
                pass

        # Dict -> common keys
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
        """
        Ensure key_elements is always a dict.
        Accepts:
          - dict → return as-is
          - "" or None → {}
          - "a=1, b=2" → {"a": "1", "b": "2"}
          - "something" → {"something": True}
        """
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
        """
        Use structured data (no OCR) to classify scene.
        Returns a dict: {"scene_type": ..., "summary": ..., "key_elements": ...}
        """
        # Defensive: normalize containers
        obs = obs or {}
        state = obs.get("state", {}) if isinstance(obs.get("state"), dict) else {}

        # Pull common flags from various servers
        # Try several names so we work across backends
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

        # Map / location
        map_name = (
            (obs.get("map") or {}).get("name")
            or state.get("map_name")
            or obs.get("location")
        )
        px = (obs.get("player") or {}).get("x", (state.get("player_x")))
        py = (obs.get("player") or {}).get("y", (state.get("player_y")))

        # Party info
        party = obs.get("party") or state.get("party") or []
        party_count = len(party) if isinstance(party, list) else 0

        # Opponent (if available)
        opp_name = None
        if isinstance(obs.get("battle"), dict):
            opp_name = obs["battle"].get("opponent_name")
        if not opp_name and isinstance(state.get("battle_state"), dict):
            opp_name = state["battle_state"].get("opponent_name")

        # Priority of classification (no OCR):
        # 1) Title  2) Battle  3) Dialogue  4) Menu  5) Cutscene  6) Map  7) Unknown
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
        # MAP only if we have no dialogue/menu/cutscene signal
        if not dialogue_open and not menu_open and not cutscene:
            if map_name or (px is not None and py is not None):
                loc = map_name or "Overworld"
                pos = f"({px},{py})" if (px is not None and py is not None) else ""
                return {
                    "scene_type": "MAP",
                    "summary": f"{loc} {pos}".strip(),
                    "key_elements": f"Party={party_count}"
                }

        # Unknown: let VLM help later
        return {
            "scene_type": "UNKNOWN",
            "summary": "",
            "key_elements": ""
        }

    def should_refresh(self, frame_hash: str) -> bool:
        """Determine if we should re-run perception on this frame."""
        if self.last_hash != frame_hash:
            return True
        # force periodic refresh to catch subtler transitions
        return (time.time() - self.last_timestamp) > self.cache_ttl

    def analyze(self, observation, memory):
        # 1) Always normalize to an image to avoid .resize errors downstream
        img = self._extract_frame_from_observation(observation)
        if img is None and isinstance(observation, dict):
            # We can still classify from state even with no image
            pass

        # 2) Structured-state-first classification (no OCR)
        inferred = self._infer_scene_from_state(
            observation if isinstance(observation, dict) else {},
            memory,
        )

        # If we have a confident scene (not UNKNOWN), enrich it and return
        if inferred["scene_type"] != "UNKNOWN":
            # 1) normalize KE right away so later lines never crash
            normalized_ke = self._normalize_key_elements(inferred.get("key_elements"))

            # 2) enrich from raw observation
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

            # make clock easy to spot
            if "clock" in (inferred.get("summary") or "").lower():
                normalized_ke["clock"] = True

            inferred["key_elements"] = normalized_ke
            return inferred

        # 3) Ambiguous => cautiously ask VLM (image+prompt)
        #    Keep this prompt compact; the goal is only to disambiguate.
        if img is None:
            # No image and state unknown: fall back to safest default
            add_text_update("⚠️ Perception: neither image nor state → fallback CUTSCENE", category="PERCEPTION")
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
            add_text_update(f"⚠️ Perception VLM error: {e} → fallback CUTSCENE", category="PERCEPTION")
            return {"scene_type": "CUTSCENE", "summary": "Scripted scene.", "key_elements": ""}

        # 4) Parse the VLM text minimally (be liberal in what you accept)
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

        # Sanity: bound SceneType to known set or fall back to CUTSCENE
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
        # if server already gave walkable info, keep it
        if walk:
            normalized_ke["walkable"] = walk
        else:
            # else, at least mark that we don't know
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
        return {"scene_type": scene_type, "summary": summary, "key_elements": normalized_ke}

    def _parse_result(self, text: str) -> dict:
        """Parse LLM output into structured fields with guardrails."""
        try:
            # Normalize whitespace and split into lines
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            result = {"scene_type": "UNKNOWN", "summary": "", "key_elements": {}}

            for line in lines:
                lower = line.lower()
                if lower.startswith("scenetype:"):
                    result["scene_type"] = line.split(":", 1)[1].strip().upper()
                elif lower.startswith("summary:"):
                    result["summary"] = line.split(":", 1)[1].strip()
                elif lower.startswith("keyelements:"):
                    raw = line.split(":", 1)[1].strip()
                    # Try parse as JSON or key=value pairs
                    try:
                        result["key_elements"] = json.loads(raw)
                    except Exception:
                        parts = [p.strip() for p in raw.replace(";", ",").split(",") if p.strip()]
                        kv = {}
                        for p in parts:
                            if "=" in p:
                                k, v = p.split("=", 1)
                                kv[k.strip()] = v.strip()
                            else:
                                kv[p] = True
                        result["key_elements"] = kv

            return result

        except Exception as e:
            logger.warning(f"Failed to parse perception result: {e}")
            return {
                "scene_type": "UNKNOWN",
                "summary": text[:120],
                "key_elements": {}
            }

def perception_step(vlm, observation, memory):
    """
    Compatibility wrapper. Converts dict-based memory into MemoryModule
    before running perception.
    """
    from agent.memory import MemoryModule  # safe local import

    # Wrap dict into MemoryModule if needed
    if isinstance(memory, dict):
        temp_mem = MemoryModule()
        temp_mem.state.update(memory)
        memory = temp_mem

    # Instantiate the perception module correctly
    module = PerceptionModule(vlm)
    # Depending on your analyze signature:
    result = module.analyze(observation, memory)
    # enforce dict key_elements even from here
    if isinstance(result, dict):
        per_ke = result.get("key_elements")
        if not isinstance(per_ke, dict):
            result["key_elements"] = module._normalize_key_elements(per_ke)
    return result