#!/usr/bin/env python3
"""
action.py
----------
Executes deterministic emulator inputs based on the agent's current plan.

Responsibilities:
- Translate single-line directives into valid button sequences
- Debounce duplicate inputs
- Provide fallback behavior on invalid plans
"""

import time
import re
import logging
from utils.helpers import add_text_update
from utils.llm_logger import log_llm_interaction
from .system_prompt import ACTION_RULES

logger = logging.getLogger(__name__)

VALID_BUTTONS = {"A", "B", "START", "SELECT", "UP", "DOWN", "LEFT", "RIGHT"}
MOVE_PATTERN = re.compile(r"MOVE\s+(UP|DOWN|LEFT|RIGHT)\s+(\d+)", re.IGNORECASE)
PRESS_PATTERN = re.compile(r"PRESS\s+([A-Z]+)", re.IGNORECASE)
MENU_PATTERN = re.compile(r"MENU\s*→\s*(.+)", re.IGNORECASE)


class ActionModule:
    """Handles execution of button commands from planner output."""

    def __init__(self, emulator_interface, hold_time: float = 0.08, cooldown: float = 0.05):
        """
        Args:
            emulator_interface: an object exposing .press_button(str) and .release_button(str)
            hold_time: time to hold each button press
            cooldown: delay between distinct inputs
        """
        self.emu = emulator_interface
        self.hold_time = hold_time
        self.cooldown = cooldown
        self.last_action = None
        self.last_exec_time = 0.0
        self.action_count = 0

    # ------------------------------------------------------------------
    # Main interface
    # ------------------------------------------------------------------
    def normalize_button_name(self, button: str) -> str:
        """
        Normalize planner outputs to valid GBA button names.
        Converts strings like 'PRESS A' or 'PRESS START' to 'A', 'START'.
        """
        return button.replace("PRESS ", "").strip().upper()

    def execute(self, plan: dict) -> dict:
        """
        Execute the given plan as button presses.

        Args:
            plan (dict): {"action": "MOVE UP 3", "reason": "..."}
        Returns:
            dict with execution metadata.
        """
        start = time.time()
        raw_action = plan.get("action", "").strip().upper()

        # Debounce duplicate actions if issued too close together
        if raw_action == self.last_action and (time.time() - self.last_exec_time) < 0.25:
            logger.debug(f"Skipping duplicate action: {raw_action}")
            return {"executed": False, "action": raw_action, "reason": "duplicate"}

        try:
            commands = self._parse_action(raw_action)
            for cmd in commands:
                self._press(cmd)

            self.last_action = raw_action
            self.last_exec_time = time.time()
            self.action_count += len(commands)

            log_llm_interaction(
                interaction_type="action",
                prompt=raw_action,
                response=str(commands),
                duration=time.time() - start,
                metadata={"actions": len(commands)},
                model_info={"module": "action"}
            )

            add_text_update(f"🎮 Executed: {raw_action}", category="ACTION")

            return {"executed": True, "action": raw_action, "commands": commands}

        except Exception as e:
            logger.warning(f"Action execution failed: {e}")
            safe_cmd = ["A"]
            self._press("A")
            return {"executed": False, "action": raw_action, "error": str(e), "fallback": safe_cmd}

    # ------------------------------------------------------------------
    # Parsing logic
    # ------------------------------------------------------------------

    def _parse_action(self, action: str):
        """
        Parse an action string into a list of valid button presses.
        Supported directives:
            MOVE <dir> <steps>
            PRESS <button>
            INTERACT
            MENU → <path>
        """
        commands = []

        # Normalize once
        action = action.strip().upper()

        # ✅ Direct single-button commands from planner/queue, e.g. "A", "START", "UP"
        if action in VALID_BUTTONS:
            return [action]

        # MOVE UP 3
        move_match = MOVE_PATTERN.match(action)
        if move_match:
            direction, steps = move_match.groups()
            steps = min(int(steps), 8)  # guardrail: no long paths
            commands.extend([direction] * steps)
            return commands

        # PRESS A
        press_match = PRESS_PATTERN.match(action)
        if press_match:
            button = self.normalize_button_name(press_match.group(1))
            if button in VALID_BUTTONS:
                commands.append(button)
                return commands

        # INTERACT
        if action.startswith("INTERACT"):
            commands.append("A")
            return commands

        # MENU →
        menu_match = MENU_PATTERN.match(action)
        if menu_match:
            path = menu_match.group(1).split("→")
            for step in path:
                step = step.strip().upper()
                if step in VALID_BUTTONS:
                    commands.append(step)
                elif step in ["BAG", "POKEMON", "SAVE"]:
                    # these require START then A
                    commands.extend(["START", "A"])
                else:
                    commands.append("A")  # fallback confirmation
            return commands

        # ✅ Simple comma/space-separated button lists, e.g. "A, A, START"
        tokens = re.split(r"[,\s]+", action)
        buttons = [t for t in tokens if t in VALID_BUTTONS]
        if buttons:
            return buttons

        # If nothing matched, fallback to PRESS A
        commands = [self.normalize_button_name(c) for c in commands]
        logger.debug(f"No valid pattern found for action: {action}, defaulting to A.")
        return commands or ["A"]

    # ------------------------------------------------------------------
    # Emulator interaction
    # ------------------------------------------------------------------

    def _press(self, button: str):
        """Physically press a button via emulator interface."""
        if button not in VALID_BUTTONS:
            logger.warning(f"Ignored invalid button: {button}")
            return

        try:
            self.emu.press_button(button)
            time.sleep(self.hold_time)
            self.emu.release_button(button)
            time.sleep(self.cooldown)
        except Exception as e:
            logger.error(f"Emulator button press failed for {button}: {e}")

def action_step(vlm, game_state, planning_output, perception_output):
    module = ActionModule(emulator_interface=None)

    # existing normalization
    if isinstance(planning_output, dict):
        raw_action = planning_output.get("action", "") or ""
    elif isinstance(planning_output, str):
        raw_action = planning_output
    else:
        raw_action = ""

    # existing tolerant parsing (yours already handles "A, A, A, ..." fine)
    buttons = module._parse_action(raw_action)
    buttons = [b for b in buttons if b in VALID_BUTTONS]
    summary = (perception_output or {}).get("summary", "").lower()

    # keep your name-entry guard
    if any(k in summary for k in ("name", "character", "rename")):
        return {"action": buttons}

    # default: whatever planner asked for
    return {"action": buttons}
