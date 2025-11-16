#!/usr/bin/env python3
"""action.py - Button execution"""

import logging

logger = logging.getLogger(__name__)

VALID_BUTTONS = {"A", "B", "START", "SELECT", "UP", "DOWN", "LEFT", "RIGHT"}


class ActionModule:
    def __init__(self, emulator_interface=None):
        self.emu = emulator_interface
    
    def _parse_action(self, action_str):
        if not action_str:
            return ["A"]
        
        action = str(action_str).strip().upper()
        
        if action in VALID_BUTTONS:
            return [action]
        
        buttons = []
        for token in action.split():
            if token in VALID_BUTTONS:
                buttons.append(token)
        
        return buttons if buttons else ["A"]
    
    def execute(self, plan):
        if isinstance(plan, dict):
            raw_action = plan.get("action", "")
        elif isinstance(plan, str):
            raw_action = plan
        else:
            raw_action = ""
        
        buttons = self._parse_action(raw_action)
        buttons = [b for b in buttons if b in VALID_BUTTONS]
        
        if not buttons:
            buttons = ["A"]
        
        print(f"[ACTION] Executing: {buttons}")
        
        return {"executed": True, "action": raw_action, "commands": buttons}


def action_step(vlm, game_state, planning_output, perception_output):
    module = ActionModule(emulator_interface=None)
    
    if isinstance(planning_output, dict):
        raw_action = planning_output.get("action", "")
    elif isinstance(planning_output, str):
        raw_action = planning_output
    else:
        raw_action = ""
    
    buttons = module._parse_action(raw_action)
    buttons = [b for b in buttons if b in VALID_BUTTONS]
    
    return {"action": buttons}
