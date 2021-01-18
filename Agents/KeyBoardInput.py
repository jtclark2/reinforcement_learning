"""
Leverages the "keyboard" library: https://pypi.org/project/keyboard/
"""

import keyboard
from enum import Enum

class KeyMap(Enum):
    Left = ('left', 0)
    No_Force = ('no_force', 1)
    Right = ('right', 2)

class KeyBoardInput:
    def get_action(self):
        action = KeyMap.No_Force
        if keyboard.is_pressed('left'): # 'a'
            action = KeyMap.Left
        elif keyboard.is_pressed('right'): # 'd'
            action = KeyMap.Right
        # print(f"action: {action.name}: {action.value} ")
        return action.value[0]

if __name__ == "__main__":
    k = KeyBoardInput()
    while True:
        print(k.get_action())
