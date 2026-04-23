"""
Compatibility shim for old imports.

This project now uses:
  - Brain.py        (DQN model)
  - Run_aloha.py    (ALOHA training entry)

If older code imports DQN_ALOHA from this file, it will still work.
"""

from Brain import DQN as DQN_ALOHA

