"""
modules/ — Modules fonctionnels de haut niveau inspirés des aires de Broca et Wernicke.

Ce package implémente les deux modules principaux du framework :
- wernicke.py  : Module sémantique (encodeur) — traitement et représentation du sens
- broca.py     : Module syntaxique (décodeur) — génération et vérification de structure
- arcuate.py   : Fascicule arqué (canal inter-modules) — communication spikante bidirectionnelle
"""

from modules.wernicke import WernickeModule
from modules.broca import BrocaModule
from modules.arcuate import ArcuateFasciculus

__all__ = ["WernickeModule", "BrocaModule", "ArcuateFasciculus"]
