"""
core/ — Primitives de bas niveau du framework SNN-PC hybride.

Ce package implémente les composants fondamentaux :
- neuron.py    : Neurone LIF augmenté par les erreurs de prédiction PC
- synapse.py   : Synapse avec plasticité STDP compatible PC
- oscillator.py: Horloge thêta/gamma pour la coordination temporelle
- encoding.py  : Encodage des erreurs de prédiction en trains de spikes
"""

from core.neuron import LIFNeuron
from core.synapse import STDPSynapse
from core.oscillator import OscillatoryClock
from core.encoding import rate_encode, phase_encode

__all__ = [
    "LIFNeuron",
    "STDPSynapse",
    "OscillatoryClock",
    "rate_encode",
    "phase_encode",
]
