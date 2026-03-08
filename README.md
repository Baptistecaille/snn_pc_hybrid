# SNN-PC Hybrid — Broca–Wernicke Architecture

A modular framework combining **Spiking Neural Networks (SNN)** and **Predictive Coding (PC)**, structured around a dual architecture inspired by Broca and Wernicke brain areas.

---

## Mathematical Model

### 1. LIF Neuron augmented by Predictive Coding

The membrane dynamics follow:

$$\tau_m \frac{dV}{dt} = -(V - V_{rest}) + R \cdot I_{syn}(t) + \eta \cdot \varepsilon(t)$$

where $\varepsilon(t) = r_{observed}(t) - r_{predicted}(t)$ is the prediction error. Discretized with Euler explicit at step $\Delta t$:

$$V(t+\Delta t) = \alpha_m V(t) + (1 - \alpha_m) V_{rest} + \frac{\Delta t}{\tau_m} [R \cdot I_{syn} + \eta \cdot \varepsilon]$$

with $\alpha_m = 1 - \Delta t / \tau_m$.

### 2. Surrogate Gradient for Spikes

The Heaviside spike function $H(V - V_{th})$ is non-differentiable. The surrogate:

$$\sigma'(x) = \frac{1}{\beta \left(1 + |x/\beta|\right)^2}$$

approximates its derivative in backpropagation while preserving binary spike generation in the forward pass.

### 3. Variational Free Energy

The main optimization objective:

$$\mathcal{F} = \frac{1}{2} \sum_{\ell \in \{W, B\}} \sum_i \frac{(\varepsilon_i^{(\ell)})^2}{\sigma_\ell^2}$$

Minimizing $\mathcal{F}$ drives both inference (reducing $\varepsilon$) and learning (improving predictions).

### 4. Theta-Gamma Coupling

The oscillatory clock implements:

$$\phi_\theta(t) = 2\pi f_\theta t, \quad \phi_\gamma(t) = 2\pi f_\gamma t$$
$$A_\gamma(t) = \frac{1}{2}(1 + \cos(\phi_\theta(t)))$$

Gamma amplitude is gated by theta phase, creating temporal windows for semantic binding.

### 5. Kuramoto Convergence Condition

For convergence on cyclic graphs, the coupling matrix $W$ must satisfy:

$$\gamma > \sigma_{max}(W) \cdot L$$

where $\sigma_{max}(W)$ is the largest singular value and $L \leq 1$ is the Lipschitz constant of the phase coupling. Cyclic messages are damped by $f(n) = e^{-\lambda n}$.

The synchronization order parameter:

$$r = \left|\frac{1}{N} \sum_{k=1}^{N} e^{i\phi_k}\right| \in [0, 1]$$

measures global phase coherence ($r \approx 1$ = convergence, $r \approx 0$ = divergence).

---

## Project Structure

```
snn_pc_hybrid/
├── config.py           # Global hyperparameters (dataclass)
├── core/
│   ├── neuron.py       # LIF neuron augmented by PC error
│   ├── synapse.py      # STDP synapse compatible with PC
│   ├── oscillator.py   # Theta/gamma oscillatory clock
│   └── encoding.py     # Spike encoding (rate + temporal/phase)
├── modules/
│   ├── wernicke.py     # Semantic module (encoder)
│   ├── broca.py        # Syntactic module (decoder)
│   └── arcuate.py      # Arcuate fasciculus (inter-module channel)
├── graph/
│   ├── pc_gnn.py       # PC on arbitrary graphs
│   ├── phase_sync.py   # Kuramoto convergence condition
│   └── message_passing.py  # Message passing with visit history
├── training/
│   ├── loss.py         # Variational free energy F
│   └── surrogate.py    # Surrogate gradient for spikes
├── experiments/
│   ├── toy_language.py     # SVO sequence experiment
│   └── convergence_test.py # Convergence on cyclic graphs
├── tests/
│   ├── test_neuron.py
│   ├── test_encoding.py
│   └── test_convergence.py
└── results/            # Auto-generated PNG figures
```

---

## Installation

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

**Python >= 3.10** required. For GPU acceleration, install the CUDA-compatible version of PyTorch from [pytorch.org](https://pytorch.org).

---

## Running the Experiments

### Toy Language Experiment

Tests the full Wernicke–Broca pipeline on subject-verb-object sequences:

```bash
cd snn_pc_hybrid
python experiments/toy_language.py
```

**Output figures** (saved in `results/`):
- `raster_plot.png` — Spike raster for Wernicke (blue) and Broca (red)
- `free_energy_kuramoto.png` — F(t) and r(t) over training
- `token_accuracy.png` — Generation accuracy over epochs

### Convergence Test on Cyclic Graphs

Compares convergence with vs without cyclic damping across graph topologies:

```bash
python experiments/convergence_test.py
```

**Output figures:**
- `convergence_comparison.png` — F(t) and r(t) per topology, with/without damping
- `graph_topologies.png` — Visual representation of tested graphs

### Running Tests

```bash
# From the snn_pc_hybrid/ directory
python -m pytest tests/ -v

# Or individually
python tests/test_neuron.py
python tests/test_encoding.py
python tests/test_convergence.py
```

### Interactive Notebook

```bash
jupyter notebook demo.ipynb
```

---

## Example Output (toy_language.py)

```
════════════════════════════════════════════════════════════
Expérience : SNN-PC sur séquences sujet-verbe-objet
Vocabulaire : 46 tokens | Dataset : 200 samples
Config : dim_W=128, dim_B=128
════════════════════════════════════════════════════════════
Époque   1/30 | F=2.8451 | Acc=0.062 | r=0.523
Époque   5/30 | F=1.9234 | Acc=0.187 | r=0.641
Époque  10/30 | F=1.2187 | Acc=0.312 | r=0.712
Époque  15/30 | F=0.8934 | Acc=0.437 | r=0.778
Époque  20/30 | F=0.6102 | Acc=0.562 | r=0.831
Époque  25/30 | F=0.4871 | Acc=0.625 | r=0.867
Époque  30/30 | F=0.3924 | Acc=0.687 | r=0.891

Figures sauvegardées dans : ../results
Précision finale : 0.687
Énergie libre finale : 0.3924
```

---

## Key Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `tau_m` | 20.0 ms | Membrane time constant |
| `tau_syn` | 5.0 ms | Synaptic time constant |
| `dt` | 0.1 ms | Simulation timestep |
| `theta_freq` | 6.0 Hz | Theta oscillation (global clock) |
| `gamma_freq` | 40.0 Hz | Gamma oscillation (local binding) |
| `eta_pc` | 0.1 | PC error weight in membrane dynamics |
| `n_inference_steps` | 20 | PC inference iterations per timestep |
| `A_plus` / `A_minus` | 0.01 / 0.012 | STDP potentiation / depression |
| `gamma_stability` | 0.1 | Kuramoto stability parameter γ |
| `cycle_damping_lambda` | 0.5 | Cyclic damping: f(n) = exp(-λn) |

---

## Theoretical Background

This framework bridges three computational neuroscience theories:

1. **Predictive Coding** (Rao & Ballard, 1999; Friston, 2005): The brain minimizes prediction errors through hierarchical message passing. Each level predicts the level below; errors propagate upward.

2. **Spiking Neural Networks**: Biological neurons communicate via discrete spike events. The LIF model captures the integrate-and-fire dynamics with refractory periods.

3. **Kuramoto Synchronization** (Kuramoto, 1984; Strogatz, 2000): Phase-coupled oscillators can synchronize. Here, the Kuramoto order parameter $r$ measures whether Broca and Wernicke have reached a coherent shared representation.

The **theta-gamma coupling** hypothesis (Lisman & Jensen, 2013) provides the temporal scaffolding: each theta cycle contains ~6-7 gamma sub-cycles, allowing sequential semantic items to be processed in parallel within a single working memory window.

---

## References

- Friston, K. (2005). A theory of cortical responses. *Phil. Trans. R. Soc. B*.
- Rao, R.P.N. & Ballard, D.H. (1999). Predictive coding in the visual cortex. *Nature Neuroscience*.
- Hagoort, P. (2005). On Broca, brain, and binding. *Trends in Cognitive Sciences*.
- Lisman, J. & Jensen, O. (2013). The theta-gamma neural code. *Neuron*.
- Neftci, E.O. et al. (2019). Surrogate Gradient Learning in SNNs. *IEEE Signal Processing Magazine*.
- Strogatz, S.H. (2000). From Kuramoto to Crawford. *Physica D*.
