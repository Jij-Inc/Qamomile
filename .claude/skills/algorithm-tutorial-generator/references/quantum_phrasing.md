# Quantum-computing phrasing reference

Read this before drafting or copy-editing a Qamomile tutorial. The audience already knows quantum computing's vocabulary, and AI-generated prose tends to drift into phrasings that read as unnatural or wrong to that audience.

## Table of contents

- [Standard verbs](#standard-verbs)
- [Standard nouns](#standard-nouns)
- [Phrasings to avoid](#phrasings-to-avoid)
- [Math and notation conventions](#math-and-notation-conventions)

## Standard verbs

- **Apply** a gate / operator (not "run", "execute", "send through", "use" a gate).
- **Measure** in a basis (not "read out"); the *result* is an *outcome* or a *bitstring*, not a "reading".
- **Prepare** a state (not "create", "make", "produce" a state for the initial preparation step).
- **Act on** a qubit / register (e.g. "$H$ acts on qubit 0"), not "modify" / "change".
- **Evolve** under a Hamiltonian (not "let the qubit evolve through"). Time evolution is "evolution under $H$".
- **Entangle** is a verb only between subsystems. Don't say "the state becomes entangled" without naming what it is entangled with.
- **Decohere** is intransitive ("the qubit decoheres"), not "the qubit gets decohered".
- A **circuit depth** is the longest critical path; "circuit length" is non-standard.

## Standard nouns

- **Qubit**, not "quantum bit" (except possibly once at first introduction). **Register** for a group of qubits.
- **Ancilla** (plural **ancillae** or **ancillas**) for helper qubits, with a sentence stating what role it plays.
- **Ansatz** (plural **ansätze** in formal writing, **ansatzes** acceptable). It is *a* parameterised state family, not "the ansatz function".
- **Hamiltonian**, **observable**, **expectation value**, **Pauli string**, **stabiliser/stabilizer**, **fidelity**, **state vector** — use exactly as the literature does. Don't paraphrase ("quantum energy operator", "average quantum value").
- **Shot** for a single sampled execution; **counts** for the dict of bitstring → frequency.
- **Bra**/**ket** notation $|\psi\rangle$ / $\langle\psi|$ for pure states; **density matrix** $\rho$ for mixed states. Do not write "the quantum state $\psi$" without the ket.

## Phrasings to avoid

These signal a non-native writer of the field. Re-read the surrounding paragraph and fix.

- "Quantum advantage" / "quantum supremacy" — only use when the *paper you are summarising* makes that specific claim. Never as flavour text.
- "Quantum parallelism" — imprecise and contested; describe what the algorithm actually does (interference, amplitude amplification, …) instead.
- "The qubit collapses" as a general statement — measurement projects the state onto an eigenspace of the measured observable; say *projection* or *post-measurement state* when precision matters.
- "Quantum probability" / "quantum randomness" — redundant. The probability of a measurement outcome is just a probability.
- "Quantum acceleration" — the standard term is **speedup** (polynomial, exponential, …).
- "Quantum gate operation" — pleonasm; "gate" suffices.
- "Solve in superposition" — superposition is a state, not an action.
- "Quantum noise" used as a catch-all — name the channel (depolarising, dephasing, amplitude damping, readout error, …) when it matters.
- Treating "Hamiltonian" as a synonym for "energy" — a Hamiltonian *is the operator*; its expectation value is the energy.
- Calling a controlled gate "a CNOT" when it isn't — `CX` / `CNOT` is specifically controlled-X. Use **controlled-$U$** for the general construction.

## Math and notation conventions

- Reduced Planck constant $\hbar$ is typically set to 1; if you keep it, say so once.
- Pauli operators: $X$, $Y$, $Z$ (not $\sigma_x$ / $\sigma_y$ / $\sigma_z$ unless the source paper uses them and switching would obscure the citation).
- Rotation gates include the standard $1/2$ factor in Qamomile: $\mathrm{RZ}(\theta) = e^{-i\theta Z / 2}$. The section guide (`references/section_guide.md` → "4. Implementation" → "Step 2: Circuit Definition") spells this out next to the rotation code; never silently drop the factor in `## Implementation` prose.
- Use $\ket{0}, \ket{1}, \ket{+}, \ket{-}$ in math, and `|0>`, `|1>` in code/print statements — don't mix.

If a sentence feels stilted or over-formal, read it aloud as if explaining at a whiteboard. The tutorials in `docs/en/algorithm/` are deliberately conversational at the math level and terse at the code level — match that register.
