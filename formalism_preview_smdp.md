# Preview: SMDP / Options Formalism for Variable-Delay Real-Time RL

This is a preview of a cleaner formalization for the main paper section. It does **not** edit the TeX. The goal is to make the core object an SMDP over budget-parameterized options, while still contrasting with RTMDP as the fixed-delay special case.

## From RTMDP to Budgeted Options

Let the underlying environment be a standard MDP

\[
E = (\mathcal S, \mathcal A, P, r, \gamma).
\]

At primitive frame \(t\), the environment is in state \(s_t \in \mathcal S\), the agent emits an action \(a_t \in \mathcal A\), the environment transitions according to

\[
s_{t+1} \sim P(\cdot \mid s_t, a_t),
\]

and yields reward \(r(s_t, a_t)\).

The key issue in real-time settings is that deliberation takes time: if the agent spends several frames planning, the environment continues to evolve before the carefully planned action can be applied.

Ramstedt et al. formalize the simplest such case with the Real-Time Markov Decision Process (RTMDP), which assumes a fixed 1-step action delay. RTMDP augments the state with the action currently in execution,

\[
x_t = (s_t, a_t),
\]

and defines the transition

\[
P_{\mathrm{RT}}(x_{t+1} \mid x_t, \hat a_t)
=
P(s_{t+1} \mid s_t, a_t)\,\delta(a_{t+1} - \hat a_t).
\]

This is appropriate when delay is a fixed property of the system. In our setting, however, delay is a **decision**: at each decision point the agent chooses how long to deliberate, and the environment is actively controlled during that deliberation period by a fast reflex policy. That makes the natural abstraction a semi-Markov decision process over temporally extended actions.

## Primitive Controllers

We assume access to two controllers defined on the primitive MDP \(E\):

- A fast reflex policy \(\pi_{\mathrm{reflex}}(a \mid s)\), which can be evaluated within one frame.
- A planner that, given a state \(s\) and a budget \(K\), runs MCTS for \(K\) frames and returns a terminal action distribution. In committed-action environments, the tree explicitly rolls forward the same \(K-1\) committed steps that will be executed in the environment, so the planner's terminal action is selected for the future landing state rather than the current state.

We write this planner abstractly as

\[
\pi_{\mathrm{plan}}^{(K)}(a \mid s).
\]

This notation suppresses the internal tree-search computation: \(\pi_{\mathrm{plan}}^{(K)}\) should be understood as the action distribution induced by launching a \(K\)-frame planning routine from state \(s\).

## Budget-Parameterized Options

Let the meta-action space be a finite set of planning budgets

\[
\mathcal K = \{1,2,\dots,K_{\max}\}.
\]

Choosing \(K \in \mathcal K\) invokes an option \(o_K\). Each option has:

- Initiation set \(I_K = \mathcal S\).
- Deterministic termination after \(K\) primitive frames.
- An internal policy consisting of \(K-1\) committed reflex steps followed by one planner action.

Concretely, if option \(o_K\) is initiated in state \(s_t\), then for primitive steps \(j=0,\dots,K-2\),

\[
a_{t+j} \sim \pi_{\mathrm{reflex}}(\cdot \mid s_{t+j}),
\qquad
s_{t+j+1} \sim P(\cdot \mid s_{t+j}, a_{t+j}),
\]

and on the final step,

\[
a_{t+K-1} \sim \pi_{\mathrm{plan}}^{(K)}(\cdot \mid s_t),
\qquad
s_{t+K} \sim P(\cdot \mid s_{t+K-1}, a_{t+K-1}).
\]

The important modeling point is that \(\pi_{\mathrm{plan}}^{(K)}(\cdot \mid s_t)\) is not merely "an action for \(s_t\) held stale for \(K\) frames." Rather, it denotes the outcome of a planning computation launched at \(s_t\) whose internal tree explicitly simulates the \(K-1\) committed steps before choosing the terminal action.

This defines an option-induced transition kernel

\[
P_K(s' \mid s)
:=
\Pr(s_{t+K}=s' \mid s_t=s,\ o_K),
\]

and an option-level reward

\[
R_K(s)
:=
\mathbb E\!\left[
\sum_{j=0}^{K-1} \gamma^j r(s_{t+j}, a_{t+j})
\;\middle|\;
s_t=s,\ o_K
\right].
\]

## Meta-Level SMDP

The gating policy is then a meta-policy over options:

\[
\pi_{\mathrm{gate}}(K \mid s_t).
\]

At each meta-decision state \(s_t\), the agent samples a budget \(K\), executes option \(o_K\), receives the discounted cumulative reward over the next \(K\) primitive frames, and returns to the meta-level in state \(s_{t+K}\).

The induced control problem is therefore an SMDP with:

- Meta-state space \(\mathcal S\).
- Meta-action space \(\mathcal K\).
- Holding time \(\tau(K)=K\).
- Transition kernel \(P_K(s' \mid s)\).
- Reward \(R_K(s)\).

Its Bellman equation is

\[
V(s)
=
\sum_{K \in \mathcal K}
\pi_{\mathrm{gate}}(K \mid s)
\left[
R_K(s)
+
\gamma^K
\sum_{s'} P_K(s' \mid s) V(s')
\right].
\]

Equivalently,

\[
V(s_t)
=
\mathbb E_{K \sim \pi_{\mathrm{gate}}(\cdot \mid s_t)}
\left[
\sum_{j=0}^{K-1}\gamma^j r_{t+j}
+
\gamma^K V(s_{t+K})
\right].
\]

This is the cleanest expression of the problem the gating network is solving: not "which primitive action should I take now," but "which temporally extended computation-and-control routine should I invoke now."

## Why This Is Better Than a Generalized RTMDP

One could try to generalize RTMDP to \(K\)-step delays by augmenting the state with a queue of pending actions. That is awkward here for two reasons.

First, \(K\) is chosen by the agent, so delay is endogenous rather than fixed. A state-augmentation approach would need to represent a variable-length pipeline of pending actions.

Second, the intermediate behavior is not passive waiting. During the deliberation window the agent is still acting through \(\pi_{\mathrm{reflex}}\), and those actions are recomputed from the current state at each frame. The process is therefore better understood as a temporally extended action with internal structure than as a delayed primitive action.

In short:

- **RTMDP**: fixed exogenous delay; augment the state with the action currently in flight.
- **Our framework**: chosen delay; active control during the delay window; natural representation is an SMDP over options.

RTMDP remains useful as a contrast and as a limiting case in which delay is fixed and the intermediate control structure is removed. But once \(K\) varies and the intermediate frames are actively controlled, the options/SMDP view is the more faithful formalization.

## Committed-Action Environments

In committed-action environments such as Pac-Man, Snake, and real-time Tetris, option \(o_K\) has the following operational interpretation:

1. At state \(s_t\), the gate chooses \(K\).
2. The planner begins a \(K\)-frame MCTS computation.
3. For the next \(K-1\) frames, the agent executes committed actions from \(\pi_{\mathrm{reflex}}\).
4. The MCTS tree rolls forward those same \(K-1\) committed steps internally.
5. On the final frame, the planner's chosen terminal action is applied.

This makes the option semantics match the real-time deployment semantics: the cost of deeper planning is exactly the additional environmental progression incurred before the terminal planner action lands.

## Clock Environments as a Special Case

Clock environments fit the same formalism with a degenerate committed controller.

There, the board state does not evolve while the agent thinks; only the clock changes. The committed steps are therefore no-ops on the board, and choosing \(K\) means "consume \(K\) units of time and then apply the planner action." The same meta-policy \(\pi_{\mathrm{gate}}(K \mid s)\) still defines an SMDP over options, but the option's internal primitive actions are time-consuming no-ops rather than reflex controls.

## Possible Positioning in the Paper

A concise way to present the contrast would be:

> RTMDP formalizes real-time control with a fixed one-step action delay by augmenting the state with the action currently in execution. Our setting differs in two ways: the delay is chosen by the agent, and the environment remains under active control during the delay window through a fast reflex policy. We therefore model each budget choice as a temporally extended option and the resulting problem as a semi-Markov decision process over budget-parameterized options.

If you want to sharpen it even further, the one-line identity of the framework is:

> A gating policy chooses among budget-parameterized options whose internal policy executes \(K-1\) reflex steps and terminates with a planner action selected for the future landing state.
