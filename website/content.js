window.PAPER_SITE = {
  meta: {
    title: "Learning Planning Budgets in Real-Time RL",
    description:
      "A flat paper-style project page for variable-delay real-time RL, AlphaZero-style planning, and state-dependent compute budgets.",
    ogImage: "assets/figures/option_timeline.gif",
  },
  paper: {
    title: "Learning Planning Budgets in Real-Time RL",
    authors: [
      { name: "Aneesh Muppidi", href: "https://aneeshers.github.io" },
      { name: "Firas Darwish", href: "https://firasdarwish.com" },
      { name: "Dylan Cope", href: "https://dylancope.com" },
      { name: "Joao F. Henriques", href: "https://joao.science" },
      { name: "Jakob Nicolaus Foerster", href: "https://www.jakobfoerster.com" },
    ],
    links: [
      { label: "GitHub", href: "https://github.com/Aneeshers/Real-time-RL", icon: "assets/icons/github.png" },
      { label: "Paper", href: "https://openreview.net/attachment?id=co1yOG9PHM&name=pdf", icon: "assets/icons/arxiv-square.svg" },
      { label: "Figures", href: "https://github.com/Aneeshers/Real-time-RL/tree/main/figures", icon: "assets/icons/pdf.png" },
      { label: "Code", href: "https://github.com/Aneeshers/Real-time-RL", icon: "assets/icons/python.png" },
    ],
    openingMedia: [
      {
        src: "assets/figures/option_timeline.gif",
        alt: "Budgeted option timeline",
        caption: "The opening idea is simple: choose how long to think, then act from the future state that arrives after that delay.",
      },
    ],
    abstract:
      "Real-time reinforcement learning is different from standard RL because the world keeps moving while the agent plans. We study variable-delay real-time RL, where a gate chooses how long to deliberate at each decision point on top of a frozen AlphaZero-style MCTS planner. The resulting policy spends more compute when the state is risky or constrained, outperforms fixed-budget and heuristic baselines across Pac-Man, real-time Tetris, Snake, Speed Hex, and Speed Go, and transfers to a two-GPU deployment without retraining.",
  },
  sections: [
    {
      id: "story",
      title: "The Story",
      blocks: [
        {
          type: "bullet",
          items: [
            "The paper studies what happens when an RL agent cannot assume that the environment waits for it. In ordinary RL, planning is free from the world’s point of view; in real-time RL, thinking longer changes the state you eventually act in.",
            "That turns planning itself into a control problem. Instead of using a fixed search budget everywhere, the agent learns when to react immediately and when to spend extra time on MCTS.",
            "The gate is intentionally lightweight. It sits on top of a frozen planner, reads the current state and planner features, and chooses a planning budget K before the MCTS action lands.",
          ],
        },
      ],
    },
    {
      id: "alphazero",
      title: "AlphaZero And Latency",
      blocks: [
        {
          type: "prose",
          paragraphs: [
            "AlphaZero is a policy-value network plus MCTS that improves the final action by running more rollouts at test time, and the budgeted-option formalism makes the delay explicit: a choice of K means K-1 filler actions followed by the planner’s action.",
          ],
        },
        {
          type: "callout",
          html: "AlphaZero-style MCTS buys better actions with more simulations, but the same increase also raises decision latency. In real-time settings, that delay matters because the state changes before the final action lands.",
        },
        {
          type: "figure",
          src: "assets/figures/mcts_tree.gif",
          alt: "Animated MCTS tree",
          caption: "More rollouts refine the search tree, but every extra rollout pushes the action further into the future.",
        },
        {
          type: "figure",
          src: "assets/figures/scaling.pdf",
          alt: "Co-scaling between planning quality and latency",
          caption: "Planning quality and latency co-scale with the number of MCTS simulations: the blue curve tracks return or win rate, while the red curve tracks inference latency.",
        },
      ],
    },
    {
      id: "method",
      title: "Variable-Delay Control",
      blocks: [
        {
          type: "prose",
          paragraphs: [
            "We frame the problem as a semi-Markov decision process whose holding time is selected by the gate. Each meta-action chooses a budget K, collects discounted reward while the world advances, and then resumes planning from the state reached after that delay.",
          ],
        },
        {
          type: "equation",
          tex: String.raw`R_t = \sum_{k=0}^{K_t - 1} \gamma^k r_{t+k}, \qquad V(s_t) = \mathbb{E}_{K_t \sim \pi_{\mathrm{gate}}}\!\left[R_t + \gamma^{K_t} V(s_{t+K_t})\right]`,
          note: "The discount changes with the selected holding time, so the meta-policy learns the cost of waiting directly.",
        },
        {
          type: "code",
          language: "python",
          code: String.raw`K_t = gate(s_t)

for _ in range(K_t - 1):
    a_t = pi_reflex(s_t)
    s_t = env.step(a_t)

a_t = mcts(s_t)`,
        },
      ],
    },
    {
      id: "results",
      title: "Results",
      blocks: [
        {
          type: "prose",
          paragraphs: [
            "Across the benchmark suite, the learned gate beats fixed-budget and heuristic baselines because it adapts compute to the state. The policy is not merely averaging over budgets; it reallocates them based on danger, density, reachability, and clock pressure.",
          ],
        },
        {
          type: "figure",
          src: "assets/figures/main_results_horizontal.pdf",
          alt: "Main results comparison",
          caption: "Headline result: adaptive gating outperforms fixed budgets and heuristics across Pac-Man, real-time Tetris, Snake, Speed Hex, and Speed Go.",
        },
        {
          type: "figure",
          src: "assets/figures/strategy_band.pdf",
          alt: "Strategy allocation figure",
          caption: "The gate reallocates compute across budgets instead of collapsing to a single fixed K.",
        },
        {
          type: "figure",
          src: "assets/figures/pacman_gate.gif",
          alt: "Pac-Man gate animation",
          caption: "Pac-Man makes the state dependence intuitive: as ghosts move closer, the gate shifts from deeper planning toward immediate reaction, and the selected budget follows the nearest-ghost distance.",
        },
        {
          type: "figure",
          src: "assets/figures/interpretability_combined_alt.pdf",
          alt: "Interpretability figure",
          caption: "Deeper planning appears when the state is dangerous, dense, or otherwise constrained.",
        },
      ],
    },
    {
      id: "deployment",
      title: "Deployment",
      blocks: [
        {
          type: "prose",
          paragraphs: [
            "The deployment story closes the loop. Training already simulates planning delay, so the learned gate transfers to a two-GPU setting where the environment keeps running while MCTS works on the second device.",
          ],
        },
        {
          type: "figure",
          src: "assets/figures/deployment_timeline.pdf",
          alt: "Deployment timeline",
          caption: "A K=4 meta-step spans four frames while MCTS runs in parallel on the second GPU.",
        },
        {
          type: "figure",
          src: "assets/figures/deployment.pdf",
          alt: "Deployment summary figure",
          caption: "Simulation-trained policies transfer to hardware deployment with small deadline misses only at the tightest frame budgets.",
        },
      ],
    },
  ],
  footer: {
    left: "Real-time RL project page built from the minimal paper template.",
    right:
      '<a href="https://github.com/Aneeshers/research-paper">This page is built from the minimal paper template.</a>',
  },
};
