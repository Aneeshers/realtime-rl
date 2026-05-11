window.PAPER_SITE = {
  meta: {
    title: "Finding the Time to Think: Adaptive MCTS in Real-Time RL",
    description:
      "A flat paper-style project page for variable-delay real-time RL, AlphaZero-style planning, and state-dependent compute budgets.",
    ogImage: "assets/figures/mcts_tree.gif",
  },
  paper: {
    title: "Finding the Time to Think:<br />Adaptive MCTS in Real-Time RL",
    authors: [
      { name: "Aneesh Muppidi", href: "https://aneeshers.github.io" },
      { name: "Firas Darwish", href: "https://firasdarwish.com" },
      { name: "Dylan Cope", href: "https://dylancope.com" },
      { name: "Joao F. Henriques", href: "https://joao.science" },
      { name: "Jakob Nicolaus Foerster", href: "https://www.jakobfoerster.com" },
    ],
    links: [
      { label: "GitHub", href: "https://github.com/Aneeshers/Real-time-RL", icon: "assets/icons/github.png" },
      { label: "Paper", href: "https://github.com/Aneeshers/Real-time-RL/blob/main/neurips_2026.tex", icon: "assets/icons/arxiv-square.svg" },
      { label: "Figures", href: "https://github.com/Aneeshers/Real-time-RL/tree/main/figures", icon: "assets/icons/pdf.png" },
      { label: "Code", href: "https://github.com/Aneeshers/Real-time-RL", icon: "assets/icons/python.png" },
    ],
    openingMedia: [
      {
        src: "assets/figures/mcts_tree.gif",
        alt: "Animated MCTS tree",
        caption: "AlphaZero-style MCTS gets better as the search tree grows, but each additional rollout costs more latency before the move lands.",
      },
      {
        src: "assets/figures/option_timeline.gif",
        alt: "Budgeted option timeline",
        caption: "Variable-delay real-time RL turns planning budget into an option: spend K frames thinking, then act from the future state.",
      },
      {
        src: "assets/figures/pacman_gate.gif",
        alt: "Pac-Man gate animation",
        caption: "The gate learns when to react quickly and when to spend more compute in dangerous states.",
      },
    ],
    abstract:
      "Real-time reinforcement learning is different from standard RL because the world keeps moving while the agent plans. We study variable-delay real-time RL, where a gate chooses how long to deliberate at each decision point on top of a frozen AlphaZero-style MCTS planner. The resulting policy spends more compute when the state is risky or constrained, outperforms fixed-budget and heuristic baselines across Pac-Man, real-time Tetris, Snake, Speed Hex, and Speed Go, and transfers to a two-GPU deployment without retraining.",
  },
  highlight:
    "AlphaZero-style MCTS buys better actions with more simulations, but the same increase also raises decision latency. In real-time settings, that delay matters because the state changes before the final action lands.",
  sections: [
    {
      id: "overview",
      title: "Overview",
      blocks: [
        {
          type: "prose",
          paragraphs: [
            "The paper generalizes fixed-delay real-time MDPs to a variable-delay setting. Instead of hard-coding one planning delay, the agent chooses a budget K state by state and executes K-1 filler actions before the planned action lands.",
            "That choice is the core problem: some states reward deeper search, while others need an immediate response. The gate learns that tradeoff on top of a frozen planner, so the meta-decision stays cheap relative to MCTS itself.",
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
            "The opening gif shows the central pressure in AlphaZero-style planning: more MCTS rollouts sharpen the action estimate, but they also delay the move. In ordinary RL that delay is invisible because the environment waits; in real-time RL, it directly changes the state you eventually act in.",
          ],
        },
        {
          type: "figure",
          src: "assets/figures/scaling.pdf",
          alt: "Co-scaling between planning quality and latency",
          caption: "Planning quality and latency co-scale with the number of MCTS simulations. The blue curve tracks return or win rate, while the red curve tracks inference latency.",
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
      id: "method",
      title: "Variable-Delay Control",
      blocks: [
        {
          type: "prose",
          paragraphs: [
            "The method is framed as a semi-Markov decision process with holding time chosen by the gate. Each meta-action selects a duration K and the agent collects discounted reward over the K-frame option.",
          ],
        },
        {
          type: "equation",
          tex: String.raw`R_t = \sum_{k=0}^{K_t - 1} \gamma^k r_{t+k}, \qquad V(s_t) = \mathbb{E}_{K_t \sim \pi_{\mathrm{gate}}}\!\left[R_t + \gamma^{K_t} V(s_{t+K_t})\right]`,
          note: "The gate trains on per-meta-step advantages with the discount adjusted by the selected holding time.",
        },
        {
          type: "callout",
          html: "The point is not to penalize computation separately. Instead, delay is modeled through the environment dynamics and the holding time of the option itself.",
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
            "The gate beats fixed-budget policies and simple heuristics across all five environments, and the learned allocation is state-dependent rather than collapsing to a single budget.",
          ],
        },
        {
          type: "figureGrid",
          columns: 2,
          items: [
            {
              src: "assets/figures/main_results_horizontal.pdf",
              alt: "Main results comparison",
              caption: "Headline result: adaptive gating outperforms fixed budgets and heuristics across Pac-Man, real-time Tetris, Snake, Speed Hex, and Speed Go.",
            },
            {
              src: "assets/figures/strategy_band.pdf",
              alt: "Strategy allocation figure",
              caption: "The policy reallocates compute across budgets instead of sticking to one fixed K.",
            },
            {
              src: "assets/figures/interpretability_combined_alt.pdf",
              alt: "Interpretability figure",
              caption: "Deeper planning appears when the state is dangerous, dense, or otherwise constrained.",
            },
            {
              src: "assets/figures/deployment.pdf",
              alt: "Deployment summary figure",
              caption: "Simulation-trained policies transfer to a real two-GPU deployment with small latency misses on the tightest deadlines.",
            },
          ],
        },
      ],
    },
    {
      id: "deployment",
      title: "Two-GPU Deployment",
      blocks: [
        {
          type: "prose",
          paragraphs: [
            "The deployment setup splits the environment and MCTS across two GPUs. Planning happens asynchronously while the environment keeps moving, which is exactly the behavior the training setup was designed to simulate.",
          ],
        },
        {
          type: "figureGrid",
          columns: 2,
          items: [
            {
              src: "assets/figures/deployment_timeline.pdf",
              alt: "Deployment timeline",
              caption: "A K=4 meta-step spans four frames while MCTS runs in parallel on the second GPU.",
            },
            {
              src: "assets/figures/option_timeline.gif",
              alt: "Option timeline animation",
              caption: "Budgeted options provide the bridge between the planning model and the real-time execution trace.",
            },
          ],
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
