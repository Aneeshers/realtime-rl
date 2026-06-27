window.PAPER_SITE = {
  meta: {
    title: "Finding the Time to Think in Real-Time RL",
    description:
      "Project page for variable-delay real-time RL: a lightweight gate learns how long to run AlphaZero-style MCTS at each decision, on top of a frozen planner.",
    ogImage: "assets/figures/main_results_horizontal.png",
  },
  paper: {
    title: "Finding the Time to Think in Real-Time RL",
    authors: [
      { name: "Aneesh Muppidi", href: "https://aneeshers.github.io", equal: true },
      { name: "Firas Darwish", href: "https://firasdarwish.com", equal: true },
      { name: "Dylan Cope", href: "https://dylancope.com" },
      { name: "Joao F. Henriques", href: "https://joao.science" },
      { name: "Jakob Nicolaus Foerster", href: "https://www.jakobfoerster.com" },
    ],
    authorsNote: "* indicates equal contribution",
    affiliations:
      '<a href="https://bold-lab.ai/">British Open-ended Learning and Discovery Lab (BOLD)</a>, University of Oxford,' +
      '<br>' +
      '<a href="https://www.robots.ox.ac.uk/~vgg/">Visual Geometry Group (VGG)</a>, University of Oxford,' +
      '<br>' +
      '<a href="https://www.stats.ox.ac.uk/">Department of Statistics</a>, University of Oxford',
    links: [
      { label: "Paper (PDF)", href: "assets/finding-the-time-to-think.pdf", icon: "assets/icons/pdf.png" },
      { label: "Code", href: "https://github.com/Aneeshers/realtime-rl-code", icon: "assets/icons/github.png" },
      { label: "Checkpoints", href: "https://huggingface.co/Aneesh19/realtime-rl-checkpoints", icon: "assets/icons/hf-mark.png" },
      { label: "Video", href: "assets/trailer_music.mp4", icon: "assets/icons/python.png" },
    ],
    openingMedia: [
      {
        src: "assets/trailer.mp4",
        alt: "Real-time RL explainer trailer",
        caption: "A short tour of the whole idea: standard RL lets you think for free, the real world never waits, and a learned gate decides how long to think at each step. (Sound on the Video link above.)",
      },
    ],
    abstract:
      "Real-time reinforcement learning is different from standard RL because the world keeps moving while the agent plans. We study variable-delay real-time RL, where a gate chooses how long to deliberate at each decision point on top of a frozen AlphaZero-style MCTS planner. The resulting policy spends more compute when the state is risky or constrained, outperforms fixed-budget and heuristic baselines across Pac-Man, real-time Tetris, Snake, Speed Hex, and Speed Go, and transfers to a two-GPU deployment without retraining.",
  },
  sections: [
    {
      id: "problem",
      title: "The Problem",
      blocks: [
        {
          type: "bullet",
          items: [
            "In ordinary RL the environment waits while the agent deliberates, so planning is free from the world&rsquo;s point of view. The animation below shows that idealized setting: Pac-Man pauses the world, imagines a few rollouts, picks one, and only then moves.",
            "Real-time RL removes that luxury &mdash; the world keeps moving while you think. Deliberate too long and a ghost reaches you before your plan is ready; react instantly and you act on a weak, unplanned policy.",
            "That turns &ldquo;how long to think&rdquo; into a control problem: the agent should react immediately in some states and spend real search time in others.",
          ],
        },
        {
          type: "figure",
          src: "assets/figures/rtrl_scene1.mp4",
          alt: "Standard RL: the world waits while you think",
          caption: "Standard RL. The board freezes, the planner imagines rollouts, the best one is chosen, and the agent acts &mdash; the environment never moved while it planned.",
        },
        {
          type: "figure",
          src: "assets/figures/rtrl_scene2.mp4",
          alt: "The real world never waits",
          caption: "Real-time RL. Think too slow and the ghost catches you; think too fast and the greedy move is weak. Neither fixed strategy works.",
        },
      ],
    },
    {
      id: "alphazero",
      title: "Planning Costs Time",
      blocks: [
        {
          type: "callout",
          tone: "yellow",
          html: "AlphaZero is a policy-value network plus MCTS that improves the final action by running more rollouts at test time. More simulations buy better actions &mdash; but the same increase raises decision latency, and in real-time settings that delay is paid as progress in the world before the action lands.",
        },
        {
          type: "figure",
          src: "assets/figures/mcts_tree_scaling.gif",
          alt: "Animated MCTS tree and scaling curves",
          caption:
            "More rollouts refine the search tree, but every extra rollout pushes the action further into the future. Planning quality and latency co-scale with the number of MCTS simulations.",
        },
      ],
    },
    {
      id: "method",
      title: "Learning When To Think",
      blocks: [
        {
          type: "prose",
          paragraphs: [
            "We train a lightweight gate on top of a frozen planner. It reads the current state and planner features and chooses a planning budget K before the MCTS action lands. We frame this as a semi-Markov decision process whose holding time is the chosen budget: each meta-action collects discounted reward while the world advances, then resumes planning from the state reached after that delay.",
          ],
        },
        {
          type: "figure",
          src: "assets/figures/rtrl_scene3.mp4",
          alt: "The adaptive gate",
          caption: "The gate oscillates: it thinks deeply when the ghosts are far and reacts instantly when one is close &mdash; deep, react, deep &mdash; spending compute only where it matters.",
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
      title: "Across Real-Time Games",
      blocks: [
        {
          type: "prose",
          paragraphs: [
            "We deploy the gate across five real-time games &mdash; Pac-Man, Snake, real-time Tetris, Speed Hex, and Speed Go. The learned gate beats fixed-budget and heuristic baselines everywhere because it adapts compute to the state: danger, board density, reachability, and clock pressure.",
          ],
        },
        {
          type: "figure",
          src: "assets/figures/rtrl_scene5.mp4",
          alt: "Results across four games",
          caption: "Four of the five environments, with the per-game bar chart: the green adaptive-gate bar tops every fixed thinking budget.",
        },
        {
          type: "figure",
          src: "assets/figures/main_results_horizontal.png",
          alt: "Main results comparison",
          caption: "Headline result: adaptive gating outperforms fixed budgets and heuristics across all five environments.",
        },
        {
          type: "figure",
          src: "assets/figures/strategy_band.png",
          alt: "Strategy allocation figure",
          caption: "The gate reallocates compute across budgets instead of collapsing to a single fixed K.",
        },
        {
          type: "figure",
          src: "assets/figures/pacman_gate.gif",
          alt: "Pac-Man gate animation",
          caption: "As ghosts move closer, the gate shifts from deeper planning toward immediate reaction, and the selected budget tracks the nearest-ghost distance.",
        },
      ],
    },
    {
      id: "deployment",
      title: "Real-Time Deployment",
      blocks: [
        {
          type: "prose",
          paragraphs: [
            "Training already simulates the planning delay, so the learned gate transfers to a true two-GPU setup with no retraining: one GPU runs the environment at a fixed frame rate while the second runs MCTS, and committed reflex actions keep the agent moving until the planned action arrives.",
          ],
        },
        {
          type: "figure",
          src: "assets/figures/rtrl_scene4.mp4",
          alt: "Two-GPU real-time deployment",
          caption: "One GPU holds the environment (it never pauses, 9 FPS); the other runs the MCTS planner. State and action packets cross between them while the execution timeline is drawn live.",
        },
        {
          type: "figure",
          src: "assets/figures/deployment.png",
          alt: "Deployment summary figure",
          caption: "Simulation-trained policies transfer to hardware deployment, with small deadline misses only at the tightest frame budgets.",
        },
      ],
    },
    {
      id: "acknowledgements",
      title: "Acknowledgements",
      blocks: [
        {
          type: "prose",
          paragraphs: [
            'We thank <a href="https://justinsvegliato.com">Justin Svegliato</a> for valuable feedback on our metareasoning definitions and framing, and <a href="https://scholar.google.com/citations?user=7TVJf1gAAAAJ&hl=en">Mattie Fellows</a> and <a href="https://uljad.com">Uljad Berdica</a> for helpful discussions and feedback on earlier drafts. A.&nbsp;Muppidi and F.&nbsp;Darwish are supported by the <a href="https://www.rhodeshouse.ox.ac.uk">Rhodes Scholarship</a> (Rhodes Trust). The authors declare no competing interests.',
          ],
        },
      ],
    },
  ],
  footer: {
    left: "Finding the Time to Think in Real-Time RL.",
    right:
      '<a href="https://github.com/Aneeshers/realtime-rl-code">Code</a> &middot; <a href="assets/finding-the-time-to-think.pdf">Paper (PDF)</a>',
  },
};
