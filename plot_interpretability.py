def plot(far_state, close_state, sparse_state, dense_state, tetris_env,
         short_snake, long_snake):
    from jumanji.environments.packing.tetris.viewer import TetrisViewer

    viewer = TetrisViewer(
        num_rows=tetris_env.num_rows,
        num_cols=tetris_env.num_cols,
        render_mode="rgb_array",
    )

    # Wider, shorter canvas.
    fig = plt.figure(figsize=(17.0, 8.0))

    # Figure layout tuning.
    LEFT = 0.03
    RIGHT = 0.985
    TOP = 0.965
    BOTTOM = 0.06
    ROW_GAP = 0.055
    ROW_H = (TOP - BOTTOM - 2 * ROW_GAP) / 3.0

    # These are the important knobs.
    # Smaller BOARD_GAP makes the two state panels closer.
    BOARD_BLOCK_FRAC = 0.39
    MID_GAP_FRAC = 0.010
    PLOT_GAP_FRAC = 0.020
    PLOT1_FRAC = 0.40   # within the plot block
    BOARD_GAP_FRAC = 0.004

    def make_row_axes(y0, row_h):
        avail = RIGHT - LEFT

        board_block_w = BOARD_BLOCK_FRAC * avail
        mid_gap = MID_GAP_FRAC * avail
        plot_block_w = avail - board_block_w - mid_gap

        board_gap = BOARD_GAP_FRAC * avail
        board_w = (board_block_w - board_gap) / 2.0

        plot_gap = PLOT_GAP_FRAC * avail
        plot1_w = PLOT1_FRAC * plot_block_w
        plot2_w = plot_block_w - plot1_w - plot_gap

        x_board1 = LEFT
        x_board2 = x_board1 + board_w + board_gap
        x_plot1 = x_board2 + board_w + mid_gap
        x_plot2 = x_plot1 + plot1_w + plot_gap

        ax1 = fig.add_axes([x_board1, y0, board_w, row_h])
        ax2 = fig.add_axes([x_board2, y0, board_w, row_h])
        ax3 = fig.add_axes([x_plot1, y0, plot1_w, row_h])
        ax4 = fig.add_axes([x_plot2, y0, plot2_w, row_h])
        return ax1, ax2, ax3, ax4

    # Compute row bottoms from bottom to top.
    y_snake = BOTTOM
    y_tetris = y_snake + ROW_H + ROW_GAP
    y_pacman = y_tetris + ROW_H + ROW_GAP

    # ── Top Row: PacMan ───────────────────────────────────────────────────────
    ax_close, ax_far, ax_line, ax_pellet = make_row_axes(y_pacman, ROW_H)

    for ax, state, k_txt, title_txt, crop in [
        (ax_close, close_state, "Chosen K = 1", "Ghost close", PACMAN_CROP_LEFT),
        (ax_far,   far_state,   "Chosen K = 4", "Ghost far",   PACMAN_CROP_RIGHT),
    ]:
        ax.set_title(title_txt, fontsize=FONT_SIZE_TITLE, pad=6)
        ax.set_axis_off()
        ax.imshow(state_to_rgb(state, crop), interpolation="nearest")
        ax.text(
            0.5, -0.05, k_txt,
            transform=ax.transAxes,
            ha="center", va="top",
            fontsize=FONT_SIZE_K_BOX,
            color=C_BLACK,
        )

    ks = np.array([d[0] for d in GHOST_DIST_BY_K])
    means = np.array([d[1] for d in GHOST_DIST_BY_K])
    ses = np.array([d[2] for d in GHOST_DIST_BY_K])
    xs = np.arange(4)
    display_means = [v if not np.isnan(v) else 0.0 for v in means]
    display_ses = [v if not np.isnan(v) else 0.0 for v in ses]
    bar_alphas = [BAR_ALPHA if not np.isnan(m) else 0.25 for m in means]

    bars = ax_line.bar(xs, display_means, color=[K_COLORS[i] for i in range(4)], linewidth=0)
    for bar, alpha in zip(bars, bar_alphas):
        bar.set_alpha(alpha)

    valid_err = [i for i in range(4) if not np.isnan(means[i])]
    ax_line.errorbar(
        [xs[i] for i in valid_err],
        [display_means[i] for i in valid_err],
        yerr=[display_ses[i] for i in valid_err],
        fmt="none",
        ecolor=C_BLACK,
        elinewidth=ERR_LW,
        capsize=CAPSIZE,
        capthick=ERR_LW,
    )
    ax_line.set_xticks(xs)
    ax_line.set_xticklabels([f"K={k}" for k in ks], fontsize=FONT_SIZE_TICK)
    ax_line.set_ylabel("Nearest ghost distance", fontsize=FONT_SIZE_LABEL)
    ax_line.set_title("Threat proximity\nby chosen K", fontsize=FONT_SIZE_TITLE)
    ax_line.set_ylim(bottom=0)
    _spine_clean(ax_line)

    p_ks = np.array([d[0] for d in PELLET_FRAC_BY_K])
    p_means = np.array([d[1] for d in PELLET_FRAC_BY_K])
    p_ses = np.array([d[2] for d in PELLET_FRAC_BY_K])
    p_display = [v if not np.isnan(v) else 0.0 for v in p_means]
    p_disp_se = [v if not np.isnan(v) else 0.0 for v in p_ses]
    p_alphas = [BAR_ALPHA if not np.isnan(m) else 0.25 for m in p_means]

    bars_p = ax_pellet.bar(xs, p_display, color=[K_COLORS[i] for i in range(4)], linewidth=0)
    for bar, alpha in zip(bars_p, p_alphas):
        bar.set_alpha(alpha)

    p_valid = [i for i in range(4) if not np.isnan(p_means[i])]
    ax_pellet.errorbar(
        [xs[i] for i in p_valid],
        [p_display[i] for i in p_valid],
        yerr=[p_disp_se[i] for i in p_valid],
        fmt="none",
        ecolor=C_BLACK,
        elinewidth=ERR_LW,
        capsize=CAPSIZE,
        capthick=ERR_LW,
    )
    ax_pellet.set_xticks(xs)
    ax_pellet.set_xticklabels([f"K={k}" for k in p_ks], fontsize=FONT_SIZE_TICK)
    ax_pellet.set_ylabel("Pellet fraction", fontsize=FONT_SIZE_LABEL)
    ax_pellet.set_title("Pellet fraction\nby chosen K", fontsize=FONT_SIZE_TITLE)
    ax_pellet.set_ylim(0, 1.1)
    _spine_clean(ax_pellet)

    # ── Middle Row: Tetris ────────────────────────────────────────────────────
    ax_sparse, ax_dense, ax_fill, ax_piece = make_row_axes(y_tetris, ROW_H)

    for ax, state, k_txt, title_txt in [
        (ax_sparse, sparse_state, "Chosen K = 1", "Sparse board"),
        (ax_dense,  dense_state,  "Chosen K = 4", "Dense board"),
    ]:
        ax.set_title(title_txt, fontsize=FONT_SIZE_TITLE, pad=6)
        draw_tetris(state, ax, viewer)
        ax.text(
            0.5, -0.05, k_txt,
            transform=ax.transAxes,
            ha="center", va="top",
            fontsize=FONT_SIZE_K_BOX,
            color=C_BLACK,
        )

    k_vals = np.array([d[0] for d in BOARD_FILL_BY_K])
    fills = np.array([d[1] for d in BOARD_FILL_BY_K])
    f_ses = np.array([d[2] for d in BOARD_FILL_BY_K])
    ax_fill.fill_between(k_vals, fills - f_ses, fills + f_ses, color=LINE_COLOR, alpha=FILL_ALPHA)
    ax_fill.plot(
        k_vals, fills, "o-",
        color=LINE_COLOR,
        linewidth=LINE_LW,
        markersize=MARKER_SIZE,
        markeredgewidth=0,
    )
    ax_fill.errorbar(
        k_vals, fills, yerr=f_ses,
        fmt="none",
        ecolor=LINE_COLOR,
        elinewidth=ERR_LW,
        capsize=CAPSIZE,
        capthick=ERR_LW,
        alpha=0.6,
    )
    ax_fill.set_xticks(k_vals)
    ax_fill.set_xticklabels([f"K={int(k)}" for k in k_vals], fontsize=FONT_SIZE_TICK)
    ax_fill.set_ylabel("Board fill fraction", fontsize=FONT_SIZE_LABEL)
    ax_fill.set_title("Board density\nby chosen K", fontsize=FONT_SIZE_TITLE)
    ax_fill.set_ylim(bottom=0)
    _spine_clean(ax_fill)

    pieces = [d[0] for d in PIECE_MEAN_K]
    p_means = np.array([d[1] for d in PIECE_MEAN_K])
    p_ses = np.array([d[2] for d in PIECE_MEAN_K])
    y = np.arange(len(pieces))
    ax_piece.barh(y, p_means, height=0.55, color=BAR_COLOR, alpha=BAR_ALPHA, linewidth=0)
    ax_piece.errorbar(
        p_means, y, xerr=p_ses,
        fmt="none",
        ecolor=C_BLACK,
        elinewidth=ERR_LW,
        capsize=CAPSIZE,
        capthick=ERR_LW,
    )
    ax_piece.set_yticks(y)
    ax_piece.set_yticklabels(pieces, fontsize=FONT_SIZE_TICK)
    ax_piece.set_xlabel("Mean chosen K", fontsize=FONT_SIZE_LABEL)
    ax_piece.set_title("Piece complexity\nvs. deliberation", fontsize=FONT_SIZE_TITLE)
    _spine_clean(ax_piece)
    ax_piece.set_xlim(left=2.5, right=3.1)

    # ── Bottom Row: Snake ─────────────────────────────────────────────────────
    ax_short, ax_long, ax_reach, ax_posteat = make_row_axes(y_snake, ROW_H)

    for ax, state, k_txt, title_txt in [
        (ax_short, short_snake, "Chosen K = 2", "Short snake"),
        (ax_long,  long_snake,  "Chosen K = 3", "Long snake"),
    ]:
        draw_snake(state, ax)
        ax.set_title(title_txt, fontsize=FONT_SIZE_TITLE, pad=6)
        ax.text(
            0.5, -0.05, k_txt,
            transform=ax.transAxes,
            ha="center", va="top",
            fontsize=FONT_SIZE_K_BOX,
            color=C_BLACK,
        )

    k_idx = [0, 1, 2, 3]
    reach_means = [SNAKE_REACH_BY_K[k][1] for k in k_idx]
    reach_ses = [SNAKE_REACH_BY_K[k][2] for k in k_idx]
    bar_colors = [K_COLORS[k] for k in k_idx]
    bar_alphas = [BAR_ALPHA if not np.isnan(reach_means[k]) else 0.25 for k in k_idx]
    display_means = [v if not np.isnan(v) else 0.0 for v in reach_means]
    display_ses = [v if not np.isnan(v) else 0.0 for v in reach_ses]
    xs = np.arange(4)

    bars = ax_reach.bar(xs, display_means, color=bar_colors, linewidth=0)
    for bar, alpha in zip(bars, bar_alphas):
        bar.set_alpha(alpha)

    valid_err = [i for i in range(4) if not np.isnan(reach_means[i])]
    ax_reach.errorbar(
        [xs[i] for i in valid_err],
        [display_means[i] for i in valid_err],
        yerr=[display_ses[i] for i in valid_err],
        fmt="none",
        ecolor=C_BLACK,
        elinewidth=ERR_LW,
        capsize=CAPSIZE,
        capthick=ERR_LW,
    )
    ax_reach.set_xticks(xs)
    ax_reach.set_xticklabels(["K=1", "K=2", "K=3", "K=4"], fontsize=FONT_SIZE_TICK)
    ax_reach.set_ylabel("Reachable cells", fontsize=FONT_SIZE_LABEL)
    ax_reach.set_title("Reachability\nby chosen K", fontsize=FONT_SIZE_TITLE)
    ax_reach.set_ylim(0, 148)
    ax_reach.axhline(144, color=C_MID_GRAY, linestyle="--", linewidth=0.8)
    _spine_clean(ax_reach)

    x_k = np.arange(4)
    width = 0.36
    ax_posteat.bar(
        x_k - width / 2,
        SNAKE_K_OVERALL,
        width=width,
        label="Overall",
        linewidth=0,
        color=[K_COLORS[k] for k in range(4)],
        alpha=0.55,
    )
    ax_posteat.bar(
        x_k + width / 2,
        SNAKE_K_POST_EAT,
        width=width,
        label="Post-eating",
        linewidth=0,
        color=[K_COLORS[k] for k in range(4)],
        alpha=BAR_ALPHA,
    )
    ax_posteat.set_xticks(x_k)
    ax_posteat.set_xticklabels(["K=1", "K=2", "K=3", "K=4"], fontsize=FONT_SIZE_TICK)
    ax_posteat.set_ylabel("% of steps", fontsize=FONT_SIZE_LABEL)
    ax_posteat.set_title("Overall vs. \npost-eating", fontsize=FONT_SIZE_TITLE)
    ax_posteat.set_ylim(0, 95)
    ax_posteat.annotate(
        "4.2x",
        xy=(2 + width / 2, SNAKE_K_POST_EAT[2]),
        xytext=(2 + width / 2 + 0.05, SNAKE_K_POST_EAT[2] + 8),
        fontsize=FONT_SIZE_ANNOT,
        color=K_COLORS[2],
        arrowprops=dict(arrowstyle="-", color=K_COLORS[2], lw=1.2),
        ha="left",
    )
    ax_posteat.legend(fontsize=FONT_SIZE_ANNOT, frameon=False, loc="upper right")
    _spine_clean(ax_posteat)

    out = os.path.join(FIGS, "interpretability_combined.pdf")
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)