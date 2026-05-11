(function renderPaperSite() {
  const data = window.PAPER_SITE;
  if (!data) {
    throw new Error("PAPER_SITE is not defined.");
  }

  const $ = (id) => document.getElementById(id);

  const hero = $("hero");
  const sectionsRoot = $("sections");
  const footer = $("footer");

  function updateMetadata() {
    document.title = data.meta.title;
    const description = document.querySelector('meta[name="description"]');
    const ogTitle = document.querySelector('meta[property="og:title"]');
    const ogDescription = document.querySelector('meta[property="og:description"]');
    const ogImage = document.querySelector('meta[property="og:image"]');

    if (description) description.setAttribute("content", data.meta.description);
    if (ogTitle) ogTitle.setAttribute("content", data.meta.title);
    if (ogDescription) ogDescription.setAttribute("content", data.meta.description);
    if (ogImage) ogImage.setAttribute("content", data.meta.ogImage);
  }

  function authorMarkup(author) {
    return author.href
      ? `<a href="${author.href}">${author.name}</a>`
      : author.name;
  }

  function escapeHTML(text) {
    return text
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function highlightPython(code) {
    const keywords = new Set([
      "from",
      "import",
      "as",
      "def",
      "class",
      "return",
      "for",
      "in",
      "if",
      "else",
      "elif",
      "while",
      "with",
      "try",
      "except",
      "finally",
      "raise",
      "pass",
      "break",
      "continue",
      "True",
      "False",
      "None",
      "and",
      "or",
      "not",
    ]);
    const tokenPattern =
      /('(?:[^'\\]|\\.)*'|"(?:[^"\\]|\\.)*"|\b\d+(?:\.\d+)?\b|\b[A-Za-z_][A-Za-z0-9_]*\b|\s+|.)/g;

    return code
      .split("\n")
      .map((line) => {
        const commentIndex = line.indexOf("#");
        const body = commentIndex >= 0 ? line.slice(0, commentIndex) : line;
        const comment = commentIndex >= 0 ? line.slice(commentIndex) : "";

        const tokens = body.match(tokenPattern) || [];
        let html = "";
        for (let i = 0; i < tokens.length; i += 1) {
          const token = tokens[i];
          if (/^\s+$/.test(token)) {
            html += token;
          } else if (/^'(?:[^'\\]|\\.)*'$|^"(?:[^"\\]|\\.)*"$/.test(token)) {
            html += `<span class="code-string">${escapeHTML(token)}</span>`;
          } else if (/^\d+(?:\.\d+)?$/.test(token)) {
            html += `<span class="code-number">${token}</span>`;
          } else if (/^[A-Za-z_][A-Za-z0-9_]*$/.test(token) && keywords.has(token)) {
            html += `<span class="code-keyword">${token}</span>`;
          } else if (
            /^[A-Za-z_][A-Za-z0-9_]*$/.test(token) &&
            tokens[i + 1] === "("
          ) {
            html += `<span class="code-func">${token}</span>`;
          } else {
            html += escapeHTML(token);
          }
        }

        if (comment) {
          html += `<span class="code-comment">${escapeHTML(comment)}</span>`;
        }

        return html || "&nbsp;";
      })
      .join("\n");
  }

  function linkMarkup(link) {
    return `
      <a class="paper-link" href="${link.href}" aria-label="${link.label}" title="${link.label}">
        <img class="paper-link-icon" src="${link.icon}" alt="" aria-hidden="true" />
        <span>${link.label}</span>
      </a>
    `;
  }

  function renderHero() {
    const openingMedia = data.paper.openingMedia
      ? `
        <div class="hero-media-grid">
          ${data.paper.openingMedia
            .map(
              (item) => `
                <figure class="hero-media-item">
                  <img src="${item.src}" alt="${item.alt}" />
                  <figcaption>${item.caption}</figcaption>
                </figure>
              `
            )
            .join("")}
        </div>
      `
      : "";

    hero.innerHTML = `
      <div class="hero-copy">
        <h1>${data.paper.title}</h1>
        <p class="authors-line">${data.paper.authors.map(authorMarkup).join(", ")}</p>
        <div class="hero-links">
          ${data.paper.links.map(linkMarkup).join("")}
        </div>
        ${data.paper.notice ? `<p class="notice">${data.paper.notice}</p>` : ""}
        ${data.paper.noticeSecondary ? `<p class="notice secondary">${data.paper.noticeSecondary}</p>` : ""}
        <div class="abstract-box">
          <p class="abstract">${data.paper.abstract}</p>
        </div>
        ${openingMedia}
      </div>
    `;
  }

  function renderHighlight() {
    if (!data.highlight) {
      return "";
    }

    return `<p class="highlight">${data.highlight}</p>`;
  }

  function renderBlock(block) {
    if (block.type === "prose") {
      return `
        <div class="block prose-block">
          ${block.paragraphs.map((p) => `<p>${p}</p>`).join("")}
        </div>
      `;
    }

    if (block.type === "bullet") {
      return `
        <div class="block bullet-block">
          <ul class="note-list">
            ${block.items
              .map(
                (item, index) => `<li class="fade-in" style="--fade-delay:${index * 110}ms">${item}</li>`
              )
              .join("")}
          </ul>
        </div>
      `;
    }

    if (block.type === "callout") {
      return `<div class="block callout">${block.html}</div>`;
    }

    if (block.type === "code") {
      const language = block.language || "python";
      const highlighted = language === "python" ? highlightPython(block.code) : escapeHTML(block.code);
      return `
        <pre class="block code-block language-${language}"><code>${highlighted}</code></pre>
      `;
    }

    if (block.type === "equation") {
      return `
        <div class="block equation-block">
          <div class="equation-display">$$${block.tex}$$</div>
          ${block.note ? `<p class="equation-note">${block.note}</p>` : ""}
        </div>
      `;
    }

    if (block.type === "figure") {
      return `
        <figure class="block figure-block">
          <img src="${block.src}" alt="${block.alt}" />
          <figcaption>${block.caption}</figcaption>
        </figure>
      `;
    }

    if (block.type === "figureGrid") {
      return `
        <div class="block figure-grid columns-${block.columns || 2}">
          ${block.items
            .map(
              (item) => `
                <figure class="figure-block">
                  <img src="${item.src}" alt="${item.alt}" />
                  <figcaption>${item.caption}</figcaption>
                </figure>
              `
            )
            .join("")}
        </div>
      `;
    }

    return "";
  }

  function renderSections() {
    sectionsRoot.innerHTML = `
      <div class="highlight-wrap">${renderHighlight()}</div>
      ${data.sections
        .map(
          (section) => `
            <section class="paper-section" id="${section.id}">
              <div class="section-heading">
                <h2>${section.title}</h2>
              </div>
              <div class="section-body">
                ${section.blocks.map(renderBlock).join("")}
              </div>
            </section>
          `
        )
        .join("")}
    `;
  }

  function renderFooter() {
    footer.innerHTML = `
      <p>${data.footer.left}</p>
      <p>${data.footer.right}</p>
    `;
  }

  function renderMath() {
    if (typeof renderMathInElement !== "function") {
      return;
    }

    renderMathInElement(document.body, {
      delimiters: [
        { left: "$$", right: "$$", display: true },
        { left: "$", right: "$", display: false },
        { left: "\\(", right: "\\)", display: false },
        { left: "\\[", right: "\\]", display: true },
      ],
      throwOnError: false,
    });
  }

  function setupFadeIns() {
    const items = document.querySelectorAll(".fade-in");
    if (!items.length || typeof IntersectionObserver === "undefined") {
      items.forEach((item) => item.classList.add("visible"));
      return;
    }

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add("visible");
            observer.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.15 }
    );

    items.forEach((item) => observer.observe(item));
  }

  updateMetadata();
  renderHero();
  renderSections();
  renderFooter();
  renderMath();
  setupFadeIns();
})();
