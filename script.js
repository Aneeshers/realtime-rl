const reveals = document.querySelectorAll("[data-reveal]");

const revealObserver = new IntersectionObserver(
  (entries) => {
    for (const entry of entries) {
      if (entry.isIntersecting) {
        entry.target.classList.add("is-visible");
        revealObserver.unobserve(entry.target);
      }
    }
  },
  {
    threshold: 0.12,
  }
);

for (const node of reveals) {
  revealObserver.observe(node);
}

const lightbox = document.getElementById("lightbox");
const lightboxImage = document.getElementById("lightbox-image");
const lightboxCaption = document.getElementById("lightbox-caption");
const lightboxClose = document.getElementById("lightbox-close");
const figures = document.querySelectorAll(".figure-panel img, .hero-media img");

function closeLightbox() {
  lightbox.hidden = true;
  lightboxImage.src = "";
  lightboxImage.alt = "";
  lightboxCaption.textContent = "";
  document.body.style.overflow = "";
}

for (const image of figures) {
  image.addEventListener("click", () => {
    const caption = image.closest("figure")?.querySelector("figcaption")?.textContent ?? "";
    lightboxImage.src = image.src;
    lightboxImage.alt = image.alt;
    lightboxCaption.textContent = caption.trim();
    lightbox.hidden = false;
    document.body.style.overflow = "hidden";
  });
}

lightboxClose.addEventListener("click", closeLightbox);

lightbox.addEventListener("click", (event) => {
  if (event.target === lightbox) {
    closeLightbox();
  }
});

window.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !lightbox.hidden) {
    closeLightbox();
  }
});
