(() => {
  const systemTheme = window.matchMedia("(prefers-color-scheme: dark)");
  let themePreference = null;

  try {
    const savedTheme = localStorage.getItem("hyfunc-theme");
    if (savedTheme === "light" || savedTheme === "dark") {
      themePreference = savedTheme;
    } else if (savedTheme !== null) {
      console.warn("HyFunc: Invalid saved theme; following system appearance.");
    }
  } catch (error) {
    console.warn("HyFunc: Unable to read theme preference; following system appearance.", error);
  }

  function applyTheme(theme) {
    document.documentElement.dataset.theme = theme;
    const themeColor = document.querySelector('meta[name="theme-color"]');
    if (themeColor) {
      themeColor.content = theme === "dark" ? "#080f0c" : "#f5f7f1";
    }
    const toggle = document.getElementById("theme-toggle");
    if (toggle) {
      const nextTheme = theme === "dark" ? "light" : "dark";
      toggle.setAttribute("aria-label", `Switch to ${nextTheme} theme`);
      toggle.title = `Switch to ${nextTheme} theme`;
      toggle.querySelector(".theme-label").textContent = nextTheme === "dark" ? "Dark" : "Light";
    }
  }

  applyTheme(themePreference || (systemTheme.matches ? "dark" : "light"));
  systemTheme.addEventListener("change", (event) => {
    if (themePreference === null) {
      applyTheme(event.matches ? "dark" : "light");
    }
  });

  document.addEventListener("DOMContentLoaded", () => {
    const themeToggle = document.getElementById("theme-toggle");
    const themeStatus = document.getElementById("theme-status");
    themeToggle.hidden = false;
    applyTheme(document.documentElement.dataset.theme);
    themeToggle.addEventListener("click", () => {
      themePreference = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
      applyTheme(themePreference);
      try {
        localStorage.setItem("hyfunc-theme", themePreference);
        themeStatus.textContent = "";
      } catch (error) {
        console.warn("HyFunc: Unable to save theme preference.", error);
        themeStatus.textContent = "Theme changed, but this browser could not save your preference.";
      }
    });

    const menuToggle = document.getElementById("menu-toggle");
    const navigation = document.getElementById("site-nav");
    const mobileViewport = window.matchMedia("(max-width: 760px)");

    function setMenuOpen(open) {
      navigation.classList.toggle("is-open", open);
      menuToggle.setAttribute("aria-expanded", String(open));
      menuToggle.setAttribute("aria-label", open ? "Close navigation" : "Open navigation");
    }

    menuToggle.addEventListener("click", () => {
      setMenuOpen(menuToggle.getAttribute("aria-expanded") !== "true");
    });
    navigation.addEventListener("click", (event) => {
      if (event.target.closest("a")) {
        setMenuOpen(false);
      }
    });
    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && menuToggle.getAttribute("aria-expanded") === "true") {
        setMenuOpen(false);
        menuToggle.focus();
      }
    });
    mobileViewport.addEventListener("change", () => setMenuOpen(false));
    menuToggle.hidden = false;
    document.documentElement.classList.add("js");

    const copyButton = document.getElementById("copy-citation");
    const copyLabel = document.getElementById("copy-label");
    const copyStatus = document.getElementById("copy-status");
    const citation = document.getElementById("bibtex");
    let resetCopyLabel;
    copyButton.hidden = false;
    copyButton.addEventListener("click", async () => {
      window.clearTimeout(resetCopyLabel);
      copyButton.disabled = true;
      try {
        if (!navigator.clipboard?.writeText) {
          throw new Error("Clipboard access is unavailable.");
        }
        await navigator.clipboard.writeText(`${citation.textContent.trim()}\n`);
        copyLabel.textContent = "Copied";
        copyStatus.textContent = "BibTeX copied to clipboard.";
        resetCopyLabel = window.setTimeout(() => {
          copyLabel.textContent = "Copy BibTeX";
        }, 2500);
      } catch (error) {
        console.warn("HyFunc: Unable to copy the citation.", error);
        copyLabel.textContent = "Copy BibTeX";
        copyStatus.textContent = "Clipboard unavailable. Select the citation below or use Download .bib.";
        citation.closest("pre").focus();
        const selection = window.getSelection();
        if (selection) {
          const range = document.createRange();
          range.selectNodeContents(citation);
          selection.removeAllRanges();
          selection.addRange(range);
        }
      } finally {
        copyButton.disabled = false;
      }
    });
  });
})();
