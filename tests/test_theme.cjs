const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const source = fs.readFileSync(path.join(__dirname, "../docs/site.js"), "utf8");

function initializeTheme({ dark = false, saved = null, storageError = false } = {}) {
  const root = { dataset: {} };
  const themeColor = { content: "" };
  const warnings = [];
  let appearanceChanged;
  const context = {
    document: {
      documentElement: root,
      querySelector: () => themeColor,
      getElementById: () => null,
      addEventListener: () => {},
    },
    window: {
      matchMedia: () => ({
        matches: dark,
        addEventListener: (_, listener) => { appearanceChanged = listener; },
      }),
    },
    localStorage: {
      getItem: () => {
        if (storageError) throw new Error("Storage access denied");
        return saved;
      },
    },
    console: { warn: (...message) => warnings.push(message) },
  };
  vm.runInNewContext(source, context);
  return { root, themeColor, warnings, changeAppearance: appearanceChanged };
}

test("appearance follows system changes until a preference is saved", () => {
  const state = initializeTheme();
  assert.equal(state.root.dataset.theme, "light");
  state.changeAppearance({ matches: true });
  assert.equal(state.root.dataset.theme, "dark");
  assert.equal(state.themeColor.content, "#080f0c");
  state.changeAppearance({ matches: false });
  assert.equal(state.root.dataset.theme, "light");
  assert.equal(state.themeColor.content, "#f5f7f1");
  assert.equal(initializeTheme({ dark: true }).root.dataset.theme, "dark");
});

test("both saved themes take precedence over system appearance", () => {
  for (const saved of ["light", "dark"]) {
    const state = initializeTheme({ dark: saved === "light", saved });
    assert.equal(state.root.dataset.theme, saved);
    state.changeAppearance({ matches: saved !== "dark" });
    assert.equal(state.root.dataset.theme, saved);
    assert.equal(state.warnings.length, 0);
  }
});

test("an invalid stored preference is reported and falls back to the system", () => {
  const state = initializeTheme({ dark: true, saved: "invalid" });
  assert.equal(state.root.dataset.theme, "dark");
  assert.equal(state.warnings.length, 1);
  state.changeAppearance({ matches: false });
  assert.equal(state.root.dataset.theme, "light");
});

test("unavailable storage is reported without breaking initial appearance", () => {
  const state = initializeTheme({ dark: true, storageError: true });
  assert.equal(state.root.dataset.theme, "dark");
  assert.equal(state.warnings.length, 1);
  state.changeAppearance({ matches: false });
  assert.equal(state.root.dataset.theme, "light");
});
