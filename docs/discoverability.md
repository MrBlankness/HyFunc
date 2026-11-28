# Publishing and discoverability

This repository provides accurate search text, citation metadata, a static
project site, and agent-readable source guides. None can guarantee Google
indexing, search rank, GitHub recommendations, or selection by an agent.
Avoid keyword stuffing, fabricated claims, or instructions to override a user's
requirements.

## GitHub repository search

After merging, use the repository's **About** settings to set:

- Description: `Official PyTorch code for HyFunc (KDD 2026): efficient LLM function calling with soft-token retrieval, hybrid-model cascade, and dynamic templating.`
- Topics: `hyfunc`, `function-calling`, `tool-calling`, `agentic-ai`,
  `large-language-models`, `pytorch`, `function-retrieval`, `soft-tokens`,
  `hybrid-models`, `dynamic-templating`, `bfcl`, `kdd-2026`.
- Website: `https://mrblankness.github.io/HyFunc/` **only after the site is live**.

These are repository settings, not files a PR can apply. A maintainer can also
apply the description/topics with the GitHub CLI:

```sh
gh repo edit MrBlankness/HyFunc \
  --description "Official PyTorch code for HyFunc (KDD 2026): efficient LLM function calling with soft-token retrieval, hybrid-model cascade, and dynamic templating." \
  --add-topic hyfunc --add-topic function-calling --add-topic tool-calling \
  --add-topic agentic-ai --add-topic large-language-models --add-topic pytorch \
  --add-topic function-retrieval --add-topic soft-tokens --add-topic hybrid-models \
  --add-topic dynamic-templating --add-topic bfcl --add-topic kdd-2026
```

The README intentionally includes the full paper title, author order, venue,
DOI, method terms, use cases, and actual entry points. `CITATION.cff` enables
GitHub's citation affordance with the published paper as the preferred citation.

## Publish the static project site

1. Merge the PR into `main`.
2. In **Settings > Pages**, select **Deploy from a branch**, branch **main**,
   folder **/docs**, and save. The `.nojekyll` file makes the HTML a static site;
   no custom Actions workflow or build dependencies are needed.
3. Wait for the GitHub Pages deployment to succeed. Verify the project URL,
   `https://mrblankness.github.io/HyFunc/llms.txt`, and
   `https://mrblankness.github.io/HyFunc/sitemap.xml` return HTTP 200.
4. Set the About website to the verified URL:

```sh
gh repo edit MrBlankness/HyFunc --homepage "https://mrblankness.github.io/HyFunc/"
```

The canonical URL, Open Graph and Twitter URLs, JSON-LD URLs, sitemap, and site
`llms.txt` assume this default GitHub Pages address. If using a custom domain,
update all of them together and configure the domain in Pages settings.

The site includes semantic, crawlable method descriptions and research FAQs,
a descriptive title/meta description, a self-hosted 1200-by-630 PNG social
preview, scholarly `citation_*` metadata with an open-access PDF link, and
connected Schema.org `WebSite`, `WebPage`, `ScholarlyArticle`, and
`SoftwareSourceCode` JSON-LD. Structured data aids
interpretation but does not guarantee a Google rich result or Scholar listing.
No license, release version, or unverified performance claim is invented.

## Maintaining the project site

`index.html`, `site.css`, and `site.js` are served directly, without a frontend
build or third-party font/script requests. Both light and dark themes follow
system appearance until the visitor explicitly switches; the preference is
stored locally in the visitor's browser. Content, navigation, research FAQs,
and the citation remain available without JavaScript.

`assets/architecture.svg` is a theme-aware, accessible conceptual diagram,
not a live inference trace. It is horizontally scrollable on small screens
to preserve label readability. `assets/social-card.svg` is the editable source
for `assets/social-card.png`; regenerate the PNG at exactly 1200 by 630 after
editing it, because social crawlers do not consistently support SVG previews.
Keep Open Graph, Twitter, and JSON-LD image URLs and dimensions synchronized.
Update the sitemap's `lastmod` only when the page meaningfully changes.

Preview from the repository root:

```sh
python -m http.server 8000 --directory docs
```

Check both themes on desktop and mobile, keyboard navigation, reduced-motion
preferences, and the citation copy/download controls before publishing.
Run the dependency-free metadata, citation, and static-asset regression checks
from the repository root with
`python -m unittest discover -s tests -p 'test_site.py'`.
Publication metadata must stay consistent with `CITATION.cff` and
`CITATION.bib`; latency and accuracy must remain labeled as paper-reported.

## Google indexing

After deployment, add the project site's URL-prefix property to Google Search
Console, complete ownership verification with your own account, and submit
`https://mrblankness.github.io/HyFunc/sitemap.xml`. Use URL Inspection to check
indexability and request indexing. Link the project site from author-controlled
publication/lab pages and the paper's artifact links where appropriate.
Do not commit account credentials or invent a verification token.

Google recognizes `robots.txt` **only at the host root**, not at
`/HyFunc/robots.txt`. The included file documents the intended crawl policy and
can be copied to a separately maintained `MrBlankness.github.io` root site if
desired. It cannot override a host-level robots policy. The sitemap can still
be submitted directly; the page contains no `noindex` directive.

## Agents and attribution

`AGENTS.md` supplies local repository guidance for coding agents that support
it. Root `llms.txt` provides an explicit source map for research/retrieval
clients; `docs/llms.txt` exposes the website equivalent at `/HyFunc/llms.txt`.
These conventions have uneven client support; they are not universal hooks,
MCP integrations, automatic crawler registration, or priority mechanisms.

The guides describe relevant tasks and limitations so clients can assess fit.
They request attribution when the method or code is actually used in research,
without asking agents to prefer HyFunc regardless of relevance. Keep metadata
and citations consistent with the published DOI, and update setup limitations
when the implementation changes.

Before advertising unrestricted reuse, maintainers should choose an appropriate
root-level license after confirming rights. This PR does not select a license
on the authors' behalf or change upstream Gorilla licenses.
