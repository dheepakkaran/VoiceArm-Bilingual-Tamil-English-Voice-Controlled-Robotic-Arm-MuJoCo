# docs/

The project's showcase page, served by GitHub Pages from this directory.

`index.html` is the whole page and `media/` holds everything it embeds. This is
the single source: adding a figure means dropping a file in `media/` and
referencing it, with no build step and nothing to keep in sync elsewhere.

The page exists because GitHub renders a relative `.mp4` in a README as a link
rather than a player, so there was nowhere the execution videos actually played.

Enable it under Settings, Pages, Deploy from a branch, `main` / `/docs`.
