# CodeArena — Frontend

Next.js (App Router) + Tailwind CSS interface for the competitive programming API in the parent
directory.

## Getting started

The Express API must be running first (see `../README.md`):

```bash
cd ..           # API project
npm run dev     # http://localhost:4000

cd web
npm install
cp .env.local.example .env.local   # NEXT_PUBLIC_API_URL
npm run dev                        # http://localhost:3000
```

| Script              | Purpose                        |
| ------------------- | ------------------------------ |
| `npm run dev`       | Dev server on port 3000        |
| `npm run build`     | Production build               |
| `npm start`         | Serve the production build     |
| `npm run typecheck` | `tsc --noEmit`                 |

### Environment

| Variable              | Default                 | Notes                                   |
| --------------------- | ----------------------- | --------------------------------------- |
| `NEXT_PUBLIC_API_URL` | `http://localhost:4000` | API base URL, no trailing slash or `/api` |

Used by both the server-side fetches (problem list/detail) and the browser (run, submit, auth).

## Pages

### `/problems`

Server-rendered list of every problem with difficulty badges and links into the workspace. Shows an
actionable message if the API is unreachable or the database hasn't been seeded.

### `/problems/[slug]`

Three-pane workspace, both dividers draggable (and keyboard-adjustable with the arrow keys):

- **Left** — title, difficulty badge, markdown description (constraints and explanation render from
  the markdown via `react-markdown` + `remark-gfm`, including GFM tables), sample input/output, and
  the visible sample test cases.
- **Right top** — Monaco editor with a language selector (Python 3 / C++ / JavaScript), a Reset
  button that restores the starter template, and a light/dark toggle that drives both Tailwind's
  `dark` class and Monaco's `vs`/`vs-dark` theme.
- **Right bottom** — output console with a **Test results** tab (verdict badge, `passed/total`, max
  runtime and memory, per-case rows expandable to show input / expected / your output / stderr) and a
  **Raw output** tab (stdout per visible case).

Action bar:

| Button       | Calls                       | Behaviour                                              |
| ------------ | --------------------------- | ------------------------------------------------------ |
| **Run Code** | `POST /api/submissions/run` | Visible sample cases only; nothing is saved            |
| **Submit**   | `POST /api/submissions`     | All cases including hidden ones; saves the verdict     |

Shortcuts: `Ctrl/⌘+Enter` runs, `Ctrl/⌘+Shift+Enter` submits.

## Notes

- **Submitting requires auth**, so there's a compact sign-in / register popover in the header. The
  JWT goes to `localStorage` and is attached by an axios request interceptor. Submit stays disabled
  until you sign in. The seeded account is `coder@example.com` / `Password123!`.
- **Hidden test cases stay hidden.** The API only sends pass/fail/runtime/memory for them, and the
  UI renders them as non-expandable rows labelled `hidden`.
- **Mock mode is called out.** When the API has no `JUDGE0_API_URL`, the console shows a banner
  saying the code was not actually executed.
- **Editor drafts persist** in `localStorage` per problem *and* language, so switching languages
  keeps each version.
- **Monaco loads from a CDN** (jsDelivr) via `@monaco-editor/react`'s default loader, so a first
  load without internet access will sit on "Loading editor…". Run/Submit still work — the code lives
  in React state. To work fully offline, self-host Monaco with `loader.config({ paths: { vs: … } })`.
- Descriptions that open with a heading repeating the problem title (the seeded ones do) have that
  heading stripped so the title isn't rendered twice.
- `npm audit` reports 2 advisories (1 low, 1 moderate) coming from `monaco-editor`'s bundled
  `dompurify`. There's no non-breaking fix available; both are in the editor dependency, not this
  code.
