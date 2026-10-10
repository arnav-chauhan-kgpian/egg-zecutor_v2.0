/**
 * Editor-side language metadata, keyed by Judge0 language id.
 *
 * The backend is the source of truth for which ids exist (GET
 * /executions/languages); this only supplies the Monaco grammar and a starter
 * script. Unknown ids fall back to plaintext rather than breaking the editor.
 */
interface EditorLanguage {
  /** Monaco language id. */
  monaco: string;
  starter: string;
}

const ARTIFACT_HINT_PY = `# Emit files back to the UI by printing an artifact marker:
#   ::artifact:<name>:<mime>:<base64>::`;

export const EDITOR_LANGUAGES: Record<number, EditorLanguage> = {
  71: {
    monaco: 'python',
    starter: `import base64, io, json, sys

${ARTIFACT_HINT_PY}

result = {"message": "hello from the execution engine", "n": 42}
print(json.dumps(result, indent=2))

# Example: return a JSON artifact
payload = base64.b64encode(json.dumps(result).encode()).decode()
print(f"::artifact:result.json:application/json:{payload}::")
`,
  },
  70: { monaco: 'python', starter: 'print "hello from Python 2"\n' },
  63: {
    monaco: 'javascript',
    starter: `const result = { message: "hello from the execution engine", n: 42 };
console.log(JSON.stringify(result, null, 2));

// Emit a file back to the UI:
const payload = Buffer.from(JSON.stringify(result)).toString("base64");
console.log(\`::artifact:result.json:application/json:\${payload}::\`);
`,
  },
  74: { monaco: 'typescript', starter: 'const x: number = 42;\nconsole.log(x);\n' },
  54: {
    monaco: 'cpp',
    starter: `#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cout << "hello from the execution engine" << endl;
    return 0;
}
`,
  },
  50: {
    monaco: 'c',
    starter: `#include <stdio.h>

int main(void) {
    printf("hello from the execution engine\\n");
    return 0;
}
`,
  },
  62: {
    monaco: 'java',
    starter: `public class Main {
    public static void main(String[] args) {
        System.out.println("hello from the execution engine");
    }
}
`,
  },
  60: {
    monaco: 'go',
    starter: `package main

import "fmt"

func main() {
    fmt.Println("hello from the execution engine")
}
`,
  },
  72: { monaco: 'ruby', starter: 'puts "hello from the execution engine"\n' },
  73: {
    monaco: 'rust',
    starter: `fn main() {
    println!("hello from the execution engine");
}
`,
  },
  80: { monaco: 'r', starter: 'cat("hello from the execution engine\\n")\n' },
  46: { monaco: 'shell', starter: 'echo "hello from the execution engine"\n' },
};

export function editorLanguage(languageId: number): EditorLanguage {
  return EDITOR_LANGUAGES[languageId] ?? { monaco: 'plaintext', starter: '' };
}

/** localStorage key for per-language draft code. */
export function draftKey(languageId: number) {
  return `hrc.draft.${languageId}`;
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export function formatDuration(ms: number | null): string {
  if (ms == null) return '—';
  if (ms < 1000) return `${ms} ms`;
  return `${(ms / 1000).toFixed(2)} s`;
}

export function formatMemory(kb: number | null): string {
  if (kb == null) return '—';
  if (kb < 1024) return `${kb} KB`;
  return `${(kb / 1024).toFixed(1)} MB`;
}
