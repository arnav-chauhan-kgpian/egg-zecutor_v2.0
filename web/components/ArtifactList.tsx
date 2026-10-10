'use client';

import { useEffect, useState } from 'react';
import { fetchArtifactBlob } from '@/lib/api';
import { formatBytes } from '@/lib/languages';
import type { ArtifactSummary, Execution } from '@/lib/types';

const PREVIEWABLE_IMAGE = /^image\/(png|jpeg|gif|webp|svg\+xml)$/;
const PREVIEWABLE_TEXT = /^(text\/|application\/(json|xml|javascript))/;

/**
 * Fetches an artifact and renders it inline where that makes sense.
 *
 * The download route is authenticated, so the blob is fetched with the bearer
 * token and turned into an object URL rather than being pointed at directly
 * from an <img src>.
 */
function ArtifactPreview({ execution, artifact }: { execution: Execution; artifact: ArtifactSummary }) {
  const [objectUrl, setObjectUrl] = useState<string | null>(null);
  const [text, setText] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const isImage = PREVIEWABLE_IMAGE.test(artifact.mimeType);
  const isText = PREVIEWABLE_TEXT.test(artifact.mimeType);

  useEffect(() => {
    if (!isImage && !isText) return;
    // Don't inline a huge payload — offer it as a download instead.
    if (artifact.sizeBytes > 2 * 1024 * 1024) return;

    let revoked = false;
    let url: string | null = null;

    fetchArtifactBlob(execution.id, artifact.id)
      .then(async (blob) => {
        if (revoked) return;
        if (isText) {
          setText(await blob.text());
          return;
        }
        url = URL.createObjectURL(blob);
        setObjectUrl(url);
      })
      .catch(() => setError('Could not load this artifact.'));

    return () => {
      revoked = true;
      if (url) URL.revokeObjectURL(url);
    };
  }, [execution.id, artifact.id, isImage, isText, artifact.sizeBytes]);

  if (error) return <p className="text-xs text-red-600 dark:text-red-400">{error}</p>;

  if (isImage && objectUrl) {
    return (
      // eslint-disable-next-line @next/next/no-img-element
      <img
        src={objectUrl}
        alt={artifact.name}
        className="max-h-96 w-auto rounded border border-slate-200 bg-white dark:border-slate-700"
      />
    );
  }

  if (isText && text !== null) {
    return (
      <pre className="max-h-64 overflow-auto rounded bg-slate-50 p-2 font-mono text-xs text-slate-800 dark:bg-slate-950 dark:text-slate-200">
        {text}
      </pre>
    );
  }

  return null;
}

async function download(execution: Execution, artifact: ArtifactSummary) {
  const blob = await fetchArtifactBlob(execution.id, artifact.id);
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = artifact.name;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}

export function ArtifactList({ execution }: { execution: Execution }) {
  if (execution.artifacts.length === 0) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-2 p-6 text-center">
        <p className="text-sm text-slate-400 dark:text-slate-500">No artifacts emitted.</p>
        <p className="max-w-md font-mono text-[11px] text-slate-400 dark:text-slate-600">
          ::artifact:&lt;name&gt;:&lt;mime&gt;:&lt;base64&gt;::
        </p>
        <p className="max-w-md text-xs text-slate-400 dark:text-slate-500">
          Print that marker on its own line to return a file. It is stripped from stdout.
        </p>
      </div>
    );
  }

  return (
    <div className="h-full space-y-3 overflow-auto p-3">
      {execution.artifacts.map((artifact) => (
        <div
          key={artifact.id}
          className="rounded-lg border border-slate-200 p-3 dark:border-slate-800"
        >
          <div className="mb-2 flex items-center justify-between gap-3">
            <div className="min-w-0">
              <p className="truncate text-sm font-medium text-slate-800 dark:text-slate-100">
                {artifact.name}
              </p>
              <p className="text-xs text-slate-500 dark:text-slate-400">
                {artifact.mimeType} · {formatBytes(artifact.sizeBytes)}
              </p>
            </div>
            <button
              type="button"
              onClick={() => void download(execution, artifact)}
              className="shrink-0 rounded-md border border-slate-300 px-2.5 py-1 text-xs font-medium text-slate-700 transition hover:bg-slate-100 dark:border-slate-700 dark:text-slate-200 dark:hover:bg-slate-800"
            >
              Download
            </button>
          </div>
          <ArtifactPreview execution={execution} artifact={artifact} />
        </div>
      ))}
    </div>
  );
}
