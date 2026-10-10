/**
 * Text embeddings for the MemoryArtifact vector store.
 *
 * Provider chain (first usable wins):
 *   1. HuggingFace Inference API (`sentence-transformers/all-MiniLM-L6-v2` by
 *      default, override with HF_EMBEDDING_MODEL; auth via HF_API_KEY/HF_TOKEN,
 *      optional but strongly recommended) — real semantic vectors.
 *   2. OpenAI `text-embedding-3-small` (1536 dims natively) when OPENAI_API_KEY
 *      is set.
 *   3. Deterministic hash-based pseudo-embedding so RAG/memory tools keep
 *      working offline — with clearly degraded recall.
 *
 * The pgvector column is `vector(1536)`, wider than most HF sentence models
 * (MiniLM = 384). HF vectors are fitted by truncation/zero-padding and then
 * L2-normalised; padding is consistent across rows so cosine ordering stays
 * meaningful. The fitted length is always EMBEDDING_DIM.
 */

export const EMBEDDING_DIM = 1536;

export type EmbeddingProvider = 'huggingface' | 'openai' | 'fallback';

function fallbackEmbed(text: string): number[] {
  const vec = new Array<number>(EMBEDDING_DIM).fill(0);
  // Word shingles + char trigrams hashed into buckets, then L2-normalised.
  const words = text.toLowerCase().split(/[^a-z0-9]+/).filter(Boolean);
  const tokens: string[] = [...words];
  const flat = text.toLowerCase().replace(/\s+/g, ' ');
  for (let i = 0; i + 3 <= flat.length && tokens.length < 4000; i += 1) {
    tokens.push(flat.slice(i, i + 3));
  }
  for (const token of tokens) {
    let h1 = 2166136261;
    let h2 = 16777619;
    for (let i = 0; i < token.length; i += 1) {
      const c = token.charCodeAt(i);
      h1 = Math.imul(h1 ^ c, 16777619);
      h2 = Math.imul(h2 ^ (c + i * 31), 2246822519);
    }
    const idx = Math.abs(h1) % EMBEDDING_DIM;
    const idx2 = Math.abs(h2) % EMBEDDING_DIM;
    vec[idx] += 1;
    vec[idx2] += 0.5;
  }
  return normalize(vec);
}

function normalize(vec: number[]): number[] {
  let norm = 0;
  for (const v of vec) norm += v * v;
  norm = Math.sqrt(norm) || 1;
  return vec.map((v) => v / norm);
}

/** Truncates or zero-pads any model output to EMBEDDING_DIM, then normalises. */
function fitToDim(vec: number[]): number[] {
  if (vec.length === EMBEDDING_DIM) return normalize(vec);
  const fitted =
    vec.length > EMBEDDING_DIM ? vec.slice(0, EMBEDDING_DIM) : [...vec, ...new Array(EMBEDDING_DIM - vec.length).fill(0)];
  return normalize(fitted);
}

function meanPool(matrix: number[][]): number[] {
  const dim = matrix[0].length;
  const pooled = new Array<number>(dim).fill(0);
  for (const row of matrix) {
    for (let i = 0; i < dim; i += 1) pooled[i] += row[i] ?? 0;
  }
  return pooled.map((v) => v / Math.max(1, matrix.length));
}

/**
 * Accepts every shape the HF Inference API returns for embedding models:
 * a pooled vector, token embeddings, or a batch of token embeddings.
 */
function toVector(payload: unknown): number[] {
  if (Array.isArray(payload) && payload.length > 0 && typeof payload[0] === 'number') {
    return payload as number[];
  }
  if (Array.isArray(payload) && Array.isArray(payload[0])) {
    const first = payload as unknown[];
    if (first.length > 0 && typeof first[0] === 'number') {
      return payload as number[];
    }
    // number[][][] batch → first sequence; number[][] → token matrix.
    const matrix = (typeof first[0] === 'object' ? (first[0] as number[][]) : (payload as number[][])).filter(
      (row) => Array.isArray(row) && row.length > 0 && row.every((v) => typeof v === 'number'),
    );
    if (matrix.length > 0) return meanPool(matrix);
  }
  throw new Error('Unrecognised embedding response shape');
}

function hfToken(): string | undefined {
  return process.env.HF_API_KEY ?? process.env.HF_TOKEN ?? undefined;
}

async function postEmbedding(endpoint: string, model: string, text: string, token?: string): Promise<number[]> {
  const res = await fetch(endpoint, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    body: JSON.stringify({ inputs: text.slice(0, 8000), options: { wait_for_model: true } }),
    signal: AbortSignal.timeout(60_000),
  });
  if (!res.ok) {
    const detail = await res.text().catch(() => '');
    throw new Error(`HuggingFace embeddings HTTP ${res.status} (${model}): ${detail.slice(0, 200)}`);
  }
  return toVector(await res.json());
}

async function huggingfaceEmbed(text: string): Promise<number[]> {
  const model = process.env.HF_EMBEDDING_MODEL ?? 'sentence-transformers/all-MiniLM-L6-v2';
  const token = hfToken();
  const encoded = encodeURIComponent(model);
  // Current router first, legacy endpoint as fallback (it still serves many models).
  try {
    return fitToDim(await postEmbedding(`https://router.huggingface.co/hf-inference/models/${encoded}`, model, text, token));
  } catch (first) {
    try {
      return fitToDim(await postEmbedding(`https://api-inference.huggingface.co/models/${encoded}`, model, text, token));
    } catch {
      throw first;
    }
  }
}

async function openaiEmbed(text: string, apiKey: string): Promise<number[]> {
  const model = process.env.OPENAI_EMBEDDING_MODEL ?? 'text-embedding-3-small';
  const res = await fetch('https://api.openai.com/v1/embeddings', {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${apiKey}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ model, input: text.slice(0, 8000) }),
    signal: AbortSignal.timeout(20_000),
  });
  if (!res.ok) throw new Error(`OpenAI embeddings HTTP ${res.status}`);
  const body = (await res.json()) as { data?: Array<{ embedding?: number[] }> };
  const embedding = body.data?.[0]?.embedding;
  if (!Array.isArray(embedding) || embedding.length === 0) {
    throw new Error('OpenAI embeddings returned no vector');
  }
  if (embedding.length !== EMBEDDING_DIM) {
    throw new Error(`Expected ${EMBEDDING_DIM} dims, got ${embedding.length}`);
  }
  return embedding;
}

export async function embedText(text: string): Promise<{ vector: number[]; provider: EmbeddingProvider }> {
  try {
    return { vector: await huggingfaceEmbed(text), provider: 'huggingface' };
  } catch {
    // Cold model, unknown model id, no network — try the next provider.
  }
  const openaiKey = process.env.OPENAI_API_KEY;
  if (openaiKey) {
    try {
      return { vector: await openaiEmbed(text, openaiKey), provider: 'openai' };
    } catch {
      // Fall through to the offline pseudo-embedding rather than failing the tool.
    }
  }
  return { vector: fallbackEmbed(text), provider: 'fallback' };
}

/** Renders a pgvector literal: `[0.1,0.2,...]`. */
export function toVectorLiteral(vector: number[]): string {
  return `[${vector.join(',')}]`;
}
