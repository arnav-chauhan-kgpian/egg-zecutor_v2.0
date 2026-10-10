import { tool } from '@langchain/core/tools';
import { z } from 'zod';
import { errorMessage, truncate, type ToolContext } from './common';

export const webSearchInputSchema = z.object({
  query: z.string().min(1).max(1000),
});

export type WebSearchInput = z.infer<typeof webSearchInputSchema>;

interface SearchHit {
  title: string;
  url: string;
  snippet: string;
}

async function tavilySearch(query: string, apiKey: string): Promise<{ hits: SearchHit[]; answer: string | null }> {
  const res = await fetch('https://api.tavily.com/search', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ api_key: apiKey, query, max_results: 5, include_answer: true }),
    signal: AbortSignal.timeout(15_000),
  });
  if (!res.ok) throw new Error(`Tavily HTTP ${res.status}`);
  const body = (await res.json()) as {
    answer?: string;
    results?: Array<{ title?: string; url?: string; content?: string }>;
  };
  return {
    hits: (body.results ?? []).slice(0, 5).map((r) => ({
      title: r.title ?? 'Untitled',
      url: r.url ?? '',
      snippet: truncate(r.content ?? '', 500) ?? '',
    })),
    answer: body.answer ?? null,
  };
}

async function braveSearch(query: string, apiKey: string): Promise<{ hits: SearchHit[] }> {
  const res = await fetch(
    `https://api.search.brave.com/res/v1/web/search?q=${encodeURIComponent(query)}&count=5`,
    {
      headers: { Accept: 'application/json', 'X-Subscription-Token': apiKey },
      signal: AbortSignal.timeout(15_000),
    },
  );
  if (!res.ok) throw new Error(`Brave HTTP ${res.status}`);
  const body = (await res.json()) as {
    web?: { results?: Array<{ title?: string; url?: string; description?: string }> };
  };
  return {
    hits: (body.web?.results ?? []).slice(0, 5).map((r) => ({
      title: r.title ?? 'Untitled',
      url: r.url ?? '',
      snippet: truncate(r.description ?? '', 500) ?? '',
    })),
  };
}

/**
 * Web search with graceful degradation: Tavily > Brave > structured fallback.
 * Never throws — without a key the LLM gets an explicit fallback payload
 * telling it to answer from its own knowledge or ask the user.
 */
export function createWebSearchTool(_ctx: ToolContext = {}) {
  return tool(
    async (input: WebSearchInput): Promise<string> => {
      try {
        const tavilyKey = process.env.TAVILY_API_KEY;
        if (tavilyKey) {
          const { hits, answer } = await tavilySearch(input.query, tavilyKey);
          return JSON.stringify({ provider: 'tavily', query: input.query, answer, results: hits }, null, 2);
        }
        const braveKey = process.env.BRAVE_SEARCH_API_KEY ?? process.env.BRAVE_API_KEY;
        if (braveKey) {
          const { hits } = await braveSearch(input.query, braveKey);
          return JSON.stringify({ provider: 'brave', query: input.query, results: hits }, null, 2);
        }
        return JSON.stringify(
          {
            provider: 'none',
            query: input.query,
            results: [],
            message:
              'No search provider configured (set TAVILY_API_KEY or BRAVE_SEARCH_API_KEY). ' +
              'Answer from model knowledge, state the limitation, or ask the user to provide sources.',
          },
          null,
          2,
        );
      } catch (error) {
        return JSON.stringify(
          {
            provider: 'error',
            query: input.query,
            results: [],
            message: `Live search failed (${errorMessage(error)}). Answer from model knowledge and state the limitation.`,
          },
          null,
          2,
        );
      }
    },
    {
      name: 'web_search',
      description:
        'Search the web for current or external information. Returns titles, URLs and snippets. Works without a key (structured fallback) but live results need TAVILY_API_KEY or BRAVE_SEARCH_API_KEY.',
      schema: webSearchInputSchema,
    },
  );
}

/** Stateless — no session binding required. */
export const webSearchTool = createWebSearchTool();
