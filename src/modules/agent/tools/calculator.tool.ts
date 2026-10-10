import { tool } from '@langchain/core/tools';
import { z } from 'zod';
import { errorMessage, type ToolContext } from './common';

export const calculatorInputSchema = z.object({
  expression: z.string().min(1).max(2000),
});

export type CalculatorInput = z.infer<typeof calculatorInputSchema>;

/**
 * Deterministic arithmetic — no eval(), no Function(), no sandbox round-trip.
 * Recursive-descent parser: numbers, + - * / % ^ (right-assoc), parentheses,
 * unary +/-, functions sqrt sin cos tan asin acos atan log ln exp abs floor
 * ceil round, constants pi e. Angles in radians.
 */

type Token =
  | { kind: 'num'; value: number }
  | { kind: 'op'; value: string }
  | { kind: 'name'; value: string }
  | { kind: 'lparen' }
  | { kind: 'rparen' }
  | { kind: 'comma' };

const FUNCTIONS: Record<string, (...args: number[]) => number> = {
  sqrt: (x) => {
    if (x < 0) throw new Error('sqrt of negative number');
    return Math.sqrt(x);
  },
  sin: (x) => Math.sin(x),
  cos: (x) => Math.cos(x),
  tan: (x) => Math.tan(x),
  asin: (x) => Math.asin(x),
  acos: (x) => Math.acos(x),
  atan: (x) => Math.atan(x),
  log: (x) => {
    if (x <= 0) throw new Error('log of non-positive number');
    return Math.log10(x);
  },
  ln: (x) => {
    if (x <= 0) throw new Error('ln of non-positive number');
    return Math.log(x);
  },
  exp: (x) => Math.exp(x),
  abs: (x) => Math.abs(x),
  floor: (x) => Math.floor(x),
  ceil: (x) => Math.ceil(x),
  round: (x) => Math.round(x),
};

const CONSTANTS: Record<string, number> = { pi: Math.PI, e: Math.E };

function tokenize(expression: string): Token[] {
  const tokens: Token[] = [];
  let i = 0;
  while (i < expression.length) {
    const ch = expression[i];
    if (/\s/.test(ch)) {
      i += 1;
      continue;
    }
    if (/[0-9.]/.test(ch)) {
      let j = i;
      let dots = 0;
      while (j < expression.length && /[0-9.]/.test(expression[j])) {
        if (expression[j] === '.') dots += 1;
        j += 1;
      }
      if (dots > 1) throw new Error(`Bad number near "${expression.slice(i, j)}"`);
      // Scientific notation: 1e3, 2.5e-4
      if (j < expression.length && /[eE]/.test(expression[j])) {
        let k = j + 1;
        if (expression[k] === '+' || expression[k] === '-') k += 1;
        const start = k;
        while (k < expression.length && /[0-9]/.test(expression[k])) k += 1;
        if (k > start) j = k;
      }
      const value = Number(expression.slice(i, j));
      if (!Number.isFinite(value)) throw new Error(`Bad number near "${expression.slice(i, j)}"`);
      tokens.push({ kind: 'num', value });
      i = j;
      continue;
    }
    if (/[a-zA-Z_]/.test(ch)) {
      let j = i;
      while (j < expression.length && /[a-zA-Z0-9_]/.test(expression[j])) j += 1;
      tokens.push({ kind: 'name', value: expression.slice(i, j).toLowerCase() });
      i = j;
      continue;
    }
    if (ch === '(') {
      tokens.push({ kind: 'lparen' });
      i += 1;
      continue;
    }
    if (ch === ')') {
      tokens.push({ kind: 'rparen' });
      i += 1;
      continue;
    }
    if (ch === ',') {
      tokens.push({ kind: 'comma' });
      i += 1;
      continue;
    }
    if ('+-*/%^!'.includes(ch)) {
      if (ch === '*' && expression[i + 1] === '*') {
        tokens.push({ kind: 'op', value: '^' });
        i += 2;
        continue;
      }
      tokens.push({ kind: 'op', value: ch });
      i += 1;
      continue;
    }
    throw new Error(`Unexpected character "${ch}"`);
  }
  return tokens;
}

class Parser {
  private pos = 0;

  constructor(private readonly tokens: Token[]) {}

  parse(): number {
    const value = this.additive();
    if (this.pos < this.tokens.length) throw new Error('Unexpected trailing input');
    return value;
  }

  private peek(): Token | undefined {
    return this.tokens[this.pos];
  }

  private additive(): number {
    let left = this.multiplicative();
    for (;;) {
      const t = this.peek();
      if (t?.kind === 'op' && (t.value === '+' || t.value === '-')) {
        this.pos += 1;
        const right = this.multiplicative();
        left = t.value === '+' ? left + right : left - right;
      } else return left;
    }
  }

  private multiplicative(): number {
    let left = this.power();
    for (;;) {
      const t = this.peek();
      if (t?.kind === 'op' && (t.value === '*' || t.value === '/' || t.value === '%')) {
        this.pos += 1;
        const right = this.power();
        if (t.value === '*') left *= right;
        else if (t.value === '/') {
          if (right === 0) throw new Error('Division by zero');
          left /= right;
        } else {
          if (right === 0) throw new Error('Modulo by zero');
          left %= right;
        }
      } else return left;
    }
  }

  private power(): number {
    const base = this.unary();
    const t = this.peek();
    if (t?.kind === 'op' && t.value === '^') {
      this.pos += 1;
      const exp = this.power(); // right-associative
      const result = Math.pow(base, exp);
      if (!Number.isFinite(result)) throw new Error('Result is not finite');
      return result;
    }
    return base;
  }

  private unary(): number {
    const t = this.peek();
    if (t?.kind === 'op' && (t.value === '+' || t.value === '-')) {
      this.pos += 1;
      const value = this.unary();
      return t.value === '-' ? -value : value;
    }
    return this.postfix();
  }

  private postfix(): number {
    let value = this.primary();
    const t = this.peek();
    if (t?.kind === 'op' && t.value === '!') {
      this.pos += 1;
      if (!Number.isInteger(value) || value < 0 || value > 170) {
        throw new Error('Factorial needs an integer 0..170');
      }
      let acc = 1;
      for (let i = 2; i <= value; i += 1) acc *= i;
      value = acc;
    }
    return value;
  }

  private primary(): number {
    const t = this.peek();
    if (!t) throw new Error('Unexpected end of expression');
    if (t.kind === 'num') {
      this.pos += 1;
      return t.value;
    }
    if (t.kind === 'name') {
      this.pos += 1;
      const next = this.peek();
      if (next?.kind === 'lparen') {
        this.pos += 1;
        const arg = this.additive();
        const closing = this.peek();
        if (closing?.kind !== 'rparen') throw new Error(`Missing ")" after ${t.value}(`);
        this.pos += 1;
        const fn = FUNCTIONS[t.value];
        if (!fn) throw new Error(`Unknown function "${t.value}"`);
        const result = fn(arg);
        if (!Number.isFinite(result)) throw new Error('Result is not finite');
        return result;
      }
      const constant = CONSTANTS[t.value];
      if (constant === undefined) throw new Error(`Unknown name "${t.value}"`);
      return constant;
    }
    if (t.kind === 'lparen') {
      this.pos += 1;
      const value = this.additive();
      const closing = this.peek();
      if (closing?.kind !== 'rparen') throw new Error('Missing ")"');
      this.pos += 1;
      return value;
    }
    throw new Error(`Unexpected token near position ${this.pos}`);
  }
}

export function evaluateExpression(expression: string): number {
  if (!/^[0-9a-z+\-*/%^().,! \t_eE]+$/i.test(expression)) {
    throw new Error('Expression contains disallowed characters');
  }
  const result = new Parser(tokenize(expression)).parse();
  if (!Number.isFinite(result)) throw new Error('Result is not finite');
  return result;
}

/**
 * Exact arithmetic for the LLM — use instead of mental math.
 * Stateless; pure computation, no session binding needed.
 */
export function createCalculatorTool(_ctx: ToolContext = {}) {
  return tool(
    async (input: CalculatorInput): Promise<string> => {
      try {
        const result = evaluateExpression(input.expression);
        return JSON.stringify({ expression: input.expression, result }, null, 2);
      } catch (error) {
        return `Error: calculator_engine could not evaluate the expression: ${errorMessage(error)}`;
      }
    },
    {
      name: 'calculator_engine',
      description:
        'Evaluate a math expression exactly (arithmetic, %, ^, parentheses, sqrt/sin/cos/log/ln/exp/abs, pi/e). Use for any numeric computation instead of mental math.',
      schema: calculatorInputSchema,
    },
  );
}

/** Stateless — no session binding required. */
export const calculatorEngineTool = createCalculatorTool();
