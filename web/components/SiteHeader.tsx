import Link from 'next/link';
import { AuthWidget } from './AuthWidget';
import { ThemeToggle } from './ThemeToggle';

export function SiteHeader({ children }: { children?: React.ReactNode }) {
  return (
    <header className="flex h-14 shrink-0 items-center justify-between gap-4 border-b border-slate-200 bg-white px-4 dark:border-slate-800 dark:bg-slate-900">
      <div className="flex min-w-0 items-center gap-4">
        <Link href="/playground" className="shrink-0 text-lg font-bold tracking-tight">
          Exec<span className="text-emerald-600 dark:text-emerald-400">Lab</span>
        </Link>
        {children}
      </div>

      <div className="flex shrink-0 items-center gap-3">
        <ThemeToggle />
        <AuthWidget />
      </div>
    </header>
  );
}
