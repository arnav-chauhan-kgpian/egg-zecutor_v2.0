import { Playground } from '@/components/Playground';
import { SiteHeader } from '@/components/SiteHeader';

export const metadata = {
  title: 'Execution Playground',
  description: 'Run arbitrary scripts and collect their output and artifacts.',
};

export default function PlaygroundPage() {
  return (
    <div className="flex h-dvh flex-col">
      <SiteHeader />
      <main className="min-h-0 flex-1">
        <Playground />
      </main>
    </div>
  );
}
