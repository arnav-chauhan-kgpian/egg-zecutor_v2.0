'use client';

import { useCallback, useEffect, useRef, useState } from 'react';

interface SplitPaneProps {
  direction: 'horizontal' | 'vertical';
  /** Initial size of the first pane, as a percentage. */
  initial?: number;
  min?: number;
  max?: number;
  first: React.ReactNode;
  second: React.ReactNode;
  className?: string;
}

/**
 * Two panes with a draggable divider. `horizontal` splits left/right,
 * `vertical` splits top/bottom.
 */
export function SplitPane({
  direction,
  initial = 50,
  min = 20,
  max = 80,
  first,
  second,
  className = '',
}: SplitPaneProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState(initial);
  const [dragging, setDragging] = useState(false);

  const isHorizontal = direction === 'horizontal';

  const onPointerMove = useCallback(
    (event: PointerEvent) => {
      const container = containerRef.current;
      if (!container) return;

      const rect = container.getBoundingClientRect();
      const ratio = isHorizontal
        ? ((event.clientX - rect.left) / rect.width) * 100
        : ((event.clientY - rect.top) / rect.height) * 100;

      setSize(Math.min(max, Math.max(min, ratio)));
    },
    [isHorizontal, min, max],
  );

  useEffect(() => {
    if (!dragging) return;

    const stop = () => setDragging(false);
    window.addEventListener('pointermove', onPointerMove);
    window.addEventListener('pointerup', stop);
    // Keep the cursor consistent and stop text selection while dragging.
    const previousCursor = document.body.style.cursor;
    document.body.style.cursor = isHorizontal ? 'col-resize' : 'row-resize';
    document.body.style.userSelect = 'none';

    return () => {
      window.removeEventListener('pointermove', onPointerMove);
      window.removeEventListener('pointerup', stop);
      document.body.style.cursor = previousCursor;
      document.body.style.userSelect = '';
    };
  }, [dragging, onPointerMove, isHorizontal]);

  function nudge(event: React.KeyboardEvent) {
    const back = isHorizontal ? 'ArrowLeft' : 'ArrowUp';
    const forward = isHorizontal ? 'ArrowRight' : 'ArrowDown';
    if (event.key === back) setSize((value) => Math.max(min, value - 2));
    if (event.key === forward) setSize((value) => Math.min(max, value + 2));
  }

  return (
    <div
      ref={containerRef}
      className={`flex min-h-0 min-w-0 ${isHorizontal ? 'flex-row' : 'flex-col'} ${className}`}
    >
      <div
        className="min-h-0 min-w-0 overflow-hidden"
        style={{ [isHorizontal ? 'width' : 'height']: `${size}%` }}
      >
        {first}
      </div>

      <div
        role="separator"
        aria-orientation={isHorizontal ? 'vertical' : 'horizontal'}
        aria-valuenow={Math.round(size)}
        tabIndex={0}
        onPointerDown={(event) => {
          event.preventDefault();
          setDragging(true);
        }}
        onKeyDown={nudge}
        className={`group relative shrink-0 bg-slate-200 transition-colors hover:bg-emerald-400 focus:bg-emerald-400 focus:outline-none dark:bg-slate-800 dark:hover:bg-emerald-600 ${
          isHorizontal ? 'w-1 cursor-col-resize' : 'h-1 cursor-row-resize'
        } ${dragging ? 'bg-emerald-500 dark:bg-emerald-500' : ''}`}
      >
        {/* Widen the grab area without affecting layout. */}
        <span
          aria-hidden
          className={`absolute ${isHorizontal ? '-left-1 -right-1 inset-y-0' : '-top-1 -bottom-1 inset-x-0'}`}
        />
      </div>

      <div className="min-h-0 min-w-0 flex-1 overflow-hidden">{second}</div>
    </div>
  );
}
