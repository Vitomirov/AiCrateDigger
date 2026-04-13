type ResultCardProps = {
  title: string;
  subtitle?: string;
};

export default function ResultCard({ title, subtitle }: ResultCardProps) {
  return (
    <article className="rounded-lg border border-zinc-800 bg-zinc-900 p-4">
      <h3 className="text-sm font-medium text-zinc-100">{title}</h3>
      {subtitle ? <p className="mt-2 text-xs text-zinc-400">{subtitle}</p> : null}
    </article>
  );
}
