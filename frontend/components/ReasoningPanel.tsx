type ReasoningPanelProps = {
  message: string;
};

export default function ReasoningPanel({ message }: ReasoningPanelProps) {
  return (
    <section className="rounded-lg border border-zinc-800 bg-zinc-900 p-4">
      <p className="text-xs text-zinc-400">{message}</p>
    </section>
  );
}
