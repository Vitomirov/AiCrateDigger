import ChatInput from "../components/ChatInput";
import ReasoningPanel from "../components/ReasoningPanel";
import ResultCard from "../components/ResultCard";
import { fetchHealth } from "../lib/api";

export default async function HomePage() {
  let healthStatus = "unreachable";

  try {
    const health = await fetchHealth();
    healthStatus = health.status;
  } catch {
    healthStatus = "unreachable";
  }

  return (
    <main className="min-h-screen bg-zinc-950 text-zinc-100">
      <div className="mx-auto flex min-h-screen w-full max-w-3xl flex-col gap-6 px-4 py-8">
        <h1 className="text-center text-2xl font-semibold">AiCrateDigg</h1>

        <ReasoningPanel message={`Backend health-check: ${healthStatus}`} />

        <section className="flex-1 space-y-3">
          <ResultCard title="No listings yet" subtitle="Search results will appear here." />
          <ResultCard title="Placeholder result area" subtitle="Streaming cards will pop in here." />
        </section>

        <ChatInput />
      </div>
    </main>
  );
}
