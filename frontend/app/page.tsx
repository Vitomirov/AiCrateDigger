import SearchExperience from "../components/SearchExperience";
import { fetchHealth } from "../lib/api";

export default async function HomePage() {
  let healthHint = "";
  try {
    const health = await fetchHealth();
    healthHint = health.status === "ok" ? "Backend online." : `API responded: ${health.status}`;
  } catch {
    healthHint = "Offline — search may proxy once backend wakes.";
  }

  return (
    <main className="crate-grain relative flex min-h-[100dvh] w-full flex-col overflow-x-hidden bg-crate-panel">
      <p className="sr-only" aria-live="polite">
        {healthHint}
      </p>
      <SearchExperience />
    </main>
  );
}
