import type { ListingResultDto } from "@/lib/api-types";

import ListingResultCard from "./ListingResultCard";

type SearchResultsListProps = {
  listings: ListingResultDto[];
  compact?: boolean;
};

export default function SearchResultsList({ listings, compact = true }: SearchResultsListProps) {
  if (listings.length === 0) {
    return null;
  }

  return (
    <div className="relative z-[2] w-full px-3 pb-6 pt-2 sm:px-5 md:pb-10">
      <div className="mx-auto flex max-w-xl flex-col gap-4 md:max-w-2xl">
        {listings.map((row) => (
          <ListingResultCard key={row.url} listing={row} compact={compact} />
        ))}
      </div>
    </div>
  );
}
