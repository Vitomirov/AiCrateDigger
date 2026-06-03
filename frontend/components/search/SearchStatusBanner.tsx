type SearchStatusBannerProps = {
  message: string;
  variant: "error" | "info";
};

export default function SearchStatusBanner({ message, variant }: SearchStatusBannerProps) {
  if (variant === "error") {
    return (
      <div className="shrink-0 px-4 pb-2" role="alert">
        <p className="rounded-md border border-red-700/55 bg-black/75 px-3 py-2 text-center text-[0.78rem] font-semibold leading-snug text-red-200">
          {message}
        </p>
      </div>
    );
  }

  return (
    <div className="shrink-0 px-4 pb-2 sm:pb-3">
      <p className="rounded-md border border-crate-gold/35 bg-black/65 px-3 py-2 text-center text-[0.78rem] leading-snug text-crate-cream/80">
        {message}
      </p>
    </div>
  );
}
