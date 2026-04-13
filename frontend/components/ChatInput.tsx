type ChatInputProps = {
  placeholder?: string;
};

export default function ChatInput({ placeholder = "Search records worldwide..." }: ChatInputProps) {
  return (
    <div className="sticky bottom-0 w-full border border-zinc-800 bg-zinc-900 p-3 shadow-lg">
      <input
        type="text"
        placeholder={placeholder}
        className="w-full rounded-md border border-zinc-700 bg-zinc-950 px-4 py-3 text-sm text-zinc-100 outline-none placeholder:text-zinc-500"
      />
    </div>
  );
}
