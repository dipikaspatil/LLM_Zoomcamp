"use client"; // needed because this component uses hooks (useState) and browser APIs (fetch) —
              // Next.js App Router components are server-only by default unless marked otherwise

import { useState } from "react";
import { streamChat } from "@/lib/streamChat";

type Section = "world_cup" | "knowledge";
type Message = { role: "user" | "assistant"; text: string };

export default function ChatInterface() {
  const [section, setSection] = useState<Section>("world_cup");
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!input.trim() || isStreaming) return;

    const question = input;
    setInput("");

    // Add the user's message, then an empty assistant placeholder that
    // we'll fill in token by token as the stream arrives
    setMessages((prev) => [
      ...prev,
      { role: "user", text: question },
      { role: "assistant", text: "" },
    ]);
    setIsStreaming(true);

    await streamChat(
      question,
      section,
      (token) => {
        // Append each incoming token to the last message (the assistant placeholder)
        setMessages((prev) => {
          const updated = [...prev];
          const last = updated[updated.length - 1];
          updated[updated.length - 1] = { ...last, text: last.text + token };
          return updated;
        });
      },
      () => setIsStreaming(false)
    );
  }

  return (
    <div className="flex flex-col h-screen max-w-4xl mx-auto p-4">
      {/* Section picker */}
      <div className="flex gap-2 mb-4">
        {(["world_cup", "knowledge"] as Section[]).map((s) => (
          <button
            key={s}
            onClick={() => setSection(s)}
            className={`px-4 py-2 rounded ${
              section === s ? "bg-blue-600 text-white" : "bg-gray-200 text-gray-900"
            }`}
          >
            {s === "world_cup" ? "World Cup" : "Knowledge"}
          </button>
        ))}
      </div>

      {/* Input - moved above the message list, made bigger/more prominent */}
      <form onSubmit={handleSubmit} className="flex gap-3 mb-6">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder={`Ask about ${section === "world_cup" ? "the World Cup" : "football knowledge"}...`}
          className="flex-1 border-2 border-gray-300 rounded-xl px-6 py-6 text-xl bg-white text-gray-900 shadow-sm"
          disabled={isStreaming}
        />
        <button
          type="submit"
          disabled={isStreaming}
          className="bg-blue-600 text-white px-8 py-4 text-xl rounded-xl disabled:opacity-50"
        >
          Send
        </button>
      </form>


      {/* Message list - now grows downward below the input as the conversation continues */}
      <div className="flex-1 overflow-y-auto space-y-3">
        {messages.map((m, i) => (
          <div
            key={i}
            className={`p-3 rounded max-w-[80%] ${
              m.role === "user" ? "bg-blue-100 text-gray-900 ml-auto" : "bg-gray-100 text-gray-900"
            }`}
          >
            {m.text || (isStreaming && i === messages.length - 1 ? "..." : "")}
          </div>
        ))}
      </div>
    </div>
  );
}
