"use client";

import { useState, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { streamChat } from "@/lib/streamChat";
import { fetchConversations, fetchConversationMessages, Conversation } from "@/lib/api";
import ConversationSidebar from "@/components/ConversationSidebar";

type Message = { role: "user" | "assistant"; text: string };

export default function ChatInterface() {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [threadId, setThreadId] = useState(() => crypto.randomUUID());
  const [conversations, setConversations] = useState<Conversation[]>([]);

  // Load the conversation list once when the page first loads
  useEffect(() => {
    fetchConversations().then(setConversations).catch(() => {});
  }, []);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!input.trim() || isStreaming) return;

    const question = input;
    setInput("");

    setMessages((prev) => [
      ...prev,
      { role: "user", text: question },
      { role: "assistant", text: "" },
    ]);
    setIsStreaming(true);

    await streamChat(
      question,
      threadId,
      (token) => {
        setMessages((prev) => {
          const updated = [...prev];
          const last = updated[updated.length - 1];
          updated[updated.length - 1] = { ...last, text: last.text + token };
          return updated;
        });
      },
      () => {
        setIsStreaming(false);
        // Refresh the sidebar — this message may have created a brand-new
        // conversation, or bumped an existing one to the top of the list
        fetchConversations().then(setConversations).catch(() => {});
      }
    );
  }

  async function handleSelectConversation(selectedThreadId: string) {
    if (selectedThreadId === threadId || isStreaming) return;
    setThreadId(selectedThreadId);
    const history = await fetchConversationMessages(selectedThreadId);
    setMessages(history);
  }

  function handleNewConversation() {
    if (isStreaming) return;
    setThreadId(crypto.randomUUID());
    setMessages([]);
  }

  return (
    <div className="flex h-screen">
      <ConversationSidebar
        conversations={conversations}
        activeThreadId={threadId}
        onSelect={handleSelectConversation}
        onNewConversation={handleNewConversation}
      />

      <div className="flex flex-col flex-1 max-w-6xl mx-auto p-4">
        <form onSubmit={handleSubmit} className="flex gap-3 mb-6">
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask about football..."
            className="flex-1 border-2 border-gray-300 rounded-xl px-6 py-6 text-xl placeholder:text-base bg-white text-gray-900 shadow-sm"
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

        <div className="flex-1 overflow-y-auto space-y-3">
          {messages.map((m, i) => (
            <div
              key={i}
              className={`p-3 rounded ${
                m.role === "user"
                  ? "bg-blue-100 text-gray-900 ml-auto max-w-[80%]"
                  : "bg-gray-100 text-gray-900 max-w-full"
              }`}
            >
              {m.role === "assistant" ? (
                <ReactMarkdown
                  remarkPlugins={[remarkGfm]}
                  components={{
                    table: ({ children }) => (
                      <table className="border-collapse w-full my-2 text-sm">{children}</table>
                    ),
                    th: ({ children }) => (
                      <th className="border border-gray-300 bg-gray-200 px-2 py-1 text-left">
                        {children}
                      </th>
                    ),
                    td: ({ children }) => (
                      <td className="border border-gray-300 px-2 py-1">{children}</td>
                    ),
                  }}
                >
                  {m.text || (isStreaming && i === messages.length - 1 ? "..." : "")}
                </ReactMarkdown>
              ) : (
                m.text
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
