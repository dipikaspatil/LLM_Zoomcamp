"use client";

import { Conversation } from "@/lib/api";

type Props = {
  conversations: Conversation[];
  activeThreadId: string;
  onSelect: (threadId: string) => void;
  onNewConversation: () => void;
};

export default function ConversationSidebar({
  conversations,
  activeThreadId,
  onSelect,
  onNewConversation,
}: Props) {
  return (
    <div className="w-64 flex-shrink-0 border-r border-gray-300 h-screen overflow-y-auto p-3 bg-gray-50">
      <button
        onClick={onNewConversation}
        className="w-full mb-3 bg-blue-600 text-white px-3 py-2 rounded-lg text-sm font-medium"
      >
        + New Conversation
      </button>

      <div className="space-y-1">
        {conversations.map((c) => (
          <button
            key={c.thread_id}
            onClick={() => onSelect(c.thread_id)}
            className={`w-full text-left px-3 py-2 rounded-lg text-sm truncate ${
              c.thread_id === activeThreadId
                ? "bg-blue-100 text-blue-900 font-medium"
                : "text-gray-700 hover:bg-gray-200"
            }`}
          >
            {c.title}
          </button>
        ))}
      </div>
    </div>
  );
}
