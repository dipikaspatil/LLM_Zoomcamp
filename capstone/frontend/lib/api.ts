const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export type Conversation = {
  thread_id: string;
  title: string;
  created_at: string;
  updated_at: string;
};

export type StoredMessage = {
  role: "user" | "assistant";
  text: string;
};

export async function fetchConversations(): Promise<Conversation[]> {
  const response = await fetch(`${API_URL}/conversations`);
  if (!response.ok) throw new Error("Failed to fetch conversations");
  return response.json();
}

export async function fetchConversationMessages(threadId: string): Promise<StoredMessage[]> {
  const response = await fetch(`${API_URL}/conversations/${threadId}/messages`);
  if (!response.ok) throw new Error("Failed to fetch conversation messages");
  return response.json();
}
