type Section = "world_cup" | "knowledge";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

/**
 * Sends a question to the backend and streams the answer back token by token.
 * onToken is called for each chunk of text as it arrives; onDone when the
 * stream ends. We can't use the browser's EventSource here because it only
 * supports GET requests, and our /chat endpoint is POST with a JSON body —
 * so we read the raw streaming response ourselves instead.
 */
export async function streamChat(
  question: string,
  section: Section,
  onToken: (text: string) => void,
  onDone: () => void
) {
  const response = await fetch(`${API_URL}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, section }),
  });

  if (!response.body) throw new Error("No response body from server");

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    // Normalize CRLF to LF so downstream splitting logic works regardless
    // of which line-ending convention the server actually sends
    buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");

    const rawEvents = buffer.split("\n\n");
    buffer = rawEvents.pop() ?? "";

    for (const rawEvent of rawEvents) {
      let eventType = "message";
      let data = "";

      for (const line of rawEvent.split("\n")) {
        if (line.startsWith("event:")) eventType = line.slice(6).trim();
        if (line.startsWith("data:")) {
          let value = line.slice(5);
          // SSE spec: strip exactly one leading space (the field delimiter) if present —
          // anything beyond that is real content and must be preserved, including
          // meaningful spaces between words in a streamed token
          if (value.startsWith(" ")) value = value.slice(1);
          data += value;
        }
      }


      if (eventType === "token") onToken(data);
      if (eventType === "done") onDone();
    }
  }
}
