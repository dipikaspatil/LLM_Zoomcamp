import sqlite3
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

class SQLiteSpanExporter(SpanExporter):

    def __init__(self, db_path="traces.db"):
        # opens/creates the SQLite file, and creates the table
        # if this is the first run (IF NOT EXISTS)
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS spans (
                name TEXT,
                start_time INTEGER,   -- OTel timestamps are nanoseconds since epoch, stored raw
                end_time INTEGER,
                input_tokens INTEGER, -- NULL for spans that never call set_attribute on these
                output_tokens INTEGER,
                cost REAL
            )
        """)
        self.conn.commit()

    def export(self, spans):
        # SimpleSpanProcessor calls this once per finished span (synchronously,
        # right when the `with` block closes) — `spans` is a short list,
        # usually just the one span that just ended
        for span in spans:
            attrs = dict(span.attributes or {})
            self.conn.execute(
                "INSERT INTO spans VALUES (?, ?, ?, ?, ?, ?)",
                (
                    span.name,
                    span.start_time,
                    span.end_time,
                    # .get() returns None for spans that never set these —
                    # i.e. "rag" and "search", which only "llm" populates
                    attrs.get("input_tokens"),
                    attrs.get("output_tokens"),
                    attrs.get("cost"),
                ),
            )
        self.conn.commit()
        return SpanExportResult.SUCCESS   # tells OTel the export didn't fail

    def shutdown(self):
        self.conn.close()

    def force_flush(self):
        return True   # no buffering happening here, so nothing to flush
