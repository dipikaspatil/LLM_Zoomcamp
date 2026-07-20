from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

from sqlite_exporter import SQLiteSpanExporter

provider = TracerProvider()
provider.add_span_processor(
    SimpleSpanProcessor(SQLiteSpanExporter("traces.db"))
)
trace.set_tracer_provider(provider)

tracer = trace.get_tracer("llm-zoomcamp")

from rag_helper import RAGBase


class RAGTraced(RAGBase):

    def rag(self, query):
        with tracer.start_as_current_span("rag"):
            return super().rag(query)

    def search(self, query, num_results=5):
        with tracer.start_as_current_span("search"):
            return super().search(query, num_results=num_results)

    def llm(self, prompt):
        # capture the span reference with `as span` so we can call
        # set_attribute on it before the `with` block closes
        with tracer.start_as_current_span("llm") as span:
            response = super().llm(prompt)   # raw OpenAI response object
            usage = response.usage

            # attach the numbers we care about to THIS span
            span.set_attribute("input_tokens", usage.input_tokens)
            span.set_attribute("output_tokens", usage.output_tokens)

            cost = calculate_cost(self.model, usage)
            span.set_attribute("cost", cost)

            return response   # still returns the response, unchanged for rag()

def calculate_cost(model, usage):
    # same per-million-token pricing pattern used in lesson 4 (metrics.py)
    cost = 0
    if "gpt-5.4-mini" in model:
        cost = (usage.input_tokens * 0.15 + usage.output_tokens * 0.60) / 1_000_000
    return cost

from dotenv import load_dotenv
load_dotenv()

from starter import index, client

rag = RAGTraced(index=index, llm_client=client)

query = "How does the agentic loop keep calling the model until it stops?"
answer = rag.rag(query)
print(answer)
