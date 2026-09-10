
from .llm_client import complete
from .retrieve import retrieve

DRAFT_PROMPT = """You are drafting a customer support reply for {brand} on Twitter.

Customer message:
\"\"\"{message}\"\"\"

Intent: {intent}

Here is how {brand} has resolved similar issues in the past (most similar first).
Ground your reply in these patterns — match the brand's actual tone, policy, and
typical next steps. Do not invent a policy that doesn't appear below.

{precedents}

Write only the reply text (no preamble, no labels). Keep it in the brand's typical
voice and length (these are Twitter replies, keep it tight)."""


def format_precedents(threads: list[dict]) -> str:
    if not threads:
        return "(No sufficiently similar past resolution found — draft cautiously and " \
               "prefer general, verifiable guidance over specific policy claims.)"
    blocks = []
    for t in threads:
        turns = " -> ".join(f"[{x['author']}] {x['text']}" for x in t["turns"])
        blocks.append(f"(similarity {t['similarity']:.2f}) {turns}")
    return "\n\n".join(blocks)


def draft_reply(message: str, intent: str, brand: str) -> dict:
    precedents = retrieve(message)
    prompt = DRAFT_PROMPT.format(
        brand=brand,
        message=message,
        intent=intent,
        precedents=format_precedents(precedents),
    )
    reply = complete(prompt, role="draft", temperature=0.3)
    return {
        "reply": reply.strip(),
        "grounding_threads": [t["thread_id"] for t in precedents],
        "max_grounding_similarity": max((t["similarity"] for t in precedents), default=0.0),
    }
