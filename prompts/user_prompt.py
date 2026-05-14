lt_dataset_generation_user_prompt="""
Generate {n} Lithuanian question-answer pairs on the following topics: {topic}.

Constraints:
- Questions must be natural and realistic
- Use varied question types (what, why, how, where, when, who)
- Answers must be:
  - grammatically correct
  - clear and informative

You MUST generate exactly {n} pairs. No more, no less.
"""

lt_transformers_user_prompt = """Generate exactly {n} unique question-answer pairs in Lithuanian on the topic: {topic}.

Requirements:
1. Both questions and answers must be written in Lithuanian.
2. Each question must ask about a specific, easily verifiable fact (a date, a name, a number, a place, a measurement).
3. Each answer must be 1-3 sentences and contain the concrete fact that answers the question.
4. Use varied question types: "Kada...?" (When), "Koks/Kokia...?" (What), "Kur...?" (Where), "Kas...?" (Who), "Kiek...?" (How many).
5. Do NOT repeat questions or facts across pairs.
6. Every fact must be well-known and easily checkable via a simple search.

Output format — respond with ONLY a JSON array, nothing else:
[
  {{"question": "Lithuanian question here", "answer": "Lithuanian answer here"}},
  ...
]

Generate EXACTLY {n} pairs."""
