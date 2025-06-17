import json
import os
from collections import Counter

FEEDBACK_FILE = "feedback_for_review.jsonl"
MAX_BAD_EXAMPLES_TO_SHOW = 5

def main():
    if not os.path.exists(FEEDBACK_FILE):
        print(f"Feedback file '{FEEDBACK_FILE}' not found.")
        return

    bad_feedback_queries = []
    feedback_summary = Counter()
    bad_interactions_examples = []
    total_feedback_entries = 0

    print(f"--- Analyzing Feedback from '{FEEDBACK_FILE}' ---")

    with open(FEEDBACK_FILE, "r", encoding='utf-8') as f:
        for line in f:
            total_feedback_entries += 1
            try:
                log_entry = json.loads(line)
                feedback_value = log_entry.get("feedback")
                interaction = log_entry.get("interaction")

                if feedback_value:
                    feedback_summary[feedback_value.lower()] += 1

                if feedback_value == "bad" and interaction:
                    query = interaction.get("query")
                    if query:
                        bad_feedback_queries.append(query)

                    if len(bad_interactions_examples) < MAX_BAD_EXAMPLES_TO_SHOW:
                        bad_interactions_examples.append({
                            "query": query,
                            "answer": interaction.get("answer"),
                            "user_profile": interaction.get("user_profile", {})
                        })
            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON from line: {line.strip()}")
            except Exception as e:
                print(f"Warning: Error processing line: {line.strip()} - {e}")

    print("\n--- Overall Feedback Summary ---")
    print(f"Total feedback entries processed: {total_feedback_entries}")
    for type, count in feedback_summary.items():
        print(f"  {type.capitalize()} feedback: {count}")

    if bad_feedback_queries:
        print("\n--- Top Queries with 'Bad' Feedback ---")
        most_common_bad_queries = Counter(bad_feedback_queries).most_common(10)
        for query, count in most_common_bad_queries:
            print(f"  Count: {count} | Query: \"{query}\"") # Ensure query is quoted for readability
    else:
        print("\nNo 'bad' feedback entries found to analyze queries from.")

    if bad_interactions_examples:
        print(f"\n--- Examples of Interactions Rated 'Bad' (up to {MAX_BAD_EXAMPLES_TO_SHOW}) ---")
        for i, example in enumerate(bad_interactions_examples):
            print(f"Example {i+1}:")
            print(f"  Query: {example['query']}")
            print(f"  Answer: {example['answer']}")
            print(f"  User Profile: {example['user_profile']}")
            print("-" * 20)

    print("\n--- End of Analysis ---")
    print("Actionable Insights: Review the 'Top Queries with Bad Feedback'. Consider if these topics indicate gaps in knowledge sources (`TERMS_OF_INTEREST` in `knowledge_pipeline.py`) or if the answers suggest issues with specific graph paths or LLM reasoning for those queries.")

if __name__ == "__main__":
    main()
