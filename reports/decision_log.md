# Decision Log

## 1. Brand selection
- **AppleSupport** was selected as the target brand because the dataset contains a large number of AppleSupport conversations, giving enough historical examples for retrieval and evaluation.

## 2. Thread boundary
- A thread starts from a customer root tweet with no `in_response_to_tweet_id` and is reconstructed by following `response_tweet_id`. This was preferred over time-window clustering because unrelated customer issues can occur close together.

## 3. Resolution heuristic
- A thread is marked `resolved` when the conversation ends on a brand-authored response. This is only a proxy for resolution because the dataset does not explicitly indicate customer satisfaction or successful resolution.

## 4. Brand-specific preprocessing
- The processed dataset keeps AppleSupport conversations while preserving the original tweet IDs and reply relationships so that retrieved evidence can be traced back to historical conversations.

## 5. Intent taxonomy
- A compact 10-intent taxonomy was created from inspection of AppleSupport customer messages rather than using generic support categories.

## 6. Primary-symptom rule
- Ambiguous messages are classified by the user's primary symptom rather than the surrounding context. For example, battery drain after an iOS update is treated as `battery_power`, while an app failure after an update is treated as `app_media_issue`.

## 7. Trivial baseline
- The majority-class classifier is included as a deliberately weak baseline. It establishes how much performance comes simply from exploiting class imbalance.

## 8. Simple baseline
- TF-IDF features with Logistic Regression were chosen as the simple learned baseline because they are fast, reproducible, interpretable, and provide a meaningful comparison against the LLM classifier.

## 9. Retrieval method
- TF-IDF cosine similarity was used for historical precedent retrieval instead of embeddings. This keeps the system lightweight, transparent, and reproducible within the assignment's short runtime constraint.

## 10. Retrieval grounding
- Only historically reconstructed threads marked as resolved by the heuristic are used as retrieval precedents. The limitation of this heuristic is explicitly reported rather than treating it as ground truth.

## 11. Escalation design
- Escalation is rule-based rather than another LLM call. The decision uses classifier confidence and retrieval similarity so that escalation remains cheap and auditable.

## 12. Security escalation
- `account_security` is always escalated to a human because account and security-related requests have higher risk than ordinary troubleshooting.

## 13. Confidence threshold
- The minimum classifier confidence for auto-handling was set to `0.75`. Lower-confidence predictions are escalated instead of being answered automatically.

## 14. Grounding threshold
- The minimum historical retrieval similarity for auto-handling was set to `0.50`. If no sufficiently similar precedent exists, the system escalates rather than inventing a brand-specific policy.

## 15. Golden-set size and judge design
- A 200-example golden set was selected to stay within the assignment's requested 150–250 range. The LLM judge evaluates groundedness, relevance, actionability, and tone, while a separate agreement script compares judge scores with human ratings.