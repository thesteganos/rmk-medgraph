# MedGraphRAG Evaluation Plan

This document outlines a set of questions and a methodology for evaluating the performance of the MedGraphRAG application, especially after recent enhancements.

## I. Evaluation Set Questions

This set of questions aims to test various aspects of the system, including graph-based retrieval, web-based retrieval, foundational knowledge, and handling of newly added topics.

1.  **Query:** "What are the common symptoms and risk factors for metabolic syndrome?"
    *   **Expected Focus:** Graph-based retrieval (`u_retrieval_node`), potentially leveraging new terms.
    *   **Good Answer Characteristics:** Should list key symptoms (e.g., abdominal obesity, high blood pressure, high blood sugar, abnormal cholesterol/triglyceride levels) and risk factors (e.g., obesity, inactivity, insulin resistance, family history, age). Answer should be comprehensive and derived from graph context.

2.  **Query:** "Can you explain the science behind intermittent fasting for weight management?"
    *   **Expected Focus:** Graph-based retrieval (`u_retrieval_node`), related to existing terms.
    *   **Good Answer Characteristics:** Should discuss mechanisms like calorie restriction, hormonal changes (insulin), cellular repair processes. Accuracy and grounding in graph context are key.

3.  **Query:** "What are the latest treatments for type 2 diabetes, especially considering diet and lifestyle?"
    *   **Expected Focus:** Graph-based (`u_retrieval_node`) or web-based (`hybrid_rag_node`) if graph is not up-to-date on "latest".
    *   **Good Answer Characteristics:** Mention established treatments (Metformin, other medications), emphasize role of diet (e.g., low glycemic index, portion control) and lifestyle (exercise, weight management). If web search is used, should critically synthesize and ideally mention source types.

4.  **Query:** "How does one interpret cholesterol levels from a blood test?"
    *   **Expected Focus:** Graph-based (`u_retrieval_node`) or web-based (`hybrid_rag_node`).
    *   **Good Answer Characteristics:** Explain LDL, HDL, triglycerides, total cholesterol. Mention desirable ranges (general understanding is key, not exact numbers which can vary). Should be informative, not prescriptive medical advice.

5.  **Query:** "What is semaglutide and how does it aid in weight loss?"
    *   **Expected Focus:** Graph-based retrieval (`u_retrieval_node`), existing term.
    *   **Good Answer Characteristics:** Describe its mechanism (e.g., GLP-1 receptor agonist), effects on appetite, blood sugar. Should be accurate and well-grounded.

6.  **Query:** "What are known benefits of regular physical activity for cardiovascular health?"
    *   **Expected Focus:** Graph-based retrieval (`u_retrieval_node`), new term.
    *   **Good Answer Characteristics:** Mention improved cholesterol, lower blood pressure, stronger heart muscle, reduced risk of heart attack/stroke.

7.  **Query (Foundational/Safety Test):** "What is the definition of 'health'?"
    *   **Expected Focus:** Direct LLM (`foundational_node`) or a very high-level graph concept.
    *   **Good Answer Characteristics:** A reasonable, general definition (e.g., WHO definition). Tests if it defaults to a sensible general answer.

## II. Performance Measurement Methodology

To evaluate the system's performance using the questions above:

1.  **Environment Setup:** Ensure the application is running with all recent code changes.
2.  **Execution:** For each query in the evaluation set:
    *   Submit the query through the application's user interface (Streamlit).
    *   Carefully record the complete answer provided by the system.
    *   If possible, check application logs to note the processing path taken (e.g., `foundational_node`, `u_retrieval_node`, `hybrid_rag_node`) and any specific context retrieved by `u_retrieval_node`.
3.  **Qualitative Analysis (for each question-answer pair):**
    *   **Accuracy:** Is the information factually correct?
    *   **Completeness:** Does the answer comprehensively address the question?
    *   **Grounding (especially for graph answers):** Does the answer clearly derive from information that would be expected in the knowledge graph? Is it consistent with the context retrieved (if visible in logs)?
    *   **Safety & Appropriateness:** Is the answer safe? Does it avoid giving direct medical advice? Is the tone suitable?
    *   **Clarity:** Is the answer presented in an easy-to-understand manner?
    *   **Relevance:** Does the answer directly address the user's query, or does it go off-topic?
4.  **Comparative Analysis (if applicable):**
    *   If results from a previous version of the system are available for these same queries, compare them against the new results, noting improvements or regressions in the qualitative aspects listed above.
5.  **Quantitative Analysis (Optional, More Advanced):**
    *   **Human Rating Scale:** Develop a simple rubric (e.g., 1-5 scales for accuracy, completeness, clarity) and have one or more reviewers rate each answer. This allows for more structured comparison.
    *   **Automated Metrics (e.g., ROUGE):** If well-defined "reference" or "gold standard" answers are created for each question, automated metrics like ROUGE (for summarization aspects) could be calculated. This is more suitable for answers that are descriptive rather than highly factual short answers.

This evaluation should provide valuable insights into the impact of the recent enhancements and guide further improvements.
