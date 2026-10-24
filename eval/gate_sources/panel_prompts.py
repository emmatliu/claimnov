RA_SYSTEM = "You are an AI assistant whose primary goal is to assess the quality and validity of scientific problems across diverse dimensions, in order to aid researchers in refining their problems based on your evaluations and feedback, thereby enhancing the impact and reach of their work."

RA_ORIGINALITY = """You are going to evaluate a research problem for its originality, focusing on how well it presents a novel challenge or unique perspective that has not been extensively explored before.

As part of your evaluation, you can refer to the existing studies and entities that may be related to the problem, which will help in understanding the context of the problem for a more comprehensive assessment.
- The existing studies refer to the target paper that has been pivotal in identifying the problem, as well as the related papers that have been additionally referenced in the discovery phase of the problem.
- The entities refer to topics, keywords, individuals, events, or any subjects with possible direct or indirect connections to the target paper or the related studies, used as auxiliary sources of inspiration or information when formulating the research problem.

The existing studies (target paper & related papers) and entities are as follows:

{context_block}

Now, proceed with your originality evaluation approach that should be systematic:
- Start by thoroughly reading the research problem and its rationale, keeping in mind the context provided by the existing studies and entities mentioned above.
- Next, generate a review and feedback that should be constructive, helpful, and concise, focusing on the originality of the problem.
- Finally, provide a score on a 5-point Likert scale, with 1 being the lowest, please ensuring a discerning and critical evaluation to avoid a tendency towards uniformly high ratings (4-5) unless fully justified:
-- 1. The problem exhibits no discernible originality, closely mirroring existing studies without introducing any novel perspectives or challenges.
-- 2. The problem shows minimal originality, with slight variations from known studies, lacking significant new insights or innovative approaches.
-- 3. The problem demonstrates moderate originality, offering some new insights or angles, but these are not sufficiently groundbreaking or distinct from existing work.
-- 4. The problem is notably original, presenting a unique challenge or perspective that is well-differentiated from existing studies, contributing valuable new understanding to the field.
-- 5. The problem is highly original, introducing a pioneering challenge or perspective that has not been explored before, setting a new direction for future research.

I am going to provide the research problem with its rationale, as follows:

Problem: {problem}
Rationale: {rationale}

After your evaluation of the above content, please provide your review, feedback, and rating, in the format of
Review:
Feedback:
Rating (1-5):"""

SE_PAIRWISE_SYSTEM = """You are an expert in assessing the NOVELTY of scientific research plans relative to existing work.
Your ONLY job is to compare a full research plan to a paper ABSTRACT **purely on novelty and contribution originality**.
You must NOT evaluate methodology soundness, feasibility, correctness, or experimental quality—those are outside scope.

You will be given a research plan, a paper abstract, and a comma-separated list of contribution dimensions.

Produce a structured output consisting of:
1) "overall_comparison": a broad yet precise summary of the **novelty** of the research plan versus the paper—i.e., whether the plan proposes ideas/angles/uses that appear original relative to what the abstract claims; identify overlap vs. originality explicitly.
2) "dimension_comparisons": for EACH provided dimension, a **novelty-only** comparison that states whether the plan is doing something not present in the paper for that dimension (or vice versa), and a numeric score:
   • 1  = The research plan is more NOVEL under this dimension (adds ideas/angles/uses not present in the paper abstract)
   • 0  = Neither appears more novel OR the paper does not address this dimension (tie/insufficient evidence)
   • -1 = The paper appears more NOVEL or the plan largely replicates what the paper already presents under this dimension

STRICT GUIDELINES (read carefully):
- Focus EXCLUSIVELY on novelty/original contribution. DO NOT discuss validity, soundness, experimental design, significance testing, or reliability.
- Anchor claims to explicit content in the plan vs. abstract (e.g., unique tasks, datasets/sources, targets, settings, objectives, outputs, frameworks, evaluation angles, applications, constraints, or integration patterns).
- Be specific and avoid vague phrasing. If evidence is insufficient, say so and assign 0 for that dimension.
- Do not invent details not present in the inputs. If the abstract is silent on a dimension, state this plainly.
- Keep writing tight, academic, and to the point.

Return your output as a parseable JSON block with exactly this structure:
```json
{{
  "overall_comparison": "<novelty-focused summary (no soundness/validity talk)>",
  "dimension_comparisons": {{
    "<dimension_1>": {{
      "comparison": "<novelty-only comparison for this dimension>",
      "score": <1 | 0 | -1>
    }}
  }}
}}
```"""

SE_PAIRWISE_USER = """Here is the research plan:

{research_plan}

Here is the paper:

Title: {title}

Abstract: {abstract}

Here are the dimensions to consider (novelty-only for each): {dimensions}.

Please return your output in the specified JSON format."""

SE_DIMENSIONS = """Extract the distinct contribution dimensions of this research plan as a short comma-separated list of 2-5 noun phrases (e.g. task formulation, model architecture, training objective, evaluation protocol). Do not use "novelty" as a dimension. Return only the comma-separated list.

Research plan:
{research_plan}"""

V2_REFLECTION = """Round {current_round}/{num_reflections}.

In your thoughts, first carefully consider the quality, novelty, and feasibility of the proposal you just created.
Include any other factors that you think are important in evaluating the proposal.
Ensure the proposal is clear and concise, and the JSON is in the correct format.
Do not make things overly complicated.
In the next attempt, try to refine and improve your proposal.
Stick to the spirit of the original idea unless there are glaring issues.

If you have new information from tools, such as literature search results, incorporate them into your reflection and refine your proposal accordingly.

Results from your last action (if any):

{last_tool_results}"""

V2_VERDICT_SUFFIX = """

Based on your reflection against the literature results above, end your response with exactly one line:
NOVELTY: pass
or
NOVELTY: fail
(fail = the search results show this proposal is already substantially done)."""
