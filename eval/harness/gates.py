import json, re
from difflib import SequenceMatcher
from . import llm

JUDGE = "gpt-4o-mini"

V1_SYSTEM = """You are an ambitious AI PhD student who is looking to publish a paper that will contribute significantly to the field.
You have an idea and you want to check if it is novel or not. I.e., not overlapping significantly with existing literature or already well explored.
Be a harsh critic for novelty, ensure there is a sufficient contribution in the idea for a new conference or workshop paper.
You will be given access to the Semantic Scholar API, which you may use to survey the literature and find relevant papers to help you make your decision.
The top 10 results for any search query will be presented to you with the abstracts.

You will be given {num_rounds} to decide on the paper, but you do not need to use them all.
At any round, you may exit early and decide on the novelty of the idea.
Decide a paper idea is novel if after sufficient searching, you have not found a paper that significantly overlaps with your idea.
Decide a paper idea is not novel, if you have found a paper that significantly overlaps with your idea.

{task_description}
<experiment.py>
{code}
</experiment.py>
"""
V1_PROMPT = '''Round {current_round}/{num_rounds}.
You have this idea:

"""
{idea}
"""

The results of the last query are (empty on first round):
"""
{last_query_results}
"""

Respond in the following format:

THOUGHT:
<THOUGHT>

RESPONSE:
```json
<JSON>
```

In <THOUGHT>, first briefly reason over the idea and identify any query that could help you make your decision.
If you have made your decision, add "Decision made: novel." or "Decision made: not novel." to your thoughts.

In <JSON>, respond in JSON format with ONLY the following field:
- "Query": An optional search query to search the literature (e.g. attention is all you need). You must make a query if you have not decided this round.

A query will work best if you are able to recall the exact name of the paper you are looking for, or the authors.
This JSON will be automatically parsed, so ensure the format is precise.'''

V1_TASK = "You are proposing research ideas in machine learning."
V1_CODE = "# (no experiment template supplied for this idea)"

def _v1_fmt(papers):
    if not papers: return "No papers found."
    return "\n\n".join(
        "{i}: {title}. Unknown authors. Unknown venue, {year}.\nNumber of citations: unknown\nAbstract: {ab}".format(
            i=i, title=p["title"], year=(p.get("date") or "?")[:4], ab=p["abstract"])
        for i, p in enumerate(papers))

def _extract_json(text):
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.S) or re.search(r"(\{.*?\})", text, re.S)
    if not m: return None
    try: return json.loads(m.group(1))
    except Exception: return None

def v1_gate(idea, pool, rounds=10, tag=""):
    msgs = [{"role": "system", "content": V1_SYSTEM.format(num_rounds=rounds, task_description=V1_TASK, code=V1_CODE)}]
    idea_json = json.dumps({"Name": re.sub(r"[^a-z0-9]+", "_", idea["title"].lower())[:40],
                            "Title": idea["title"], "Experiment": idea["description"]}, indent=1)
    papers_str, queries, results_log = "", [], []
    for j in range(rounds):
        msgs.append({"role": "user", "content": V1_PROMPT.format(
            current_round=j + 1, num_rounds=rounds, idea=idea_json, last_query_results=papers_str)})
        text = llm.chat(JUDGE, msgs, cache_tag=f"v1{tag}r{j}")
        msgs.append({"role": "assistant", "content": text})
        low = text.lower()
        if "decision made: novel" in low:
            return {"gate": "v1", "novel": True, "evidence": results_log[-1] if results_log else None,
                    "trace": {"rounds": j + 1, "queries": queries}}
        if "decision made: not novel" in low:
            return {"gate": "v1", "novel": False, "evidence": results_log[-1] if results_log else None,
                    "trace": {"rounds": j + 1, "queries": queries}}
        jo = _extract_json(text)
        q = (jo or {}).get("Query", "")
        if not q:
            papers_str = "No papers found."
            continue
        queries.append(q)
        papers = pool.search(q, k=10)
        results_log.append([p["title"] for p in papers[:3]])
        papers_str = _v1_fmt(papers)
    return {"gate": "v1", "novel": False, "evidence": None,
            "trace": {"rounds": rounds, "queries": queries, "timeout": True}}

def _coi_extract(text, tag_name):
    m = re.search(rf"<{tag_name}>(.*?)</{tag_name}>", text, re.S)
    return m.group(1).strip() if m else ""

COI_QUERY_PROMPT = """
You are a scientific research expert.
Your task is to check whether the target idea is similar to existing research.

The target idea you need to check is as follows:{idea}

The topic you are studying is: {topic}

Please provide multiple search queries to find relevant papers that can help you determine whether the idea is novel(no more than 3 queries).

Output strictly in the following format:
<queries>["query1", "query2", "query3"]</queries>

For example:
<queries>["Reducing reliance on large-scale annotated data and closed-source models for planning in QA tasks","Automatic agent learning for QA","QA task planning with minimal human intervention"]</queries>
"""

COI_CHECK_PROMPT = """
You are a scientific research expert tasked with evaluating the similarity between a specified idea and existing research. Your objective is to determine if the target idea closely resembles any findings in the provided papers.

The target idea you need to check is as follows:
{idea}

The relevant papers you need to refer to are as follows:
{papers_content}

Here are your guidlines:
1. Comparison Process: Begin by thoroughly comparing each paper's ideas with the target idea. Consider the methodologies, conclusions, and underlying concepts in each paper in your analysis.
2. Similarity Assessment: If the target idea shares fundamental similarities with any existing research to the extent that they can be considered identical, classify this as plagiarism.
3. Output: Your output should provide a clear thought process, the similarity assessment, a summary of the target idea, and the ID of the most relevant similar paper.

Please output strictly in the following format:
<think>{{your thinking steps}}</think>
<similar>{{0/1}}</similar>
<summary>{{the summary of the target idea}}</summary>
<similar_paper_id>{{the id of the similar paper}}</similar_paper_id>

For example:
<think> There are my think steps:... </think>
<similar>0</similar>
<summary> It proposes ... </summary>
<similar_paper_id>0</similar_paper_id>
"""

def coi_gate(idea, pool, topic="machine learning", tag=""):
    blob = f"{idea['title']}: {idea['description']}"
    qtext = llm.chat(JUDGE, [{"role": "user", "content": COI_QUERY_PROMPT.format(idea=blob, topic=topic)}], cache_tag=f"coiq{tag}")
    try:
        queries = json.loads(_coi_extract(qtext, "queries"))[:3]
    except Exception:
        queries = [blob]
    papers, seen = [], set()
    for q in queries:
        for p in pool.search(q, k=5):
            if p["title"] not in seen:
                seen.add(p["title"]); papers.append(p)
    if not papers:
        return {"gate": "coi", "novel": True, "evidence": None,
                "trace": {"queries": queries, "n_papers": 0, "fail_open": True}}
    content = "".join(f"Paper {i}: Title:{p['title']}\n Abstract:{p['abstract']}\n" for i, p in enumerate(papers))
    text = llm.chat(JUDGE, [{"role": "user", "content": COI_CHECK_PROMPT.format(idea=blob, papers_content=content)}], cache_tag=f"coic{tag}")
    similar = _coi_extract(text, "similar")
    novel = similar != "1"
    ev = None
    if not novel:
        try: ev = papers[int(_coi_extract(text, "similar_paper_id"))]["title"]
        except Exception: pass
    return {"gate": "coi", "novel": novel, "evidence": ev,
            "trace": {"queries": queries, "n_papers": len(papers)}}

SI_QUERY = ("You are a professor in Natural Language Processing. You need to evaluate the novelty of a proposed research idea.\n"
    "The idea is:\n{idea}\n\n"
    "You want to do a round of paper search in order to find out whether the proposed project has already been done. "
    "You should propose some keywords for using the Semantic Scholar API to find the most relevant papers to this proposed idea. Formulate your query as: KeywordQuery(\"keyword\"). Give me 1 - 3 queries, the keyword can be a concatenation of multiple keywords (just put a space between every word) but please be concise and try to cover all the main aspects.\n"
    "The query keywords should be specific to the proposed research idea, in order to find whether there are similar ideas in the literature. try to include language model to find relevant papers within NLP. "
    "Your query (just return the queries with no additional text, put each one in a new line without any other explanation):")

SI_SCORE = ("You are a research assistant whose job is to read the below set of papers and score each paper based on how similar the paper is to the proposed idea.\n"
    "The proposed idea is: {idea}.\n"
    "The topic is machine learning research and it should be related to large language models and NLP broadly.\n"
    "The papers are:\n{papers}\n"
    "Please score each paper from 1 to 10 based on the similarity and relevance to the proposed idea. 10 means the paper is essentially the same as the proposed idea; 1 means the paper is not even relevant to the topic; 5 means the paper shares some similarity but some key details are different.\n"
    "Write the response in JSON format with \"paperID: score\" as the key and value for each paper.\n")

SI_MATCH = ("You are a professor specialized in Natural Language Processing. You have a project proposal and want to find related works to cite in the paper. Your job is to decide whether the given paper is directly relevant to the project and should be cited as similar work.\n"
    "The project proposal is:\n{idea}.\n"
    "The paper is:\n{paper}\n"
    "The project proposal and paper abstract are considered a match if both the research problem and the approach are the same. For example, if they are both trying to improve code generation accuracy and both propose to use retrieval augmentation. Note that the method details do not matter, you should only focus on the high-level concepts and judge whether they are directly relevant.\n"
    "You should first specify what is the proposed research problem and approach. If answering yes, your explanation should be the one-sentence summary of both the abstract and the proposal and their similarity (e.g., they are both about probing biases of language models via fictional characters). If answering no, give the short summaries of the abstract and proposal separately, then highlight their differences. Then end your response with a binary judgment, saying either \"Yes\" or \"No\". Change to a new line after your explanation and just say Yes or No with no punctuation in the end.\n")

def _si_fmt(papers):
    return "\n".join(f"paperId: p{i}\ntitle: {p['title']}\nabstract: {p['abstract']}\n"
                     for i, p in enumerate(papers))

def si_gate(idea, pool, check_n=5, tag=""):
    blob = f"{idea['title']}: {idea['description']}"
    qtext = llm.chat(JUDGE, [{"role": "user", "content": SI_QUERY.format(idea=blob)}], max_tokens=100, cache_tag=f"siq{tag}")
    queries = [re.sub(r'KeywordQuery\("(.*)"\)', r"\1", q.strip()) for q in qtext.strip().split("\n") if q.strip()][:3]
    queries.append(idea["title"] + " NLP")
    bank, seen = [], set()
    for q in queries:
        for p in pool.search(q, k=10):
            if p["title"] not in seen:
                seen.add(p["title"]); bank.append(p)
    if not bank:
        return {"gate": "si", "novel": True, "evidence": None, "trace": {"queries": queries, "n_papers": 0}}
    stext = llm.chat(JUDGE, [{"role": "user", "content": SI_SCORE.format(idea=blob, papers=_si_fmt(bank))}], max_tokens=4000, cache_tag=f"sis{tag}")
    try:
        scores = json.loads(re.search(r"\{.*\}", stext, re.S).group(0))
    except Exception:
        scores = {}
    for i, p in enumerate(bank):
        p["_score"] = scores.get(f"p{i}", 0)
    top = sorted(bank, key=lambda p: -p.get("_score", 0))[:check_n]
    for p in top:
        ptxt = f"paperId: {p['title']}\ntitle: {p['title']}\nabstract: {p['abstract']}\n"
        mtext = llm.chat(JUDGE, [{"role": "user", "content": SI_MATCH.format(idea=blob, paper=ptxt)}], max_tokens=1000, cache_tag=f"sim{tag}{p['id'][:8]}")
        verdict = mtext.strip().split("\n")[-1].strip().rstrip(".").lower()
        if verdict == "yes":
            return {"gate": "si", "novel": False, "evidence": p["title"],
                    "trace": {"queries": queries, "n_papers": len(bank)}}
    return {"gate": "si", "novel": True, "evidence": None,
            "trace": {"queries": queries, "n_papers": len(bank)}}

_STOP = set("""a an and are as at be but by for from has have how its in is it of on or that the this to was were what when where which who will with we our can into using use based new novel results result paper approach method model models via they than then so such some more most also its their there these those over under between about after before during each per towards toward within without may might should would could been being do does did done only very much many few both same other another if while because against them his her he she you your it's""".split())

def _arc_kw(text):
    seen, out = set(), []
    for t in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]+", text.lower()):
        if t not in _STOP and len(t) >= 3 and t not in seen:
            seen.add(t); out.append(t)
    return out

def arc_gate(idea, pool, threshold=0.25, tag=""):
    hyp_kw = _arc_kw(f"{idea['title']}\n{idea['description']}")
    sims = []
    for p in pool.records:
        kw_sim = 0.0
        pkw = set(_arc_kw(f"{p['title']} {p['abstract']}"))
        if pkw and hyp_kw:
            kw_sim = len(set(hyp_kw) & pkw) / len(set(hyp_kw) | pkw)
        t_sim = SequenceMatcher(None, idea["title"].lower(), p["title"].lower()).ratio()
        sim = round(0.7 * kw_sim + 0.3 * t_sim, 4)
        if sim >= threshold:
            sims.append((sim, p["title"]))
    sims.sort(reverse=True)
    top5 = sims[:5]
    if not top5:
        score, assessment = 1.0, "high"
    else:
        score = round(1.0 - top5[0][0], 4)
        assessment = "critical" if score < 0.3 else ("low" if score < 0.5 else ("moderate" if score < 0.7 else "high"))
    rec = {"critical": "abort", "low": "differentiate"}.get(assessment, "proceed")
    return {"gate": "arc", "novel": rec == "proceed", "evidence": top5[0][1] if top5 else None,
            "trace": {"novelty_score": score, "assessment": assessment, "recommendation": rec,
                      "advisory": True}}
