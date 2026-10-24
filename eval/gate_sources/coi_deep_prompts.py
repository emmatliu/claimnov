use_entities = True

def get_deep_search_query_prompt(topic = None,query = None) -> str:
    if topic and query:
        prompt = f"""
    You are a master of literature searcher, tasked with finding relevant research literatures based on a specific topic and idea.

    Currently, we would like to study the following topic: {topic}.
    And we have the following idea: {query}.

    Please provide the literature search queries you would use to search for papers related to the topic and idea.
        """
    elif topic:
        prompt = f"""
    You are a master of literature searcher, tasked with finding relevant research literatures based on a specific topic.

    Currently, we would like to study the following topic: {topic}.

    Please provide the literature search queries you would use to search for papers related to the topic. 
    """
    elif query:
        prompt = f"""
    You are a master of literature searcher, tasked with finding relevant research literatures based on a specific idea.

    Currently, we would like to search for papers related to the following idea: {query}.

    Please provide the literature search querie syou would use to search for papers related to the paper idea.
    """
    output_format = """
    Each query should be a string and should be enclosed in double quotes.It is best to output one query representing the whole and other queries representing other different aspects of the whole.(no more than 5 queries)

    Output strictly in the following format:
    <queries>["query1", "query2", ...]</queries>
        
    For example:
    <queries>["Reducing reliance on large-scale annotated data and closed-source models for planning in QA tasks","Automatic agent learning for QA","QA task planning with minimal human intervention", "Few-shot learning for QA"]</queries>
"""
    return prompt + output_format

def get_deep_check_idea_novel_search_query_prompt(idea,topic: str) -> str:
    prompt = f"""
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
    return prompt


def get_deep_rewrite_query_prompt(failed_query,topic):
    prompt = f"""
You are a master of search engine query writing. We want to utilize the literature search engine to find relevant paper.

The queries that have been used so far are as follows: {failed_query}. Unfortunately, no satisfactory answers were found. Please rewrite a query to help us locate the literature we need (do not repeat the failed query).

The topic you are studying is: {topic}.
Please provide a new search query to find the relevant papers. 

Try to make your query more concise and general so that it can be used to search for a wide range of papers.
If you failed more than 5 times, you can use a short query(no more than 5 words) to search for the paper.
Please output strictly in the following format:
<query>{{new query}}</query>

For example:
<query>Reducing reliance on large-scale annotated data and closed-source models for planning in QA tasks</query>
"""
    return prompt


def get_deep_reference_prompt(paper_content: str,topic) -> str:
    prompt = f"""
You are a scientific research expert, tasked with extracting and summarizing information from provided paper content relevant to the topic: {topic}. Your deliverables will include pertinent references, extracted entities, a detailed summary, and the experimental design.

The topic you are studying is: {topic}. (Ensure that the references are pertinent to this topic.)

Extraction Requirements:
Entities
1. Identify unique entities mentioned in the paper, such as model names, datasets, metrics, and specialized terminology.
2. Format the entities with a name followed by a brief description.
3. Ensure all entities are relevant to the specified topic ([topic]).


Summary Idea:
1. Background: Elaborate on the task's context and previous work, outlining the starting point of this paper.
2. Novelty: Describe the main innovations and contributions of this paper in comparison to prior work.
3. Contribution: Explain the primary methods used, detailing the theory and functions of each core component.
4. Detail Reason: Provide a thorough explanation of why the chosen methods are effective, including implementation details for further research.
5. Limitation: Discuss current shortcomings of the approach.

Experimental Content:
1. Experimental Process: Detail the entire experimental procedure, from dataset construction to specific steps, ensuring clarity and thoroughness.
2. Technical Details: Describe any specific technologies involved, providing detailed implementation processes.
3. Clarity of Plan: State your experimental plan concisely to facilitate understanding without unnecessary complexity.
4. Baseline: Elaborate on the baseline used, comparative methods, and experimental design, illustrating how these support and validate the conclusions drawn.
5. Verification: Explain how your experimental design assists in verifying the core idea and ensure it is detailed and feasible.

Relevance Criteria:
1. Method Relevance: References must directly correlate with the paper's methodology, indicating improvements or modifications.
2. Task Relevance: References should address the same task, even if methods differ, better have the same topic {topic}.
3. Baseline Relevance: References should serve as baselines for the methods discussed in the paper.
4. Output Format: Provide references without author names or publication years, formatted as titles only.
5. Specific paper titles will be placed between <References></References>. Based on the precise citation location and the corresponding ref_id in the paper, you need to infer the specific title of your output relevant references.


The paper content is as follows: 
{paper_content}


Please provide the entities, summary idea, experimental design, and the three most relevant references (Sort by relevance, with priority given to new ones with the same level of relevance, do not reference the original paper.) based on the paper's content.
Note: Ensure the references are pertinent to the topic you are studying: {topic}. If there are no relevant references, output <references>[]</references>.

Now please output strictly in the following format:
<entities>{{A list of entities you extract}}</entities>
<idea>{{Background: ... \nNovelty: ...\nContribution:...\nMethods:...\nDetail reason:...\nLimitation:...\n }}</idea>
<experiment>{{Step1:... Step2:...}}</experiment>
<references>["{{Title1}}", "{{Title2}}",  ...]</references>
"""
    return prompt


def get_deep_trend_idea_chains_prompt(idea_chains,entities,topic) -> str:
    entities = f"""
Here are the entities you need to know: {entities}
""" if use_entities else ""
    prompt = f"""
You are a scientific research expert tasked with summarizing the historical progression of research related to our current topic, based on the literature we have reviewed.

{entities}

The topic you are studying is: {topic}

The literature from early to late: {idea_chains}

Your objective is to outline the historical evolution of the research in light of current trends. Please follow these requirements:
Analysis of Published Viewpoints: Examine the progression of ideas across the identified papers. Detail how each paper transitions to the next—for instance, how Paper 0 leads to Paper 1, and so forth. Focus on understanding how Paper 1 builds upon the concepts in Paper 0. Elaborate on specific advancements made, including proposed modules, their designs, and the rationale behind their effectiveness in addressing previous challenges. Apply this analytical approach to each paper in the sequence.


Please present your findings in the following format:
<trend> {{The research trend you summarized based on the past work}} </trend>

Example:
<trend>from Paper 0 to Paper 1: ... \nfrom Paper 1 to Paper 2: ... \n </trend>
"""
    return prompt


def get_deep_judge_relevant_prompt(target_paper_title,target_paper_abstract,topic) -> str:
    prompt = f"""
You are an expert researcher tasked with evaluating whether a given paper is relevant to our research topic.

Below are the details of the paper you need to assess:
Title: {target_paper_title}
Abstract: {target_paper_abstract}

The topic is: {topic}

if the paper title and abstract are related to the topic, output <relevant>1</relevant>, otherwise output <relevant>0</relevant>. As long as you feel that this article has reference value for your question, you can use it to help you study the topic, it does not need to be completely consistent in topic.

Please output strictly in the following format(no extra content):
<think>{{your thinking steps}}</think>
<relevant>{{0/1}}</relevant>
    """
    return prompt


def get_deep_generate_future_direciton_prompt(idea_chains,trend,topic,entities) -> str:
    entities = f"""
Here are the entities you need to know: {entities}
""" if use_entities else ""
    prompt = f"""
You are a scientific research expert tasked with proposing future research directions based on the literature we have reviewed.

{entities}

The topic you are studying is: {topic}

The literature you have studied is as follows:
{idea_chains}

The following section delineates the progressive relationships among the previously summarized research papers:
<the begin of previous trend>{trend}</the end of previous trend>

Based on previous research, analyze how human experts think and transition from previous methods to subsequent approaches. Focus on their reasoning logic and the sources of their thought processes. Learn to emulate their reasoning patterns to further develop and guide your own research direction in a natural and coherent manner.

You are encouraged to adopt the following four modes of thinking:
1. Reflection: Reflect on scenarios where a specific method encounters significant challenges. Consider potential solutions that could effectively address these issues, make the solutions sounds reasonable, novel and amazing.
2. Analogy:  Identify a specific problem you are currently facing. Analyze whether similar issues exist in other fields and how those areas have addressed these problems. Consider if their solutions can be adapted and applied to resolve your current challenge.
3. Deep Dive: Some methods may present specific approaches to addressing a particular problem. Consider whether there are aspects that could be modified to enhance their rationale and effectiveness.
4. Imitate: Analyze the research methods used in previous studies and reflect on how they have progressively advanced the research. Identify common strategies or particularly useful approaches, and use these insights to guide the direction of your next research step. (Most recommended)

Note:Each article's limitations are specific to that particular piece and should not be applied to others. Carefully consider the task at hand and analyze the potential issues you might encounter if you proceed with your original approach, reflecting on the challenges previously faced. Then, think critically about how to address these issues effectively.

You are encouraged to apply human reasoning strategies to identify future research directions based on prior studies. Aim for in-depth analysis rather than mere integration of existing ideas. Please avoid introducing unfamiliar information, ensuring that the trends you present are both authentic and reasonable. Before proposing any trends, take a moment to reflect on the principles underlying the methods you're employing and assess their relevance to your research area. 

The future research direction should be related to the topic: {topic}.
Please output strictly in the following format:
<human>{{The human reasoning way you analyzed based on the previous research}}</human>
<future>{{the future research direction}}</future>
"""
    return prompt


def get_deep_generate_idea_prompt(idea_chains,trend,topic,entities,future = None,bad_case = []) -> str:
    bad_case_content = ""
    if len(bad_case) > 0:
        bad_case_content = "The following are examples of ideas you have proposed in the past that are similar to real papers. Please avoid this situation as much as possible. You can continue to make in-depth innovations, but avoid plagiarism:\n"
        for i,(paper,summary) in enumerate(bad_case):
            bad_case_content += f"<example>{i}. Your orig idea:{summary} \n Similar paper Title: {paper.title}\n Abstract: {paper.abstract}</example>\n"

    trend = f"""
The following section delineates the progressive relationships among the previously summarized research papers:
<the begin of previous trend>{trend}</the end of previous trend>
    """ if trend else ""

    future = f"""
The following section outlines the potential future research directions based on the literature you have studied:
<the begin of future>{future}</the end of future>
    """ if future else ""


    entities = f"""
Here are the entities you need to know: {entities}
""" if use_entities else ""
    prompt = f"""
You are a scientific expert tasked with formulating a novel and innovative research idea based on your comprehensive literature review. Your objective is to propose a feasible approach that could significantly advance the field.
        
{bad_case_content}

{entities}

The topic you are studying is: {topic}

The literature you have studied is as follows:
{idea_chains}

Task: Based on the current literature, propose a research idea that incorporates the following components:

Your idea is composed of the following components: 
Motivation:
1. Provide a background for your idea, summarizing relevant past work.
2. Identify shortcomings in previous research and highlight the specific problems that remain unsolved and that you aim to address.

Novelty:
1. Distinguish your proposed method from existing methods (preferably by naming specific approaches).
2. Detail the improvements your method brings compared to previous work.
3. Clearly outline at least three contributions your idea offers to the field, including the problems it resolves and the benefits it delivers.

Method: 
1. Present a detailed description of your idea, focusing on the core method, the specific problem it solves, and enhancements over earlier research (citing relevant literature with titles).
2. Explain the step-by-step methodology, including the functions of each module and the rationale for why this approach effectively addresses previous challenges.

Please adhere to the following guidelines:
1. Your research idea should be innovative, feasible, and contribute meaningfully to the field.Please carefully examine the idea you have proposed, avoid immediate perception, and try to be different from the previous methods as much as possible.
2. Ensure your proposal is solid, clearly defined, and practical to implement. Logic should underpin your reasoning.
3. Write in clear, concise language aimed at an audience with limited background knowledge in the subject. Avoid complex technical jargon, but when professional terms are necessary, provide thorough explanations.
4. Refrain from introducing concepts from uncertain fields to prevent proposing ideas that may be incorrect or impractical.
5. When referencing other research, please include the titles of the cited papers.
6. Please avoid introducing unfamiliar information, ensuring that the trends you present are both authentic and reasonable. Before proposing any trends, take a moment to reflect on the principles underlying the methods you're employing and assess their relevance to your research area.
7. Each article's limitations are specific to that particular piece and should not be applied to others. Carefully consider the task at hand and analyze the potential issues you might encounter if you proceed with your original approach, reflecting on the challenges previously faced. Then, think critically about how to address these issues effectively.

{trend}

{future}

Please output strictly in the following format:
<motivation>{{the motivation of your idea}}</motivation>
<novelty> {{the novelty of your idea}} </novelty>
<method> {{the method of your idea}} </method>
    """
    return prompt


def get_deep_final_idea_prompt(idea_chains,trend,idea,topic):
    idea = f"""
Here is your thinking steps:
{idea}
    """ if idea else ""
    if idea and trend:
        trend = f"""The relationship between each paper are as follows: {trend}"""
    elif trend:
        trend = f"""
The following section outlines the progress relationships between the previously summarized research papers:
<the begin of summarize>{trend}</the end of summarize>
        """
    else:
        trend = ""

    prompt = f"""
    You are an scientific expert with the primary objective of proposing a research idea based on the literature you have studied. Your goal is to propose a novel, feasible, and innovative research idea that can advance the field.

    The topic you are studying is: {topic}

Here are the literature you have studied:
{idea_chains}

Task: Based on the current literature, propose a research idea that incorporates the following components:

Please adhere to the following guidelines:
1. Your research idea should be innovative, feasible, and contribute meaningfully to the field. Please carefully examine the idea you have proposed, avoid immediate perception, and try to be different from the previous methods as much as possible
2. Ensure your proposal is solid, clearly defined, and practical to implement. Logic should underpin your reasoning.
3. Write in clear language aimed at an audience with limited background knowledge in the subject. Avoid complex technical jargon, but when professional terms are necessary, provide thorough explanations.
4. Refrain from introducing concepts from uncertain fields to prevent proposing ideas that may be incorrect or impractical.
When referencing other research, please include the titles of the cited papers.


{trend}

{idea}

The idea should be related to the topic: {topic}.

The final idea should contains the title, clearly explain the origins, motivation, novelty, difference of previous works and challenges of your idea, detailing how you overcame these hurdles. 
Please output strictly in the following format:
<final_idea> {{the final idea}} </final_idea>
"""
    return prompt

def get_deep_check_idea_novel_prompt(idea,papers):
    papers_content = ""
    for i,paper in enumerate(papers):
        papers_content += f"Paper {i}: Title:{paper.title}\n Abstract:{paper.abstract}\n"
    prompt = f"""
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
    return prompt


def get_deep_generate_experiment_prompt(idea,experiments,entities) -> str:
    prompt = f"""
You are a scientific expert tasked with designing rigorous, feasible, and impactful experiments based on specified scientific questions and the methodologies derived from the idea I provide, along with relevant past research. Your goal is to assist researchers in systematically testing hypotheses and validating innovative discoveries that could significantly advance their fields.

Past Related Research Experiments: {experiments}

Here are the entities you need to know: {entities}.

Here is the idea you need to design an experiment for: {idea}.

Please propose a detailed experimental plan addressing the following points:
1. Experimental Design: Develop rigorous experiments to ensure the reliability and validity of your results. Provide a comprehensive explanation of the baseline used, comparative methods, ablation study design, and criteria for data analysis and result evaluation. Clarify how these components collectively reinforce and validate the conclusions of your research. Structure your experimental design in a clear, logical, and step-by-step manner, ensuring each step is well-defined and easy to understand.
2. Implementation of Technologies/Methods: If your experimental design involves specific technologies or methodologies, describe the implementation process in detail, including key technical aspects. For any critical concepts utilized, provide thorough explanations. For instance, if you propose a modular approach, detail its construction, components, and functionality.
3. Feasibility Assessment: Ensure your experimental plan is realistic, considering technological availability, timelines, resources, and personnel. Identify potential challenges and propose strategies for addressing them.
4. References to Previous Studies: When citing related literature, include titles and pertinent details of the original papers. Strive to use as many references as necessary to support your experimental design.
5. Visual Aids: If useful, provide pseudocode or a flowchart to illustrate the implementation process. For example, you can use pseudocode to detail the core algorithm or the model architecture, or employ a flowchart to map out the experimental procedure and data flow.
6. Clarity of Language: Use straightforward language to describe your methods, assuming the reader may have limited knowledge of the subject matter. Avoid complex jargon and utilize accessible terminology. If professional terms are necessary, please provide clear and detailed explanations.


Please output strictly in the following format:
<experiment>{{your experimental plan}}</experiment>

For example:
<experiment> Step1: ... \n Step2: ..., ..., ... </experiment>
"""
    return prompt


def get_deep_refine_experiment_prompt(experiment,suggestions,paper_infos=None,entities = None) -> str:
    infos = f"""
The literature infos you maybe need to refer to are as follows: {paper_infos}
""" if paper_infos else ""
    
    prompt = f"""
You are a research expert tasked with refining and improving an experimental plan based on the feedback received.

{infos}

The experimental plan you proposed is as follows:
{experiment}

Please propose a detailed experimental plan addressing the following points:
1. Experimental Design: Develop rigorous experiments to ensure the reliability and validity of your results. Provide a comprehensive explanation of the baseline used, comparative methods, ablation study design, and criteria for data analysis and result evaluation. Clarify how these components collectively reinforce and validate the conclusions of your research. Structure your experimental design in a clear, logical, and step-by-step manner, ensuring each step is well-defined and easy to understand.
2. Implementation of Technologies/Methods: If your experimental design involves specific technologies or methodologies, describe the implementation process in detail, including key technical aspects. For any critical concepts utilized, provide thorough explanations. For instance, if you propose a modular approach, detail its construction, components, and functionality.
3. Feasibility Assessment: Ensure your experimental plan is realistic, considering technological availability, timelines, resources, and personnel. Identify potential challenges and propose strategies for addressing them.
4. References to Previous Studies: When citing related literature, include titles and pertinent details of the original papers. Strive to use as many references as necessary to support your experimental design.
5. Visual Aids: If useful, provide pseudocode or a flowchart to illustrate the implementation process. For example, you can use pseudocode to detail the core algorithm or the model architecture, or employ a flowchart to map out the experimental procedure and data flow.
6. Clarity of Language: Use straightforward language to describe your methods, assuming the reader may have limited knowledge of the subject matter. Avoid complex jargon and utilize accessible terminology. If professional terms are necessary, please provide clear and detailed explanations.

You have received the following suggestions for improvement:
{suggestions}

Please refine your experimental plan based on the feedback provided. Ensure your refined plan is feasible, clearly defined, and addresses the feedback you received.

Please output strictly in the following format:
<experiment>{{your refined experimental plan}}</experiment>
"""
    return prompt


def get_deep_refine_experiment_search_query_prompt(experiment,suggestions):
    prompt = f"""
You are a research expert tasked with refining and improving an experimental plan based on the feedback received.

The experimental plan you proposed is as follows:
{experiment}

You have received the following suggestions for improvement:
{suggestions}

Please decide whether you need to search for relevant papers to obtain relevant knowledge to improve your experiment.

If you need to search for relevant papers, please provide a  search query(only a conciese phrase) for literature search, else provide "".
For example: if suggestions say that the dynamic query additional information and update knowledge graph described in the experiment is not clearly described, so you need to output <query>dynamic knowledge graph update</query>(only a conciese phrase) .

Please output strictly in the following format:
<query>{{the search query}}</query>, or <query></query> if no search is needed.
    
For example:
<query>Reducing reliance on large-scale annotated data and closed-source models for planning in QA tasks</query>
"""
    return prompt

def get_deep_paper_info_prompt_for_refine_experiment(paper,experiment,suggestions) -> str:
    prompt = f"""
You are a scientific research expert.
Your task is to research the relevant literature to refine your experiment.

The literature you need to study is:
{paper}

The experiment designed for the idea is:
{experiment}

You have received the following suggestions for improvement:
{suggestions}

Please extract useful information from the paper that can help you improve your experiment.For example, if the paper describes a method or dataset or matric that can be used in your experiment, you should extract this method.

Please output strictly in the following format:
<info>{{The information you extracted from the paper}}</info>
"""
    return prompt

