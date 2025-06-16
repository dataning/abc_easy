Hello everyone! Welcome to our first session of GenAI office hour - I call this GenAI Dojo - because we're going to learn together, practice together, and grow together in the field of GenAI. Again, this is an effort from GenAI pillar in PAG to create a community of practice around Agentic AI. We’re planning our own teaching sessions (like today’s), inviting practioners (in PAG or outside PAG) to share their GenAI solution experiences, insights, and methodologies, and providing a space where you can ask questions — whether about structuring GenAI projects or validating models — and get answers from the community. 

Disclimaer: This is not a lecture (although it might feel one), but rather a discussion and more improtantly rapidly evolving area in the field of GenAI. This means that instead of saying we know everything in GenAI (which is almost impossible given the speed of evolution), we want to create a space where everyone feels comfortable sharing their experiences, asking questions, and learning from each other. In a way, stay humble and stay curious are the key success in GenAI and learn together. So, please feel free to share your thoughts, ask questions, and engage in the discussion.

Today, we're going to explore the concept of Agentic AI, its implications, and how we can leverage it to create more effective and efficient AI solutions. We're going to touch on the following topics: what Agentic AI is, its implications for AI development, its mechanics, agency vs control, memeory and their evals. 

---

# Hello, Agents. 

The way I think about Agentic AI is that it is a type of AI that can act autonomously, make decisions, and take actions based on its own understanding of the world. This contrasts with traditional AI systems, which are typically designed to perform specific tasks or follow predefined rules as single-shot generators.

Instead, agentic AI systems break down tasks into manageable subtasks, propose and refine intermediate solutions, and solicit or incorporate feedback at each stage - creating an iterative, collaborative workflow.

Gaming is a great example of Agentic AI in action. In games, agents can learn from their environment, adapt to new challenges, and make decisions based on their experiences. This is similar to how Agentic AI systems can learn from their interactions with the world and improve their performance over time.

In PAG, Consider how PAG teams collaborates with PMG and other stakeholders throughout each phase — receiving requests, scoping requirements, planning and development, testing and validation, and finally showcasing outcomes — maintaining open communication and feedback every step of the way, much like team leads, project managers, and executives collaborating from initial brief to final delivery.

---

# Still, why should we care about Agentic AI?

Agentic AI is important because it allows us to create more flexible and adaptable AI systems that can handle complex tasks and environments. By enabling AI systems to act autonomously, we can reduce the need for human intervention and oversight, leading to more efficient and effective solutions.

## Avoid dumping an unvetted 1,000-word report. Instead, each chunk gets checked, so the output is more accurate

Example: ESG fund analysis where Agent A drafts findings, Agent B fact-checks against Bloomberg data, Agent C validates regulatory compliance
• Self-correction loops: Agents review their own work before delivery
• Factual verification: Cross-reference claims against multiple sources
• Consistency checking: Ensure tone and messaging align throughout
• Error reduction: 40-60% fewer hallucinations vs single-shot generation

Think of it like a relay race: while the AI polishes the executive summary, you can research supporting data

• Example: Market brief on BlackRock's Q3 earnings—Agent A pulls financials, Agent B analyzes peer comparisons, Agent C drafts investment implications simultaneously
• Parallel work: Executive summary ready in 3 minutes while detailed analysis completes
• Progressive delivery: PMG gets draft insights to start client prep while technical deep-dive finishes
• Time savings: 2-hour analyst task completed in 25 minutes

You see the chain of reasoning — no more black-box outputs. It's easier to audit, trust, and debug

• Example: "I rated this fund 'High Risk' because: (1) 67% allocation to emerging markets, (2) Sharpe ratio of 0.34 vs benchmark 0.89, (3) correlation analysis shows 0.92 with volatile tech sector"
• Audit trail: Compliance can verify each data point and methodology
• Debug easily: "Ah, the correlation calculation used wrong time period—fix this step"
• Client trust: Show clients exactly how investment recommendations were derived

Instead of treating the AI as a magic text-generator, we engage it as a partner or a teammate

• Example: AI asks "I see conflicting dividend yields—should I use ex-dividend date or payment date for this analysis?"
• Context memory: "Based on your previous preference for conservative estimates, I'm using the lower volatility measure"
• Natural feedback: "Make this more client-friendly" → AI adjusts technical jargon automatically
• Partnership: Like having a junior analyst who learns your style and asks smart questions 


You can plug in specialist routines — legal review, style enforcement

• Example: Legal Agent automatically flags "This fund structure may trigger ERISA considerations for pension clients"
• Brand consistency: Style Agent ensures all client communications match PAG voice and formatting standards
• Risk specialist: Dedicated agent that spots concentration risk, liquidity issues, and regulatory violations
• Scale: One expert's knowledge becomes available to entire team—junior analysts get senior-level insights automatically

---

# Case: OCIO Endowment Research

🎓 The Core Problem:
"$43M annual excise tax burden = 860 full scholarships cut"

Harvard faces a $43 million yearly tax (likely the 1.4% excise tax on large university endowments)
To put this in perspective: $43M could fund 860 full scholarships (~$50K each)
This tax directly reduces money available for students
📊 Harvard's Response:
"Spend rate reduced 4.7% → 4.1% since 2019"

Harvard is spending less from its endowment to preserve capital
They've cut their annual spending rate from 4.7% to 4.1% of endowment value
This is a defensive move to maintain the endowment's long-term value
🔄 Ripple Effects:
"Tuition dependency +12% above peers"

Since Harvard is spending less from endowment, they're more dependent on tuition revenue
They're relying on tuition 12% more than similar universities
This could mean higher tuition or fewer scholarships
"Q3 lobbying spend likely +40%"

Harvard will probably increase political lobbying to fight this tax
Universities are pushing back against the excise tax policy
🚨 What to Watch:
"Monitor asset reallocation and bond issuance"

Harvard might restructure investments to minimize tax impact
They might issue bonds (debt) to fund operations instead of spending endowment
"Emergency spend rate cut probable if grant cuts exceed 15%"

If federal/state grants are cut by more than 15%, Harvard will likely cut spending even more dramatically
This would be a crisis response
💡 Bottom Line:
This tax is forcing Harvard to choose between paying taxes and funding student aid. The analysis predicts this will make higher education more expensive and less accessible, while universities fight back politically and financially.

---

Agentic AI has five key components that work together to create intelligent, collaborative systems.

First, Goal Decomposition. This breaks complex tasks into manageable subtasks. Why does this matter? It handles problems that are simply too big for single AI calls. Instead of asking an AI to "analyze the entire market," you break it down into data collection, trend analysis, risk assessment, and summary writing.

Second, Orchestration. This coordinates multiple agents working together. The importance here is ensuring agents collaborate instead of conflict. Think of it like conducting an orchestra—each agent has a role, and orchestration makes sure they play in harmony.

Third, Memory Systems. These provide persistent context across conversations. This matters because the system learns from past interactions and builds knowledge over time. Unlike traditional AI that forgets everything after each session, agentic systems remember your preferences, past decisions, and successful approaches.

Fourth, Dynamic Adaptation. This adjusts strategy based on intermediate results. The key benefit is self-correction when approaches aren't working. If one method fails, the system can pivot and try alternative approaches without human intervention.

Finally, Evaluation Systems. These provide continuous assessment and feedback throughout the process. This ensures outputs meet quality standards and improve over time. Rather than hoping for good results, evaluation systems actively monitor and enhance performance.

Together, these five components transform AI from a simple question-and-answer tool into an intelligent collaborator that can handle complex, multi-step workflows.

---

The evolution from single to multi-agent systems represents a fundamental shift in AI architecture.

In the first stage, single-agent task decomposition, we have a simple workflow: one decomposer breaks a task into subtasks, assigns them to different agents, and combines the results. This is like having a project manager who divides work among team members and compiles their outputs.

The second stage introduces memory and orchestration, adding sophisticated coordination mechanisms. Here, an orchestrator manages the entire process while a memory store captures context from each agent's work. A state manager tracks progress, and a reflection engine provides quality control with feedback loops. If quality review fails, the system iterates rather than producing subpar output.

The third stage shows advanced multi-agent workflows with specialized roles. Take the example of "What's today's top AI news?" The system deploys specialized agents for topic discovery, source selection, content fetching, deduplication, semantic ranking, and summarization. Each agent has a specific expertise, and they work together through shared memory stores and coordinated state management.

This evolution—from simple task splitting to memory-driven coordination to specialized agent ecosystems with feedback loops—demonstrates how agentic AI systems become increasingly sophisticated, moving from basic parallel processing to intelligent, adaptive collaboration networks that can handle complex, real-world workflows.

--- 

Agency vs. Control

The fundamental question in AI system design is: who's in charge? This section explores the critical difference between control-flow systems where you dictate every step, and agentic systems where AI makes autonomous decisions.

----

Control Flow vs Agentic - The Core Differences

Control-flow and agentic systems represent fundamentally different approaches to AI architecture. In control-flow systems, you decide what happens next at design time—the order of operations is hard-coded and predictable. The caller supplies every input and branch point, using familiar abstractions like loops, if-else statements, and workflow DAGs. This approach is highly deterministic: same inputs always produce the same path.

Agentic systems flip this model entirely. The agent decides what happens next at run-time, typically after reasoning with an LLM or planner. Goals come from within—the agent interprets high-level objectives, sets sub-goals, and may even rewrite its own plan. Instead of explicit function calls, you have autonomous agents with toolboxes that negotiate with each other using plan-act-observe loops and dynamic tool selection.

The trade-offs are clear: control-flow excels at ETL pipelines, billing, and compliance tasks where regulatory requirements demand predictability. Agentic systems shine in research assistance, dynamic data aggregation, and user-facing chatbots that need to improvise.


----

Control Flow vs Agentic - Code Examples

The code difference illustrates this philosophical divide perfectly. Control-flow code shows a deterministic pipeline: clean data, extract features, load model, predict, postprocess—each step explicitly called in sequence. There's no ambiguity about what happens when.

Agentic code defines an objective and gives the agent tools to accomplish it. You specify "write a one-page market brief on solar-powered IoT devices," provide tools like web search and summarization, enable reflection, and let the agent figure out how to achieve the goal. The agent might search for market data, analyze trends, gather expert opinions, or take completely different approaches you never anticipated.


----

Quick Check: Control Flow or Agentic?

Here's a simple diagnostic to determine your system's nature. If you know exactly which function will be called second, you're in control-flow territory. If an LLM could decide whether to call the database or vector store, that's agentic. Can your system set brand-new sub-goals you didn't anticipate? That's agentic autonomy. Can you write unit tests that walk every execution path? You're back to control-flow predictability.

The fundamental trade-off is between control and freedom—predictable paths versus adaptive decisions. This choice shapes your entire architecture and determines whether you're building a deterministic pipeline or an intelligent collaborator.

----

Memory Systems

Memory transforms AI from forgetful responders into learning partners. This section explores how different memory architectures enable AI systems to build knowledge, maintain context, and improve over time.

----

Memory: From Scratch to Strategy

Memory represents the evolution from reactive to intelligent AI systems. Traditional GenAI operates like pure brain power—brilliant in the moment but forgetting everything between sessions. You restart every time and lose all context.

The first evolution adds notepad-style memory—basic recall of past interactions that builds continuity. Agentic AI Level 1 can remember previous conversations and build on history, like bringing notes to every meeting.

Advanced agentic systems operate like having a laptop with searchable archives. They can cross-reference data, search through extensive conversation histories, and draw insights from accumulated knowledge. The progression is clear: forgettable to continuous to insightful.

As we say: notebooks win meetings, but AI memory wins workflows. The ability to maintain persistent context across interactions transforms AI from a tool into a learning partner.

----

Memory Evolution - Traditional vs Advanced

Traditional memory systems rely on simple storage mechanisms. Langchain-style semantic memory bakes facts into prompts—useful but fixed once deployed. Episodic memory captures time-stamped events but gets overwritten when context windows fill. Procedural memory handles rules and instructions but requires redeployment for any changes.

MemGPT architecture revolutionizes this approach with sophisticated tiers. Core memory sits inside the context window, storing user profiles and personas that agents can self-edit. Chat history captures recent interactions with automatic summarization. Recall memory provides unlimited external storage for every conversation turn. Archival memory handles documents and references with full read-write access. Memory statistics guide intelligent retrieval decisions.

This architecture enables unlimited persistence, self-editing capabilities, and intelligent memory management—transforming AI from session-limited tools into continuously learning systems.

----

Memory Types in Action

Consider an executive assistant agent managing business intelligence. It integrates three memory types seamlessly: semantic memory stores executive priorities, team expertise, and strategic initiatives—the foundational knowledge. Episodic memory captures past meeting outcomes, decision patterns, and follow-up actions—the historical context. Procedural memory maintains meeting protocols, briefing formats, and escalation rules—the operational framework.

This creates executive intelligence comparable to an experienced EA who knows what each executive cares about, recalls past decisions, and follows established protocols. The technical implementation shows how core memory handles persona and user profiles, recall memory manages conversation history, and archival memory provides unlimited document storage—all working together to create persistent, evolving intelligence.


----

Evaluations

Evaluation systems ensure AI outputs meet quality standards and improve over time. This section covers how to systematically measure LLM pipeline quality beyond simple metrics.

----

What Are Evals?

Evaluations represent the systematic measurement of LLM pipeline quality. Good evaluation produces results that can be easily and unambiguously interpreted, going far beyond single accuracy numbers.

Evals cover three main areas: classification and analysis tasks like sentiment analysis and entity recognition; information extraction including data retrieval and pattern recognition; and content synthesis covering summarization and narrative construction. Each requires different evaluation approaches and metrics to ensure quality and reliability.

----

Types of Evaluations

Evaluation strategies fall into two main categories. Reference-based metrics compare LLM output against known ground-truth answers—like having official answers for multiple-choice tests. These include simple comparisons checking keyword presence and text similarity, plus execution-based tests running generated code against test cases. They're preferred when feasible because they're cheap to maintain and verify.

Reference-free metrics evaluate based on inherent properties without golden answers. These focus on rule compliance—ensuring chatbots don't give medical advice or code includes explanatory comments. Quality assessment uses LLM-as-judge for subjective evaluation and toxicity detection for harmful content. These are crucial for creative or multi-valid responses where no single correct answer exists.

The cost hierarchy is clear: start with cheap code-based checks, then reserve expensive LLM-as-judge evaluation for persistent failures that simple rules can't capture.

----

Evaluation Dilemmas: Yale Case Study

Real-world evaluation challenges emerge in the Yale University secondary PE sales case. Three critical dilemmas illustrate why good metrics must map to real-world impact.

Human label agreement shows the challenge: expert evaluators rated the situation negatively while the AI model scored it positively. When 67% of experts see distress but AI remains optimistic, you need higher expert consensus thresholds—requiring 80% or more agreement for confident labeling.

Eval versus human disagreement reveals institutional knowledge gaps: the LLM scored the analysis 8.7 out of 10 for thoroughness, but Yale's CIO rated it 3 out of 10 for missing the crisis context. The AI ignored 30+ years of precedent, suggesting evaluation systems must weight institutional memory more heavily.

The third dilemma shows how good technical metrics can miss business impact: AI achieved 96% accuracy on data monitoring, but peer confidence dropped 25%. The system missed ecosystem effects that matter more than technical precision. The fix is optimizing for industry impact rather than isolated accuracy metrics.

These cases demonstrate that evaluation systems must account for human expertise, institutional knowledge, and real-world consequences—not just technical performance measures.

----

End

Thank you for joining this exploration of agentic AI. The future of AI isn't just about better models—it's about creating systems that think, plan, remember, and collaborate. The journey from simple prompt-response to intelligent partnership has just begun.