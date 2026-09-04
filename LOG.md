## 2026-09-03 
** 📚 Learned - While i was inetgrating RAG in my project , i tried to build it from scratch to have a strong mental model and understanding how it works under the hood. Beacuse we are never required to write it from scratch we never get to understand it better , What is vector embedding , cosine similiarity , tokenization , word to vector and sentence embedding works.  
     Here is the leightweight engine I engineered to understand token tracking , spatial math and document retreival:)..      


## 2026-09-03 — Recap: the two months before this log existed

I've been building for a while already — this log starts today. Here's the compressed version of what got me here.

**🐛 Bug — the leak I almost shipped:** Early in the EV charging demand project, my forecasting models were scoring suspiciously well. Dug in and found a near-duplicate target column and a couple of leaky interaction features quietly giving the model information it shouldn't have had at prediction time. Rebuilt the feature set leakage-free (lag/rolling-window demand, cyclical time encoding) and re-benchmarked Linear Regression, Random Forest, and XGBoost against a naive baseline properly. The lesson stuck: a suspiciously good score is a bug report, not a win.

**🐛 Bug — when the LLM started making things up:** Building the resume ATS tool's bullet-rewriting feature (Gemini API), I noticed it would occasionally invent specific-sounding details for vague input bullets instead of flagging that they were vague. Built a separate vagueness-detection layer that routes thin bullets away from the rewriter entirely, rather than trusting the model to self-police.

**🐛 Bug — Docker containers that couldn't find each other:** Split the ATS tool into two containers (FastAPI backend, Streamlit frontend) to learn real multi-container patterns. Spent longer than I'd like admitting to on a networking issue that came down to one fact: inside a container, `localhost` means "this container," not "the other one." Containers reach each other by name on a shared Docker network.

**📚 Learned — APIs don't stay still:** Mid-build, `gemini-2.5-flash` got deprecated under me and I had to swap models and adjust. Small thing, but a real first taste of what "the dependency changed under you" actually feels like outside a tutorial.

**💡 Discovery — the thing I actually want to dig into next:** Catching the fabrication bug above got me genuinely curious about *why* LLMs do this — not just how to patch around it. That's part of why RAG-based semantic matching is next on my list instead of just more prompt tweaking: I want to understand retrieval as a way of constraining a model's answers, not just as a keyword-matching upgrade.

Full build details live in [ai-resume-ats-analyzer](https://github.com/baghelrashmi04/ai-resume-ats-analyzer) and [smart-ev-grid-optimization-demand-engine](https://github.com/baghelrashmi04/smart-ev-grid-optimization-demand-engine).
