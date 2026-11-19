"""Assemble report.tex from its section files (run before pdflatex)."""
from pathlib import Path

HERE = Path(__file__).resolve().parent

ABSTRACT = r"""
We build and rigorously evaluate an agentic question-answering system over the Cleantech Media Dataset
(20{,}111 articles). A Part~1 baseline, a LangGraph ReAct agent with a Chroma retriever and a summarizer, is
extended with six components: a from-scratch BM25 fused with dense retrieval by Reciprocal Rank Fusion and
cross-encoder reranking; tools served over the Model Context Protocol with a hand-written client; a
planner/parallel-researcher/writer/critic crew with conversation memory; input and output guardrails; ASD-STE100
answers with a rule-based compliance checker; and OpenAlex scholarly search. On 50 multi-document questions,
scored with retrieval metrics and a cross-family LLM judge (GPT-5.5, test-retest $\kappa=0.82$), retrieval
raises key-fact recall from 0.47 to 0.67, and BM25 beats dense embeddings (MRR 0.64 vs 0.54). The crew finds
17\% more of the referenced articles, yet a single ReAct agent writes equally good answers at one eighth of the
cost. STE answers triple readability but lose about 8\% of the reference facts, and the critic loop has no
significant effect. Ablations, an error-analysis-driven second version and bootstrap confidence intervals
make each of these trade-offs measurable.
"""

REFLECTION = r"""
\emph{[Draft written from the project log; edit it so it is in your own words.]}
This project tied the whole course together. Cleaning the corpus was HW1 again: the duplicated ``blob''
paragraphs and tokenised apostrophes would have quietly corrupted every downstream number had I not looked at
raw rows first. The fact that BM25 beat a modern embedding model echoed HW2 and HW4: distributional vectors capture
similarity, but counting words still wins when the signal is a rare proper noun. HW3's RAG pipeline became
the Part~1 baseline, and HW5's lesson that ROUGE barely separates good summaries from mediocre ones is why the
LLM judge, not overlap metrics, is the primary measure here.

The most valuable moment was a negative result. My first multi-agent crew, the part I was most excited about,
scored clearly \emph{below} the simple baseline. Tracing its failures turned up two bugs in my own evaluation
code, which had penalised the crew, as well as real design flaws. Fixing them improved the crew, but the honest
conclusion is still that a single strong agent is hard to beat on this benchmark. I learned to budget as
much effort for measurement as for building, to run ablations that change exactly one thing, and to report cost
and latency next to accuracy, because an 8$\times$ more expensive system has to justify itself. Building the MCP
server and the parallel LangGraph crew also taught me practical engineering lessons that no tutorial covers: GPU
thread safety, async task ownership in Jupyter, and caching every expensive call so that results are reproducible.

Overall, the course moved from text processing and $n$-gram models to transformers, summarization and
retrieval-augmented agents, and each assignment supplied a tool or a lesson that this project needed. I leave it
far more skeptical of impressive-looking demos and more confident that I can evaluate NLP systems carefully.
"""

tex = (HERE / "report.tex").read_text()
body = (HERE / "sec_methods.tex").read_text() + "\n" + (HERE / "sec_results.tex").read_text()
body = body.replace("%%REFLECTION%%", REFLECTION.strip())
out = tex.replace("%%ABSTRACT%%", ABSTRACT.strip()).replace("%%BODY%%", body)
(HERE / "CleanTech-Agent-Report-AsadKhan.tex").write_text(out)
print("assembled")
