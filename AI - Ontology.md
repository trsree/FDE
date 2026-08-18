# AI Ontology

## Table of Contents
1. [Generative AI](#generative-ai)
2. [Prompt Engineering](#prompt-engineering)
3. [AI Agent](#ai-agent)
4. [RAG (Retrieval-Augmented Generation)](#rag-retrieval-augmented-generation)
5. [Context Window](#context-window)
6. [Multi-turn Conversations](#multi-turn-conversations)
7. [Context Compaction](#context-compaction)
8. [Agentic AI Design](#agentic-ai-design)
9. [Agentic AI Workflow Patterns](#agentic-ai-workflow-patterns)
10. [LangChain](#langchain)
11. [LangGraph](#langgraph)

---

## Generative AI

### Definition
Generative AI refers to artificial intelligence systems capable of creating new content by learning patterns from training data and generating novel outputs in various formats (text, images, code, audio, video, etc.).

### Explanation
Generative AI models use deep learning architectures, particularly transformer-based models, to understand input patterns and produce coherent, contextually relevant output. Unlike discriminative models that classify or predict from existing data, generative models can synthesize entirely new data that resembles the training distribution.

### Key Characteristics
- **Learning from Patterns**: Trained on large datasets to internalize linguistic, visual, or domain-specific patterns
- **Token-based Generation**: Generates output sequentially, one token/element at a time
- **Probabilistic**: Uses probability distributions to select the next likely token
- **Versatility**: Can perform multiple tasks without task-specific training (few-shot, zero-shot capabilities)

### Examples
- Large Language Models (LLMs): GPT-4, Claude, Gemini
- Image Generation: DALL-E, Stable Diffusion, Midjourney
- Code Generation: GitHub Copilot, Amazon CodeWhisperer
- Music Generation: Jukebox, MuseNet

### Applications
- Content creation and copywriting
- Code generation and development assistance
- Customer service automation
- Data augmentation and synthesis
- Creative storytelling and ideation

---

## Prompt Engineering

### Definition
Prompt engineering is the practice of designing and optimizing text inputs (prompts) to guide AI models toward producing desired outputs with better accuracy, relevance, and quality.

### Explanation
Prompt engineering bridges the gap between user intent and model behavior. By carefully structuring queries, providing context, examples, and clear instructions, users can significantly improve the quality of AI-generated responses without retraining models.

### Core Techniques

#### 1. **Clear Instructions**
Specify exactly what you want the model to do with unambiguous language.
```
Good: "Summarize this article in 3 bullet points focusing on business impact"
Poor: "Tell me about this"
```

#### 2. **Context Provision**
Supply relevant background information to help the model understand the domain and constraints.
```
"You are an experienced data scientist. Explain this ML concept to a business executive."
```

#### 3. **Few-shot Prompting**
Provide examples of the desired input-output pattern.
```
Example 1: Input: "Hello" → Output: "Greeting detected"
Example 2: Input: "How are you?" → Output: "Question detected"
Now classify: "What time is it?"
```

#### 4. **Chain-of-Thought (CoT)**
Ask the model to reason step-by-step before providing final answers.
```
"Think through this problem step by step before giving your answer."
```

#### 5. **Role-based Prompting**
Assign a specific persona or role to the model.
```
"Act as a senior software architect and review this design."
```

### Best Practices
- Be specific and concise
- Include relevant constraints (length, format, tone)
- Use delimiter tokens (''', """, ###) to separate sections
- Iterate and refine based on outputs
- Test edge cases and variations

### Impact
- Reduces computational costs by improving first-pass quality
- Increases model reliability and consistency
- Enables customization without fine-tuning
- Democratizes AI usage for non-technical users

---

## AI Agent

### Definition
An AI Agent is an autonomous software system that perceives its environment through sensors/APIs, makes decisions using AI models, and takes actions to achieve predefined goals with minimal human intervention.

### Explanation
Unlike static AI systems that respond to single queries, agents operate in a loop: observing environmental state, reasoning about options, executing actions, and learning from outcomes. Agents combine multiple capabilities—language understanding, planning, tool use, memory, and error correction—to accomplish complex, multi-step objectives.

### Key Components

#### 1. **Perception**
- Sensory input from environment (APIs, databases, user input)
- Ability to interpret current state

#### 2. **Reasoning Engine**
- LLM or decision-making model
- Planning and strategy formulation
- Risk assessment

#### 3. **Action/Tools**
- API calls and function execution
- Database operations
- External service integration
- Code execution

#### 4. **Memory**
- Short-term: Current conversation/task context
- Long-term: Persistent knowledge base
- Experience/learning from past actions

#### 5. **Feedback Loop**
- Action result observation
- Error detection and correction
- Continuous improvement

### Agent Autonomy Levels

| Level | Description | Example |
|-------|-------------|---------|
| **Reactive** | Immediate response to stimuli | Chatbot responding to user input |
| **Deliberative** | Plans actions based on goals | Code generation with error checking |
| **Learning** | Adapts behavior from experience | Recommendation engines improving over time |
| **Autonomous** | Operates with minimal intervention | Self-driving vehicles, trading bots |

### Examples
- **Customer Service Agents**: Resolving issues by accessing databases and escalating as needed
- **Coding Agents**: Writing, testing, and debugging code autonomously
- **Research Agents**: Gathering information, summarizing, and synthesizing findings
- **Planning Agents**: Creating schedules and managing workflows
- **Data Analysis Agents**: Exploring datasets and generating insights

### Challenges
- Hallucination and factual errors
- Knowing when to ask for human help
- Error recovery and failure modes
- Controlling scope and preventing unintended actions

---

## RAG (Retrieval-Augmented Generation)

### Definition
RAG is a hybrid AI technique that combines information retrieval from external knowledge bases with generative AI to produce more accurate, current, and contextually grounded responses.

### Explanation
Rather than relying solely on an LLM's training data (which becomes outdated), RAG systems retrieve relevant documents or data from vector databases, knowledge bases, or search engines, then feed this retrieved context into the generative model. This ensures responses are grounded in current information and domain-specific knowledge.

### Architecture

```
User Query
    ↓
[Retrieval Module] → Search Knowledge Base → Relevant Documents/Chunks
    ↓
[Augmentation] → Combine Query + Retrieved Context
    ↓
[Generation Module] → LLM Generates Response using Augmented Input
    ↓
Final Response (Grounded + Current + Relevant)
```

### Key Components

#### 1. **Retrieval System**
- **Vector Database**: Stores embeddings of documents (Pinecone, Weaviate, Milvus)
- **Search Algorithm**: Similarity-based or keyword-based retrieval
- **Ranking**: Prioritizes most relevant documents

#### 2. **Document Processing**
- Chunking: Breaking documents into manageable pieces
- Embedding: Converting text to vector representations
- Indexing: Organizing for efficient retrieval

#### 3. **Augmentation**
- Formatting retrieved context for the LLM
- Managing context window constraints
- Relevance filtering

#### 4. **Generation**
- LLM uses retrieved context as reference
- Generates response with citations/sources
- Synthesizes multiple retrieved documents

### Workflow Example
```
User: "What are the latest features in Claude 5?"

1. Retrieve: Search knowledge base for "Claude 5 features"
   → Find: Documentation, release notes, blog posts

2. Augment: Combine query with retrieved documents
   Input: "Based on these documents: [doc1], [doc2], [doc3]
           Answer: What are the latest features in Claude 5?"

3. Generate: LLM produces response grounded in retrieved information
   Output: "Claude 5 includes X, Y, Z features (from source: [link])"
```

### Advantages
- **Accuracy**: Grounded in verified sources
- **Currency**: Uses up-to-date information
- **Transparency**: Can cite sources
- **Reduced Hallucinations**: Less likely to fabricate facts
- **Domain Expertise**: Can integrate proprietary knowledge

### Applications
- Question answering systems
- Customer support automation
- Document summarization
- Legal and compliance assistance
- Medical diagnosis support
- Technical documentation assistance

### Challenges
- Retrieval quality directly impacts output quality
- Requires maintaining current knowledge bases
- Managing large context windows efficiently
- Handling conflicting information across sources

---

## Context Window

### Definition
A context window is the maximum amount of text (measured in tokens) that an LLM can process and consider in a single request, including the user's input, system instructions, and previous conversation history.

### Explanation
Language models process text sequentially using a fixed-size window of attention. This window determines how much information the model can "see" and reference when generating responses. Larger context windows enable processing of longer documents, richer conversation history, and more complex tasks.

### Context Window Composition

```
┌─────────────────────────────────────┐
│       Total Context Window          │
├─────────────────────────────────────┤
│  System Prompt/Instructions (tokens)│
├─────────────────────────────────────┤
│  Conversation History (tokens)      │
├─────────────────────────────────────┤
│  Current User Input (tokens)        │
├─────────────────────────────────────┤
│  Available for Response (tokens)    │
└─────────────────────────────────────┘
```

### Model Context Windows

| Model | Context Window |
|-------|-----------------|
| Claude 3.5 Sonnet | 200,000 tokens (~150K words) |
| Claude 3 Opus | 200,000 tokens (~150K words) |
| GPT-4 Turbo | 128,000 tokens |
| GPT-4 | 8,000 - 32,000 tokens |
| Haiku | 200,000 tokens |

### Token Counting
- Average word ≈ 1.3 tokens
- Average page ≈ 500-750 tokens
- Average code file ≈ 2,000-5,000 tokens

### Practical Implications

#### What Fits in Context?
- **Large**: Entire books (100K+ pages)
- **Medium**: Multiple documents + conversation history
- **Small**: Single document + recent history

#### Optimization Strategies
- Summarize old conversation turns
- Retrieve relevant information (RAG)
- Compress repetitive content
- Remove unnecessary formatting

### Limitations
- **Finite Resource**: Fixed maximum prevents processing unlimited data
- **Cost**: Input tokens incur processing costs
- **Latency**: Larger contexts require more computation
- **Attention Degradation**: Some models show reduced attention to early tokens in very long contexts

### Usage Patterns

1. **Single Document Analysis**: Load entire document in context
   ```
   Analyze this annual report [200-page PDF fits in context]
   ```

2. **Document Comparison**: Multiple documents in single request
   ```
   Compare these 5 research papers [all fit in context]
   ```

3. **Conversation with History**: Maintain full chat history
   ```
   [System + 50 previous turns + current message = within context]
   ```

4. **Code Review**: Analyze multiple code files together
   ```
   Review these 3 files and their dependencies
   ```

---

## Multi-turn Conversations

### Definition
Multi-turn conversations refer to extended dialogues where the AI maintains context across multiple exchanges, allowing for iterative refinement, follow-up questions, and building understanding over a series of interactions.

### Explanation
Multi-turn conversations enable a stateful, iterative interaction pattern where each turn builds on previous exchanges. The model references earlier messages to provide coherent, contextually-aware responses, enabling users to clarify, build upon, and refine ideas over multiple interactions.

### Conversation Structure

```
Turn 1: User Query 1
        ↓
        AI Response 1 (stored in context)
        
Turn 2: User Query 2 (references Turn 1)
        ↓
        AI Response 2 (maintains history)
        
Turn 3: User Query 3 (references Turns 1-2)
        ↓
        AI Response 3 (full conversation available)
        
...continues
```

### Key Characteristics

#### 1. **Context Preservation**
Model remembers all previous messages in conversation
```
Turn 1: "I'm building a web app with React"
Turn 2: "How do I handle state?" → Model knows it's about React
Turn 3: "What about async operations?" → Continues React context
```

#### 2. **Iterative Refinement**
Users can clarify and build on responses
```
Turn 1: Generate code outline
Turn 2: "Make it more performant"
Turn 3: "Add error handling"
Turn 4: "Format according to our style guide"
```

#### 3. **Anaphora Resolution**
Model resolves pronouns and references to previous context
```
Turn 1: User mentions "API authentication"
Turn 2: User: "How do I test it?" → "it" = API authentication
```

#### 4. **Conversation State**
System maintains understanding of:
- Topic and subtopic hierarchy
- User preferences and style
- Decisions made in earlier turns
- Constraints established

### Common Patterns

#### **Exploratory Dialogue**
```
T1: "Explain machine learning basics"
T2: "What's the difference between supervised and unsupervised?"
T3: "Can you give more examples of supervised learning?"
T4: "How do I choose between different algorithms?"
```

#### **Debugging/Problem Solving**
```
T1: "My code is throwing this error"
T2: "I tried X, now I get this error"
T3: "Your suggestion didn't work because..."
T4: "Can you provide a complete example?"
```

#### **Creative Collaboration**
```
T1: "Write me a poem about nature"
T2: "Make it more humorous"
T3: "Can you make the rhyme scheme AABB?"
T4: "Perfect! Now write one about technology in the same style"
```

#### **Learning/Teaching**
```
T1: "What is quantum computing?"
T2: "Explain entanglement in simpler terms"
T3: "How does this help quantum computers solve problems?"
T4: "What's an example of a problem quantum computers excel at?"
```

### Technical Implementation

#### **Message History Management**
```json
{
  "messages": [
    {"role": "user", "content": "First question"},
    {"role": "assistant", "content": "First response"},
    {"role": "user", "content": "Follow-up question"},
    {"role": "assistant", "content": "Follow-up response"},
    {"role": "user", "content": "Current question"}
  ]
}
```

#### **Context Budget**
- Each previous message consumes tokens
- Older messages may be summarized/dropped if approaching limit
- Priority given to recent exchanges and system instructions

### Advantages
- **Better Understanding**: Model can ask clarifying questions
- **Iterative Improvement**: Refine results over multiple turns
- **Efficiency**: Build on previous responses rather than restarting
- **Natural Interaction**: More human-like dialogue flow
- **Complex Tasks**: Decompose large problems into steps

### Challenges
- **Token Consumption**: History grows, using more context tokens
- **Forgotten Context**: Very long conversations may lose early context
- **Inconsistency**: Model might contradict earlier statements
- **Error Accumulation**: Mistakes compound across turns

---

## Context Compaction

### Definition
Context compaction is a technique to reduce the token consumption of conversation history or long documents while preserving essential information, enabling longer conversations within fixed context windows.

### Explanation
As multi-turn conversations grow, the accumulated message history consumes increasing tokens, leaving fewer tokens for new input and output. Context compaction intelligently reduces this overhead through summarization, filtering, and hierarchical compression while maintaining semantic coherence and necessary details.

### Compaction Strategies

#### 1. **Summarization**
Replace verbose exchanges with concise summaries
```
Original (500 tokens):
T1: User asks detailed question about Python async programming
T2: Assistant provides long explanation with examples
T3: User asks follow-up about event loops
T4: Assistant explains with code examples

Compacted (100 tokens):
"User asked about Python async programming and event loops. 
Key points covered: asyncio fundamentals, event loop mechanics, practical patterns."
```

#### 2. **Filtering/Pruning**
Remove less important messages while keeping critical information
```
Keep:
- System instructions
- Recent user goals
- Key decisions made
- Important constraints

Remove:
- Intermediate debugging steps
- Repeated questions
- Clarifications already incorporated
- Obsolete context
```

#### 3. **Hierarchical Abstraction**
Organize conversation into levels of detail
```
Level 1 (Essential): User goals and key decisions
Level 2 (Important): Major sub-topics and approaches
Level 3 (Detail): Specific examples and code snippets
```

#### 4. **Progressive Summarization**
Create "checkpoint" summaries at intervals
```
Messages 1-10: Detail
Summary: Condensed overview
Messages 11-20: Detail
Summary: Condensed overview
Messages 21-current: Detail (most recent, preserved)
```

#### 5. **Selective History**
Maintain only messages relevant to current task
```
Current goal: "Generate database schema"
Relevant context: Earlier discussion of requirements
Dropped context: Earlier troubleshooting about deployment
```

### Implementation Example

```
Before Compaction (1500 tokens):
[System prompt: 100 tokens]
[Message 1-20: 700 tokens - initial exploration]
[Message 21-40: 600 tokens - refinement and debugging]
[Message 41-current: 100 tokens]

After Compaction (400 tokens):
[System prompt: 100 tokens]
[Conversation Summary: 200 tokens - "Explored database design options, 
  decided on PostgreSQL with this schema, resolved N+1 query issues"]
[Message 41-current: 100 tokens]

Saved: 1100 tokens (73% reduction)
Preserved: Essential context and recent work
```

### When to Use Compaction

| Scenario | Trigger |
|----------|---------|
| **Long Research** | 20+ turns on complex topic |
| **Debugging Session** | Multiple failed attempts documented |
| **Iterative Refinement** | 30+ rounds of polish and revision |
| **Memory Constraints** | Approaching 80% of context window |
| **Agent Operations** | Long-running agentic workflows |

### Automatic vs. Manual

#### Automatic Compaction
- LLM summarizes on trigger
- Transparent to user
- Risk of losing important details
- Implementation: Check token count, auto-summarize if >80%

#### Manual Compaction
- User explicitly requests "summarize our discussion"
- More control and transparency
- Better preservation of details
- Requires user awareness and action

### Tools and Implementations
- **LLM-based**: Ask Claude to summarize conversation
- **Semantic**: Group messages by topic, keep representatives
- **Extractive**: Pull key sentences from messages
- **Hierarchical**: Maintain conversation tree, compress branches

### Trade-offs

| Benefit | Cost |
|---------|------|
| More tokens for new content | Possible information loss |
| Longer conversations possible | Reduced historical detail |
| Lower API costs | Less precise reference |
| Faster processing | Potential for contradictions |

---

## Agentic AI Design

### Definition
Agentic AI Design is a systems approach to building AI applications where autonomous agents execute goal-directed behavior using reasoning, planning, memory, and tool-use capabilities, rather than simple input-output mappings.

### Explanation
Traditional AI systems follow a linear path: input → process → output. Agentic AI design introduces intelligence at each stage: understanding goals, planning approaches, deciding when to use tools, learning from outcomes, and adapting strategies. It shifts from "answering questions" to "accomplishing tasks."

### Core Design Principles

#### 1. **Autonomy with Guardrails**
- Agent operates independently within constraints
- Can make decisions and take actions without per-action approval
- Predefined boundaries prevent harmful actions

```
Autonomous: "Schedule my meetings for this week"
(Agent decides time, location, attendees)

Not Autonomous: "I'll only write files if you approve each one"
(Requires human intervention for each action)
```

#### 2. **Explicit Goal Representation**
- Clear, measurable objectives
- Hierarchical sub-goals
- Success criteria defined upfront

```
Primary Goal: "Improve customer satisfaction score by 15%"
Sub-goals:
- Reduce support ticket response time
- Increase first-contact resolution rate
- Implement customer feedback mechanisms
```

#### 3. **Tool-Augmented Reasoning**
- Agent uses domain-specific tools effectively
- Knows when and how to use available tools
- Understands tool limitations

```
Available Tools:
- Database queries
- API calls
- Code execution
- External search
- Calculation engines

Agent decides: "I need current data, so I'll query the database,
then process results using Python, then search for context."
```

#### 4. **Iterative Refinement**
- Agent learns from tool outputs
- Adjusts strategy based on results
- Self-corrects and retries

```
Try 1: Execute query → Returns error
Iteration 1: Analyze error, modify query
Try 2: Execute modified query → Returns results
Iteration 2: Validate results against expected range
Try 3: Use results for final response
```

#### 5. **Memory and Learning**
- Maintains state across interactions
- Learns from past successes and failures
- Applies learnings to new situations

```
Memory Types:
- Task-specific: Current problem context
- Session: Patterns from this session
- Long-term: Patterns across many sessions
```

### Architecture Components

```
┌─────────────────────────────────────────┐
│           AGENTIC AI SYSTEM             │
├─────────────────────────────────────────┤
│                                         │
│  ┌─────────────────────────────────┐   │
│  │    Goal & Strategy Layer        │   │
│  │  - Understand objectives        │   │
│  │  - Plan approach                │   │
│  │  - Set success criteria         │   │
│  └─────────────────────────────────┘   │
│                 ↓                       │
│  ┌─────────────────────────────────┐   │
│  │    Reasoning Engine             │   │
│  │  - Analyze current state        │   │
│  │  - Generate options             │   │
│  │  - Evaluate trade-offs          │   │
│  └─────────────────────────────────┘   │
│                 ↓                       │
│  ┌─────────────────────────────────┐   │
│  │    Decision & Action Layer      │   │
│  │  - Select best option           │   │
│  │  - Execute tool calls           │   │
│  │  - Handle errors                │   │
│  └─────────────────────────────────┘   │
│                 ↓                       │
│  ┌─────────────────────────────────┐   │
│  │    Memory & Learning Layer      │   │
│  │  - Record outcomes              │   │
│  │  - Update strategies            │   │
│  │  - Persist insights             │   │
│  └─────────────────────────────────┘   │
│                                         │
└─────────────────────────────────────────┘
         ↓              ↑
    Tools & APIs    Environment
```

### Design Considerations

#### **Scope Definition**
- What decisions can the agent make alone?
- What requires human approval?
- What are hard boundaries?

```
Agent CAN: Query databases, format data, write code, run tests
Agent CANNOT: Deploy to production, delete data, modify permissions
Agent MUST ASK: Before significant infrastructure changes
```

#### **Error Handling**
- Expected failure modes
- Recovery strategies
- Escalation paths

```
Error 1: API timeout → Retry with backoff
Error 2: Invalid input → Regenerate with constraints
Error 3: Permission denied → Escalate to human
```

#### **Safety and Control**
- Sandbox isolation
- Rate limiting
- Audit trails
- Kill switches

#### **Transparency**
- Explain reasoning
- Document tool choices
- Show step-by-step progress
- Provide reasoning justifications

### Common Pitfalls
- **Over-delegation**: Giving agent too much autonomy
- **Poor Tooling**: Agent can't accomplish task with available tools
- **Vague Goals**: Unclear success criteria lead to wrong solutions
- **No Guardrails**: Uncontrolled agent causes unintended changes
- **Silent Failures**: Agent fails without notifying users

---

## Agentic AI Workflow Patterns

### Definition
Agentic AI Workflow Patterns are recurring architectural templates and interaction patterns that define how AI agents process information, make decisions, and execute tasks in common scenarios.

### Explanation
Just as software design patterns solve recurring problems, agentic workflow patterns provide proven solutions to common agent challenges. These patterns define the flow of information, decision points, tool usage, and error handling across different types of tasks.

### Core Workflow Patterns

---

### Pattern 1: **Perception-Reasoning-Action (PRA)**

The fundamental agent loop where agent observes, thinks, then acts.

```
Perception Phase:
  1. Observe environment state
  2. Gather relevant information
  3. Identify constraints and resources
  
         ↓
         
Reasoning Phase:
  1. Analyze situation
  2. Generate possible actions
  3. Evaluate consequences
  4. Select best approach
  
         ↓
         
Action Phase:
  1. Execute selected action
  2. Observe results
  3. Update internal state
  4. Return to Perception
```

#### Example: Code Generation Agent
```
Perception: Receive user requirements
  - Analyze: language, complexity, constraints
  - Tools: parsing, requirement extraction

Reasoning: Plan implementation
  - Generate: multiple approaches
  - Evaluate: performance, maintainability, feasibility
  - Select: best approach

Action: Generate and test code
  - Execute: code generation
  - Verify: run tests
  - Refine: based on test results
```

---

### Pattern 2: **ReAct (Reasoning + Acting)**

Agent alternates between thinking and action, with thought processes directing tool use.

```
Thought 1: "I need to understand the current sales data"
Action 1: Use database_query tool
Observation: [Retrieved sales data]

Thought 2: "This data seems incomplete. Let me check for recent updates"
Action 2: Use refresh_data tool
Observation: [Updated data received]

Thought 3: "Now I can analyze and provide answer"
Action 3: Synthesize response
Observation: [Analysis complete]
```

#### Benefits
- **Explainability**: Thought process visible to user
- **Flexibility**: Reasoning can adapt to observations
- **Error Recovery**: Thinks about failures

---

### Pattern 3: **Tool Use Chain**

Agent sequences multiple tool calls to accomplish complex tasks.

```
Analyze Request
       ↓
Tool 1: Parse requirements
       ↓
Tool 2: Query database
       ↓
Tool 3: Process results
       ↓
Tool 4: Format output
       ↓
Return Response
```

#### Example: Report Generation
```
Tool 1: Extract data from source system
Tool 2: Validate data completeness
Tool 3: Calculate metrics and aggregations
Tool 4: Generate visualizations
Tool 5: Create formatted document
```

---

### Pattern 4: **Hierarchical Task Decomposition**

Break complex goals into subtasks executed hierarchically.

```
Main Goal: "Improve application performance"
│
├─ Subtask 1: Identify bottlenecks
│   ├─ Action 1a: Collect metrics
│   ├─ Action 1b: Analyze logs
│   └─ Action 1c: Profile code
│
├─ Subtask 2: Generate optimization strategies
│   ├─ Action 2a: Research solutions
│   ├─ Action 2b: Evaluate trade-offs
│   └─ Action 2c: Rank options
│
└─ Subtask 3: Implement and validate
    ├─ Action 3a: Code changes
    ├─ Action 3b: Unit tests
    └─ Action 3c: Performance tests
```

#### Benefits
- **Manageability**: Large problems become tractable
- **Parallelization**: Independent subtasks run in parallel
- **Observability**: Clear progress tracking

---

### Pattern 5: **Autonomous Refinement Loop**

Agent iteratively improves output without user intervention.

```
Generate Initial Output
       ↓
Evaluate Quality
       ↓
Does it meet criteria?
  ├─ Yes → Return
  └─ No → Refine
         ↓
     Regenerate with improvements
         ↓
     [Loop back to Evaluate]
```

#### Example: Code Generation
```
Generation 1: Create basic implementation
Evaluation: Check for common issues
Issues found: Missing error handling, inefficient algorithm
Refinement: Add error handling, optimize algorithm
Evaluation: Passes all checks
Output: Return refined code
```

---

### Pattern 6: **Multi-Agent Coordination**

Multiple specialized agents collaborate toward common goal.

```
Orchestrator Agent
       ├─ Data Agent: Gathers required information
       ├─ Analysis Agent: Performs calculations
       ├─ Validation Agent: Checks correctness
       └─ Formatting Agent: Prepares output
       
[Each agent reports results to orchestrator]
[Orchestrator synthesizes final response]
```

#### Use Cases
- Distributed problem-solving
- Specialized domain expertise
- Parallel task execution
- Scalable systems

---

### Pattern 7: **Memory-Augmented Operation**

Agent maintains and leverages memory across operations.

```
Task Received
       ↓
Check Memory:
  - Have I solved similar tasks?
  - What strategies worked?
  - What failed before?
       ↓
Apply Lessons Learned
       ↓
Execute Task
       ↓
Store Results and Insights in Memory
```

#### Memory Types
- **Episodic**: Specific past events
- **Semantic**: General knowledge
- **Procedural**: How to do things
- **Strategic**: What approaches work

---

### Pattern 8: **Exception Handling and Recovery**

Graceful handling of failures with recovery strategies.

```
Attempt Action
       ↓
Catch Error
       ↓
Classify Error Type:
  ├─ Transient (retry with backoff)
  ├─ Recoverable (try alternative approach)
  ├─ Configuration (escalate to human)
  └─ Fatal (terminate gracefully)
       ↓
Execute Recovery or Escalate
       ↓
Log and Learn
```

#### Example: API Call Pattern
```
try:
  Call API
catch Timeout Error:
  Retry with exponential backoff
catch Auth Error:
  Refresh credentials, retry
catch Rate Limit:
  Queue request, retry later
catch Permanent Error:
  Log, escalate, notify user
```

---

### Pattern 9: **Human-in-the-Loop**

Agent operates autonomously but escalates decisions requiring human judgment.

```
Agent Autonomous Zone
  - Gather information
  - Perform analysis
  - Generate recommendations
       ↓
Human Decision Zone
  - Review options
  - Apply domain expertise
  - Approve/modify decision
       ↓
Agent Execution Zone
  - Implement decision
  - Execute actions
  - Monitor results
```

#### Decision Escalation Criteria
- High-impact decisions (>threshold cost/risk)
- Ambiguous requirements
- Conflicting objectives
- Novel situations
- Ethical considerations

---

### Pattern 10: **Streaming and Progressive Output**

Agent provides results incrementally as they become available.

```
Start Processing
       ↓
Process Step 1 → Stream Result 1
       ↓
Process Step 2 → Stream Result 2
       ↓
Process Step 3 → Stream Result 3
       ↓
Complete → Final Result

Benefits:
- Faster perceived responsiveness
- Better UX for long operations
- User can start consuming results early
```

---

### Selection Guide

| Pattern | Best For | Complexity |
|---------|----------|-----------|
| **PRA** | Simple autonomy | Low |
| **ReAct** | Explainability | Low-Medium |
| **Tool Chain** | Sequential tasks | Medium |
| **Task Decomposition** | Complex problems | High |
| **Refinement Loop** | Quality optimization | Medium |
| **Multi-Agent** | Specialized expertise | High |
| **Memory-Augmented** | Learning systems | High |
| **Exception Handling** | Robust systems | Medium |
| **Human-in-Loop** | High-stakes decisions | Medium |
| **Streaming** | User experience | Low-Medium |

---

## LangChain

### Definition
LangChain is an open-source Python/JavaScript framework for building applications with Large Language Models, providing abstractions, utilities, and components for chaining LLM calls with external tools and data sources.

### Explanation
LangChain simplifies LLM application development by providing:
- **Building Blocks**: Pre-built components for common tasks
- **Chains**: Ways to combine LLMs with other tools and data
- **Agents**: Framework for creating autonomous decision-making systems
- **Memory Management**: Tools for maintaining conversation context
- **Integration**: Connectors to external APIs and databases

### Core Components

#### 1. **LLMs (Language Models)**
Wrapper around LLM APIs with unified interface
```python
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

# Unified interface regardless of provider
llm = ChatOpenAI(model="gpt-4")
# or
llm = ChatAnthropic(model="claude-3-opus-20240229")

response = llm.invoke("What is machine learning?")
```

#### 2. **Prompts and Templates**
Structured prompt management
```python
from langchain.prompts import PromptTemplate

template = """You are a helpful assistant.
Question: {question}
Answer:"""

prompt = PromptTemplate(
    input_variables=["question"],
    template=template
)

formatted = prompt.format(question="What is RAG?")
```

#### 3. **Chains**
Combine LLMs with other components in sequences
```python
from langchain.chains import LLMChain

chain = LLMChain(llm=llm, prompt=prompt)
result = chain.run(question="What is retrieval-augmented generation?")
```

#### 4. **Agents**
Enable LLMs to use tools and make decisions
```python
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool

tools = [
    Tool(name="Calculator", func=calculator_func, description="..."),
    Tool(name="Search", func=search_func, description="...")
]

agent = create_react_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools)
result = executor.invoke({"input": "What's 45 * 12 plus latest Bitcoin price?"})
```

#### 5. **Memory**
Manage conversation history
```python
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

memory = ConversationBufferMemory()
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    prompt=prompt
)

conversation.run("Hi, I'm learning about AI")
conversation.run("Can you expand on what you said?")
# Memory maintains context between calls
```

#### 6. **Retrievers and Vector Stores**
Enable RAG functionality
```python
from langchain_community.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(documents, embeddings)
retriever = vectorstore.as_retriever()

# Retrieve relevant documents
retrieved = retriever.get_relevant_documents("What is machine learning?")
```

#### 7. **Output Parsers**
Structure and validate model outputs
```python
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

class Analysis(BaseModel):
    summary: str = Field(description="Summary of findings")
    score: int = Field(description="Overall score 1-10")

parser = PydanticOutputParser(pydantic_object=Analysis)
prompt = PromptTemplate(
    template="{format_instructions}\nAnalyze: {text}",
    input_variables=["text"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain = LLMChain(llm=llm, prompt=prompt, output_parser=parser)
result = chain.run(text="Some text to analyze")
```

### Architecture Overview

```
User Input
    ↓
LangChain Application
├─ Prompt Template: Structure input
├─ LLM: Get response from model
├─ Tools: Execute functions
├─ Retrievers: Get external data
├─ Memory: Maintain context
└─ Output Parser: Structure output
    ↓
Final Output
```

### Common Patterns

#### **RAG Chain**
```python
from langchain.chains import RetrievalQA

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever
)

answer = qa.run("What's the revenue for Q4?")
# Retrieves documents, augments prompt, generates answer
```

#### **Agentic Loop**
```python
from langchain_openai import OpenAI
from langchain.agents import load_tools, initialize_agent, AgentType

tools = load_tools(["serpapi", "llm-math"], llm=llm)
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

result = agent.run("What's sqrt(144) times the current Bitcoin price?")
```

#### **Multi-step Workflow**
```python
from langchain.chains import SimpleSequentialChain

chain1 = LLMChain(llm=llm, prompt=prompt1)
chain2 = LLMChain(llm=llm, prompt=prompt2)
chain3 = LLMChain(llm=llm, prompt=prompt3)

sequential_chain = SimpleSequentialChain(chains=[chain1, chain2, chain3])
result = sequential_chain.run("Initial input")
```

### Advantages
- **Standardization**: Unified interface across multiple LLM providers
- **Rapid Development**: Pre-built components accelerate development
- **Tool Integration**: Easy connection to external APIs and databases
- **Experimentation**: Test different chains and configurations quickly
- **Community**: Large ecosystem of integrations and extensions

### Limitations
- **Dependency Size**: Large library with many dependencies
- **Abstraction Overhead**: Generic abstractions may add complexity
- **Token Efficiency**: No built-in context optimization
- **Latency**: Multiple API calls without built-in parallelization

### When to Use LangChain
- ✅ Building multi-step applications
- ✅ Need agent capabilities
- ✅ Integrating multiple tools and APIs
- ✅ RAG applications
- ✅ Rapid prototyping
- ❌ Single-prompt inference
- ❌ Latency-critical applications
- ❌ Cost-sensitive use cases with many API calls

---

## LangGraph

### Definition
LangGraph is a library for building stateful, multi-actor applications using LLMs, providing a framework for creating complex agentic systems with explicit state management and control flow.

### Explanation
LangGraph extends LangChain's capabilities with a graph-based computation model where nodes represent processing steps and edges define transitions. This enables more complex agent behaviors, including cycles, branching logic, and explicit state management—capabilities limited in linear chains.

### Core Concept: State Machines for AI

Traditional chains are linear:
```
Step 1 → Step 2 → Step 3 → Result
```

LangGraph enables complex workflows:
```
         ┌─→ Tool A ──┐
Input ──┤            ├─→ Decision ──→ Result
         └─→ Tool B ──┘
```

### Key Components

#### 1. **StateGraph**
Defines the graph structure and state transitions
```python
from langgraph.graph import StateGraph
from typing import TypedDict, Annotated
import operator

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]  # Messages accumulate
    documents: list
    next_step: str

graph = StateGraph(AgentState)
```

#### 2. **Nodes**
Processing steps that transform state
```python
def retrieve_documents(state):
    """Node: retrieve relevant documents"""
    query = state["messages"][-1]
    docs = retriever.get_relevant_documents(query)
    return {"documents": docs}

def generate_response(state):
    """Node: generate response using documents"""
    context = "\n".join([doc.page_content for doc in state["documents"]])
    response = llm.invoke(f"Context: {context}\nQuestion: {state['messages'][-1]}")
    return {"messages": [response]}

graph.add_node("retrieve", retrieve_documents)
graph.add_node("generate", generate_response)
```

#### 3. **Edges**
Connections defining state transitions and flow logic
```python
# Direct edge: always transition to next node
graph.add_edge("retrieve", "generate")

# Conditional edge: route based on state
def route_decision(state):
    """Decide where to go based on state"""
    if len(state["documents"]) == 0:
        return "search_web"
    else:
        return "generate"

graph.add_conditional_edges(
    "retrieve",
    route_decision,
    {
        "search_web": "web_search",
        "generate": "generate"
    }
)
```

#### 4. **Entry and Exit Points**
Define where execution starts and ends
```python
graph.set_entry_point("retrieve")
graph.set_finish_point("generate")

runnable = graph.compile()
```

### Architecture Example: Multi-step Agent

```
graph structure:
                ┌─→ [search_web] ──┐
    [input] → [route] ─┤          ├─→ [synthesis] → [output]
                └─→ [database] ──┘

State flows through nodes:
1. Input received with query
2. Route node decides: search_web or database based on query type
3. Execute search_web or database node
4. Synthesis node combines results
5. Output returned
```

### Implementation Pattern

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Literal

class WorkflowState(TypedDict):
    messages: list
    documents: list
    analysis: str
    next: Literal["analyze", "decide", "end"]

def retrieve_step(state):
    """Step 1: Retrieve documents"""
    docs = retriever.search(state["messages"][-1])
    return {"documents": docs}

def analyze_step(state):
    """Step 2: Analyze documents"""
    analysis = llm.invoke(f"Analyze: {state['documents']}")
    return {"analysis": str(analysis)}

def route_step(state):
    """Step 3: Decide next action"""
    if "error" in state["analysis"]:
        return "end"
    else:
        return "decide"

def decide_step(state):
    """Step 4: Make decision"""
    decision = llm.invoke(f"Based on: {state['analysis']}, decide: ...")
    return {"messages": state["messages"] + [decision]}

# Build graph
graph = StateGraph(WorkflowState)

# Add nodes
graph.add_node("retrieve", retrieve_step)
graph.add_node("analyze", analyze_step)
graph.add_node("decide", decide_step)

# Add edges
graph.set_entry_point("retrieve")
graph.add_edge("retrieve", "analyze")
graph.add_conditional_edges("analyze", route_step)
graph.add_edge("decide", END)

# Compile
workflow = graph.compile()

# Execute
result = workflow.invoke({"messages": ["What is machine learning?"], "documents": [], "analysis": "", "next": "analyze"})
```

### Advanced Features

#### **Persistence and Checkpointing**
```python
from langgraph.checkpoint.sqlite import SqliteSaver

memory = SqliteSaver.from_conn_string(":memory:")
graph = state_graph.compile(checkpointer=memory)

# Later: resume from checkpoint
config = {"configurable": {"thread_id": "abc123"}}
result = graph.invoke(input, config)  # Resumes if interrupted
```

#### **Streaming**
```python
# Stream events as they occur
for event in graph.stream(input):
    print(event)
```

#### **Human-in-the-loop**
```python
def should_continue(state):
    """Wait for human approval"""
    if state["requires_approval"]:
        return "human_approval"  # Pause for human input
    else:
        return "continue"

graph.add_node("human_approval", approval_node)
```

### Comparison: Chains vs. Graphs

| Feature | LangChain Chains | LangGraph |
|---------|------------------|-----------|
| Flow | Linear | Graph-based |
| Cycles | No | Yes |
| Branching | Limited | Rich |
| State | Implicit | Explicit |
| Visualization | Limited | Excellent |
| Complexity | Low-Medium | Medium-High |
| Loops/Iteration | Workaround | Native |
| Human-in-loop | Difficult | Built-in |
| Debugging | Harder | Easier |

### Use Cases

**LangChain Chains:**
- Simple sequential operations
- RAG pipeline
- Multi-step reasoning
- Basic tool use

**LangGraph:**
- Complex agents with decision logic
- Multi-turn interactions with state
- Workflow automation
- Systems requiring human approval
- Error recovery loops
- Long-running processes

### Example: Research Agent with LangGraph

```python
class ResearchState(TypedDict):
    query: str
    searches: Annotated[list, operator.add]  # Accumulate searches
    papers: list
    summary: str

def search_papers(state):
    """Find research papers"""
    searches = search_engine.find_papers(state["query"])
    return {"searches": searches}

def evaluate_papers(state):
    """Filter papers by relevance"""
    relevant = [p for p in state["searches"] if score(p) > threshold]
    return {"papers": relevant}

def route_research(state):
    """Decide if we have enough papers"""
    if len(state["papers"]) >= 5:
        return "summarize"
    else:
        return "search"  # Search more

def summarize_papers(state):
    """Create summary of findings"""
    summary = llm.invoke(f"Summarize these papers: {state['papers']}")
    return {"summary": summary}

graph = StateGraph(ResearchState)
graph.add_node("search", search_papers)
graph.add_node("evaluate", evaluate_papers)
graph.add_node("summarize", summarize_papers)

graph.set_entry_point("search")
graph.add_edge("search", "evaluate")
graph.add_conditional_edges("evaluate", route_research)
graph.add_edge("summarize", END)

research_agent = graph.compile()

# Run: searches → evaluates → searches more if needed → summarizes
result = research_agent.invoke({"query": "quantum computing applications", "searches": [], "papers": [], "summary": ""})
```

### Advantages
- **Explicit Control**: Clear, visual representation of workflow
- **Complex Logic**: Native support for branching, loops, and conditions
- **State Management**: Explicit state handling reduces bugs
- **Debugging**: Visualizable, step-through execution
- **Human Integration**: Pause points for human decisions
- **Persistence**: Save and resume execution

### When to Use LangGraph
- ✅ Complex multi-step workflows
- ✅ Need explicit control flow
- ✅ Requiring state management
- ✅ Loops or conditional routing
- ✅ Long-running agents
- ✅ Human-in-the-loop systems
- ❌ Simple sequential chains (use LangChain)
- ❌ Single-step inference
- ❌ Real-time, latency-critical applications

---

## Summary Table: Key Concepts

| Concept | Purpose | Complexity | Use When |
|---------|---------|-----------|----------|
| **Generative AI** | Create new content | Low | Building any AI application |
| **Prompt Engineering** | Guide AI outputs | Low | Optimizing individual requests |
| **AI Agent** | Autonomous decision-making | Medium-High | Need task automation |
| **RAG** | Ground AI in knowledge | Medium | Need current/accurate info |
| **Context Window** | Input size limit | Low | Managing costs and tokens |
| **Multi-turn** | Extended dialogue | Low | Complex problem-solving |
| **Context Compaction** | Save tokens | Medium | Long conversations |
| **Agentic Design** | Build agent systems | High | Complex autonomous systems |
| **Workflow Patterns** | Proven solutions | Medium-High | Designing agent workflows |
| **LangChain** | Build LLM apps | Medium | Rapid prototyping, multi-step |
| **LangGraph** | Complex workflows | High | Advanced agent systems |

---

## Relationships and Dependencies

```
Core Concepts
├─ Generative AI [Foundation]
│  ├─ Prompt Engineering [Optimization]
│  ├─ Context Window [Constraint]
│  └─ Multi-turn Conversations [Interaction Pattern]
│
├─ AI Agents [Autonomous Systems]
│  ├─ Agentic AI Design [Architecture]
│  ├─ Agentic AI Workflow Patterns [Implementation]
│  ├─ RAG [Knowledge Integration]
│  └─ Context Compaction [Efficiency]
│
└─ Tools and Frameworks
   ├─ LangChain [Foundation]
   │  └─ LangGraph [Advanced Workflows]
   └─ RAG [Knowledge Grounding]
```

---

## Quick Reference Guide

| Need | Concept | How |
|------|---------|-----|
| Better AI responses | Prompt Engineering | Structure queries, provide examples, iterate |
| Autonomous system | AI Agent | Use reasoning, tools, memory, decision-making |
| Accurate information | RAG | Retrieve relevant context, augment prompts |
| Long conversations | Multi-turn + Compaction | Maintain history, compress old messages |
| Complex workflows | Agentic AI Workflow Patterns | Use PRA, ReAct, Tool Chains, Hierarchical Decomposition |
| Build LLM apps | LangChain | Use chains, agents, memory, retrievers |
| Advanced agent systems | LangGraph | Define state machines, control flow, human-in-loop |
| Manage tokens | Context Window awareness + Compaction | Monitor usage, summarize history |

