# Getting Started Workflows Catalog

This catalog documents every Python file in `python/samples/getting_started/workflows/`, grouped by subdirectory, with a one-paragraph summary of what each teaches.

---

## _start-here

- **step1_executors_and_edges.py**  
  Demonstrates foundational workflow patterns using the agent_framework library. It introduces two ways to define workflow nodes: subclassing Executor for custom behavior and using the @executor decorator for function-based nodes. The sample builds a two-step workflow (uppercase conversion and text reversal) and showcases the WorkflowBuilder API to wire nodes, run the workflow, and collect outputs—providing an accessible introduction to workflow orchestration.

- **step2_agents_in_a_workflow.py**  
  Walks through integrating agents within a workflow using AzureOpenAIChatClient. Builds a two-node workflow: Writer and Reviewer agents for content creation and evaluation. Shows agent creation, workflow sequencing, and running the workflow with initial input. Illustrates automatic agent output yield and Azure credential requirements.

- **step3_streaming.py**  
  Expands on agent workflow orchestration by showing streaming mode in workflows. Defines custom Executors wrapping AzureOpenAI agents for Writer and Reviewer roles, with typed input/output. Streaming events can be observed as agents produce responses and workflow state changes arise, teaching how to distinguish agent and runner events in real time.

- **step4_using_factories.py**  
  Explains how to define executors and agents via factory functions for registration with WorkflowBuilder. Highlights lazy instantiation, state management, and the benefits for production workflows. The sample constructs a three-stage text transformation workflow and illustrates both class-based/function-based executors and agent creation, culminating in seamless streaming of outputs.

---

## agents

- **azure_ai_agents_streaming.py**  
  Wires Azure AI agents directly as edges in a workflow, illustrating how incremental streaming data is observed as agents emit partial outputs. Demonstrates streaming behavior via AgentRunUpdateEvent and explains workflow mode differences for agent data emission.

- **azure_chat_agents_function_bridge.py**  
  Constructs a pipeline with two Azure chat agents bridged by an @executor function that simulates fetching external references. This workflow demonstrates the interception, enrichment, and coordinated handoff of agent responses, along with streaming updates.

- **azure_chat_agents_streaming.py**  
  Builds a streaming workflow with Writer and Reviewer Azure chat agents. Highlights streaming of partial agent responses and agent deltas, and the API syntax for constructing such chains efficiently with agent chaining.

- **azure_chat_agents_tool_calls_with_feedback.py**  
  Demonstrates a tool-enabled agent scenario with human-in-the-loop feedback. A writer agent calls Python function tools, then Coordinator emits RequestInfoEvent for human review, which loops back human edits before a final editor agent polishes the text. Streams agent tool calls, human review events, and deltas.

- **concurrent_workflow_as_agent.py**  
  Shows building a concurrent (fan-out/fan-in) workflow orchestrating multiple agents in parallel and then wrapping the orchestration as a single agent. Useful to demonstrate parallel agent flows and response aggregation in an agent-oriented architecture.

- **custom_agent_executors.py**  
  Demonstrates combining an AI agent with tool capabilities (code interpreter) and a custom evaluator executor. The agent generates code, while the evaluator assesses correctness and resource usage, showcasing flexible workflow design.

- **group_chat_workflow_as_agent.py**  
  Presents a group chat orchestration scenario using GroupChatBuilder to coordinate a researcher and writer agent under a manager. Teaches group chat assembly, agent creation, and as_agent wrapping for transcript aggregation and reuse.

- **handoff_workflow_as_agent.py**  
  Teaches converting a handoff workflow (triage-to-specialist routing) into an agent with human-in-the-loop input requests. Explains how handoff interacts with request/response events and how to differentiate workflow vs. agent-as-wrapper modes.

- **magentic_workflow_as_agent.py**  
  Walks through configuring a Magentic workflow (research/coding agents with a manager) and running it as an agent with streaming/telemetry callbacks. Illustrates cyclic workflows and orchestration patterns.

- **mixed_agents_and_executors.py**  
  Shows flexible orchestration by mixing AI agent executors and custom function or tool steps as workflow nodes, extending the compositional capabilities of the framework.

- **sequential_workflow_as_agent.py**  
  Builds a sequential agent workflow with Writer and Reviewer agents, showing stepwise orchestration and usage of the as_agent interface for reicursive composition in higher-level workflows.

- **workflow_as_agent_human_in_the_loop.py**  
  Demonstrates workflows that escalate to a human reviewer for uncertain decisions, combining Worker/Reviewer cycles with external feedback handling and completion upon human approval.

- **workflow_as_agent_kwargs.py**  
  Shows workflows that propagate custom context (skills, user tokens, etc.) to @ai_function tools via kwargs. Teaches agent-wrapped workflows capable of context propagation for personalized actions.

- **workflow_as_agent_reflection_pattern.py**  
  Implements a workflow-as-agent with reflection and retry, cycling between Worker and Reviewer with structured feedback, iteratively improving output until approval.

- **workflow_as_agent_with_thread.py**  
  Demonstrates context and conversation history preservation across multi-turn workflows using AgentThread and ChatMessageStore, including checkpointing, resuming, and context-aware execution for stateful agent flows.

---

## checkpoint

- **checkpoint_with_human_in_the_loop.py**  
  Provides a minimal, checkpoint-aware workflow specializing in human-in-the-loop approvals. Shows how to pause, persist, and resume workflows with checkpoint state, demonstrating human gating, restoration, and looping for approvals in a product release workflow.

- **checkpoint_with_resume.py**  
  Explains enabling checkpointing for long-running workflows using InMemoryCheckpointStorage. Shows stateful worker executors with checkpoint save/restore hooks, and how to safely resume a computation pipeline after interruptions.

- **handoff_with_tool_approval_checkpoint_resume.py**  
  Demonstrates checkpoint-persisted handoff workflows with tool approval and human responses. Shows resuming from checkpoints across both user input and tool approval requests, explaining the necessary two-step resume pattern for tooling integration.

- **sub_workflow_checkpoint.py**  
  Explores checkpointing for parent workflows embedding sub-workflows. Teaches capturing and restoring checkpoints that span both parent and sub-workflow state, with complex human approval payloads and resumption after human review.

- **workflow_as_agent_checkpoint.py**  
  Shows how to enable checkpointing with agent-wrapped workflows, how to pass checkpoint storage, and how to combine thread conversation history and checkpointing for detailed, multi-turn, resumable agent runs.

---

## composition

- **sub_workflow_basics.py**  
  Introduces wiring sub-workflows within a parent workflow, offering a template for orchestrating multiple text processors (word/character count), collecting results, and summarizing via subworkflow outputs.

- **sub_workflow_kwargs.py**  
  Demonstrates context/kwargs propagation from a parent workflow through sub-workflows, enabling scenarios where authentication tokens or other configuration must traverse across all levels of orchestration.

- **sub_workflow_parallel_requests.py**  
  Shows handling parallel outbound requests from a sub-workflow to multiple parent-level executors (resource allocator, policy checker), collecting results for final aggregation—a design suitable for distributed orchestration.

- **sub_workflow_request_interception.py**  
  Explores intercepting requests from sub-workflows in the main workflow, exemplified by a smart email delivery pipeline that validates, intercepts, and routes email addresses for better security.

---

## control-flow

- **edge_condition.py**  
  Demonstrates decision workflows with conditional edges. Shows how to use Pydantic models for response validation and edge routing, constructing a minimal email spam detection pipeline using structured outputs and branching logic.

- **multi_selection_edge_group.py**  
  Uses a multi-selection edge group for fan-out/fan-in workflows. Examines branching for NotSpam/Spam/Uncertain with state sharing and conditional persistence logic in an email triage and response workflow.

- **sequential_executors.py**  
  Builds a sequential workflow with custom Executor classes for uppercasing and reversing strings, highlighting handler method design, explicit chaining, and streaming result collection.

- **sequential_streaming.py**  
  Uses function-decorated workflow steps for uppercase/reverse text operations, demonstrating function-style executor wiring, streaming, and ordered event emission.

- **simple_loop.py**  
  Shows a feedback loop (binary search game) between a guessing executor, an agent judge, and response parser. Demonstrates event-driven cycles, limited-range search and feedback-driven orchestration.

- **switch_case_edge_group.py**  
  Implements decision-driven branching using switch-case edge groups, providing a deterministic N-way router for spam detection in ambiguous emails with explicit NotSpam/Spam/Uncertain cases.

- **workflow_cancellation.py**  
  Illustrates mid-execution cancellation of long workflows using asyncio task cancellation, demonstrating both successful completion and externally-imposed cancellation and their effects.

---

## harness

- **agent_harness_basics.py**  
  Introduces the Agent Harness as a runtime for durable, observable, and controllable agent execution. Shows agent turns, transcript tracking, checkpointing, turn-limiting with both a simple API and builder-style setup.

- **agent_harness_checkpoint.py**  
  Explains harness checkpointing and resumption for durability. Demonstrates persistence of turn state and transcript, checkpoint inspection, and restarting runs after interruption.

- **agent_harness_custom_executors.py**  
  Illustrates harness composability by adding custom executors for logging, mock agent turns, and progress reporting, extending the harness beyond standard agent loops.

- **coding_tools.py**  
  Provides coding-oriented tools (file ops, command execution) as callable functions, forming the toolbox for coding-oriented agent tasks. Details secure sandboxing and resource restrictions.

- **devui_harness.py**  
  Demonstrates harness integration with DevUI for live testing or SMS relay, including context compaction, work item tracking, MCP tool integration, and configuration for agent workspace.

- **evaluate_baseline.py**  
  Runs a baseline agent interaction (no harness) for side-by-side comparison with harness-enabled runs. Useful for measuring additional controls/responsibility introduced by the harness infrastructure.

- **evaluate_harness_output.py**  
  Analyzes harness output for compliance with anti-double-emission and artifact role rules: checks response/role classification, anti-meta-references, and response brevity.

- **harness_coding_test.py**  
  Runs coding tasks through the harness in both interactive and contract-verification (automation) mode, demonstrating file creation, test execution, and evidence collection.

- **harness_repl.py**  
  Interactive command-line REPL for harness-based coding agents, providing lifecycle tracing, debug logging, file listing, and session management for development or demonstration.

- **harness_test_runner.py**  
  Executes the agent harness with a fixed prompt, recording structured results for evaluation—such as response text, files created/read, tool calls, and deliverables.

- **sms_relay.py**  
  Provides a standalone service for receiving and relaying SMS/WhatsApp messages through Azure Communication Services, enabling harness workflows to communicate via messaging.

---

## human-in-the-loop

- **agents_with_approval_requests.py**  
  Models a workflow where agents execute AI functions requiring approval, showing the hybridization of automation and oversight. Handles approval requests with auto-approval logic for demonstration, and walks through the complete loop of AI function requests and human responses.

- **concurrent_request_info.py**  
  Teaches concurrent workflows (e.g., multi-expert analysis) that pause for human review before aggregation. Demonstrates request info hooks between parallel execution and aggregation, plus synthesis/steering injection.

- **group_chat_request_info.py**  
  Demonstrates group chat workflows configured to pause for human input before selected agent turns, allowing targeted steering of conversations with participant filters.

- **guessing_game_with_human_input.py**  
  Implements a human-in-the-loop guessing game, interleaving agent guesses, human higher/lower/correct feedback, and real-time event-driven resume logic,
powered by request_info and send_responses_streaming.

---
