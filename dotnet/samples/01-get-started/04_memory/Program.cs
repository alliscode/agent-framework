// Copyright (c) Microsoft. All rights reserved.

// Agent Memory with Context Providers and Session State
//
// Context providers inject dynamic context into each agent call. This sample
// shows a provider that stores user info in session state and personalizes
// responses — the info persists across turns via the session.
//
// The .NET version also demonstrates serialization/deserialization of sessions
// and transferring memory to new sessions.

using System.Text;
using System.Text.Json;
using Azure.AI.Extensions.OpenAI;
using Azure.AI.Projects;
using Azure.Identity;
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using SampleApp;

var endpoint = Environment.GetEnvironmentVariable("FOUNDRY_PROJECT_ENDPOINT") ?? throw new InvalidOperationException("FOUNDRY_PROJECT_ENDPOINT is not set.");
var deploymentName = Environment.GetEnvironmentVariable("FOUNDRY_MODEL") ?? "gpt-5.4-mini";

AIProjectClient projectClient = new(new Uri(endpoint), new DefaultAzureCredential());
IChatClient chatClient = projectClient.GetProjectOpenAIClient().GetProjectResponsesClientForModel(deploymentName).AsIChatClientWithStoredOutputDisabled(deploymentName);

// <create_agent>
// Create the agent with a memory provider that stores user info across turns.
AIAgent agent = projectClient.AsAIAgent(new ChatClientAgentOptions()
{
    ChatOptions = new() { ModelId = deploymentName, Instructions = "You are a friendly assistant." },
    AIContextProviders = [new UserInfoMemory(chatClient)]
});
// </create_agent>

// <run_with_memory>
AgentSession session = await agent.CreateSessionAsync();

// The provider doesn't know the user yet — it will ask for their name
Console.WriteLine(await agent.RunAsync("Hello, what is the square root of 9?", session));

// Now provide the name — the provider stores it in session state
Console.WriteLine(await agent.RunAsync("My name is Alice", session));
Console.WriteLine(await agent.RunAsync("I am 20 years old", session));

// Subsequent calls are personalized — info persists via session state
Console.WriteLine(await agent.RunAsync("What is my name and age?", session));

// Inspect session state to see what the provider stored
var userInfo = agent.GetService<UserInfoMemory>()?.GetUserInfo(session);
Console.WriteLine($"\n[Session State] User Name: {userInfo?.UserName}");
Console.WriteLine($"[Session State] User Age: {userInfo?.UserAge}");
// </run_with_memory>

// --- .NET extras: session serialization and transfer ---

// Serialize the session — state is included, so it can be persisted or transferred
JsonElement sessionElement = await agent.SerializeSessionAsync(session);

// Deserialize and continue the conversation with previous memory
Console.WriteLine("\n>> Deserialized session:");
var deserializedSession = await agent.DeserializeSessionAsync(sessionElement);
Console.WriteLine(await agent.RunAsync("What is my name and age?", deserializedSession));

// Transfer memory to a brand-new session
Console.WriteLine("\n>> New session with transferred memory:");
var newSession = await agent.CreateSessionAsync();
if (userInfo is not null && agent.GetService<UserInfoMemory>() is UserInfoMemory memory)
{
    memory.SetUserInfo(newSession, userInfo);
}
Console.WriteLine(await agent.RunAsync("What is my name and age?", newSession));

namespace SampleApp
{
    // <context_provider>
    /// <summary>
    /// A context provider that remembers user info in session state.
    /// Before each call, it injects personalization instructions (or asks for missing info).
    /// After each call, it extracts and stores the user's name and age.
    /// </summary>
    internal sealed class UserInfoMemory : AIContextProvider
    {
        private readonly ProviderSessionState<UserInfo> _sessionState;
        private IReadOnlyList<string>? _stateKeys;
        private readonly IChatClient _chatClient;

        public UserInfoMemory(IChatClient chatClient, Func<AgentSession?, UserInfo>? stateInitializer = null)
        {
            this._sessionState = new ProviderSessionState<UserInfo>(
                stateInitializer ?? (_ => new UserInfo()),
                this.GetType().Name);
            this._chatClient = chatClient;
        }

        public override IReadOnlyList<string> StateKeys => this._stateKeys ??= [this._sessionState.StateKey];

        public UserInfo GetUserInfo(AgentSession session)
            => this._sessionState.GetOrInitializeState(session);

        public void SetUserInfo(AgentSession session, UserInfo userInfo)
            => this._sessionState.SaveState(session, userInfo);

        protected override async ValueTask StoreAIContextAsync(InvokedContext context, CancellationToken cancellationToken = default)
        {
            var userInfo = this._sessionState.GetOrInitializeState(context.Session);

            // Extract user name and age from the conversation if not already known
            if ((userInfo.UserName is null || userInfo.UserAge is null) && context.RequestMessages.Any(x => x.Role == ChatRole.User))
            {
                var result = await this._chatClient.GetResponseAsync<UserInfo>(
                    context.RequestMessages,
                    new ChatOptions()
                    {
                        Instructions = "Extract the user's name and age from the message if present. If not present return nulls."
                    },
                    cancellationToken: cancellationToken);

                userInfo.UserName ??= result.Result.UserName;
                userInfo.UserAge ??= result.Result.UserAge;
            }

            this._sessionState.SaveState(context.Session, userInfo);
        }

        protected override ValueTask<AIContext> ProvideAIContextAsync(InvokingContext context, CancellationToken cancellationToken = default)
        {
            var userInfo = this._sessionState.GetOrInitializeState(context.Session);

            // Inject personalization instructions based on stored user info
            StringBuilder instructions = new();
            instructions
                .AppendLine(
                    userInfo.UserName is null ?
                        "You don't know the user's name yet. Ask for it politely." :
                        $"The user's name is {userInfo.UserName}. Always address them by name.")
                .AppendLine(
                    userInfo.UserAge is null ?
                        "You don't know the user's age yet. Ask for it politely." :
                        $"The user's age is {userInfo.UserAge}.");

            return new ValueTask<AIContext>(new AIContext
            {
                Instructions = instructions.ToString()
            });
        }
    }
    // </context_provider>

    internal sealed class UserInfo
    {
        public string? UserName { get; set; }
        public int? UserAge { get; set; }
    }
}
