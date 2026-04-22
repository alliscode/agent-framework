// Copyright (c) Microsoft. All rights reserved.

// MCP Server with OAuth Authentication — Connect to a protected MCP server
//
// This sample demonstrates how to create an AI agent that connects to an
// MCP server requiring OAuth 2.0 authentication. It handles the full OAuth
// flow including dynamic client registration and browser-based authorization.

using System.Diagnostics;
using System.Net;
using System.Text;
using System.Web;
using Azure.AI.Projects;
using Azure.Identity;
using Microsoft.Agents.AI;
using Microsoft.Extensions.Logging;
using ModelContextProtocol.Client;

// --- Configuration ---
var endpoint = Environment.GetEnvironmentVariable("FOUNDRY_PROJECT_ENDPOINT") ?? throw new InvalidOperationException("FOUNDRY_PROJECT_ENDPOINT is not set.");
var deploymentName = Environment.GetEnvironmentVariable("FOUNDRY_MODEL") ?? "gpt-5.4-mini";

// --- HTTP client setup ---
using var sharedHandler = new SocketsHttpHandler
{
    PooledConnectionLifetime = TimeSpan.FromMinutes(2),
    PooledConnectionIdleTimeout = TimeSpan.FromMinutes(1)
};
using var httpClient = new HttpClient(sharedHandler);

var consoleLoggerFactory = LoggerFactory.Create(builder => builder.AddConsole());

// --- Connect to the protected MCP server with OAuth ---
var serverUrl = "http://localhost:7071/";
var transport = new HttpClientTransport(new()
{
    Endpoint = new Uri(serverUrl),
    Name = "Secure Weather Client",
    OAuth = new()
    {
        DynamicClientRegistration = new()
        {
            ClientName = "ProtectedMcpClient",
        },
        RedirectUri = new Uri("http://localhost:1179/callback"),
        AuthorizationRedirectDelegate = HandleAuthorizationUrlAsync,
    }
}, httpClient, consoleLoggerFactory);

await using var mcpClient = await McpClient.CreateAsync(transport, loggerFactory: consoleLoggerFactory);

// Discover tools exposed by the MCP server
var mcpTools = await mcpClient.ListToolsAsync().ConfigureAwait(false);

// --- Create the agent with MCP tools ---
AIAgent agent = new AIProjectClient(
    new Uri(endpoint),
    new DefaultAzureCredential())
     .AsAIAgent(model: deploymentName, instructions: "You answer questions related to the weather.", tools: [.. mcpTools]);

// --- Run the agent ---
Console.WriteLine(await agent.RunAsync("Get current weather alerts for New York?"));

// Handles the OAuth authorization URL by starting a local HTTP server and opening a browser.
// This implementation demonstrates how SDK consumers can provide their own authorization flow.
static async Task<string?> HandleAuthorizationUrlAsync(Uri authorizationUrl, Uri redirectUri, CancellationToken cancellationToken)
{
    Console.WriteLine("Starting OAuth authorization flow...");
    Console.WriteLine($"Opening browser to: {authorizationUrl}");

    var listenerPrefix = redirectUri.GetLeftPart(UriPartial.Authority);
    if (!listenerPrefix.EndsWith("/", StringComparison.InvariantCultureIgnoreCase))
    {
        listenerPrefix += "/";
    }

    using var listener = new HttpListener();
    listener.Prefixes.Add(listenerPrefix);

    try
    {
        listener.Start();
        Console.WriteLine($"Listening for OAuth callback on: {listenerPrefix}");

        OpenBrowser(authorizationUrl);

        var context = await listener.GetContextAsync();
        var query = HttpUtility.ParseQueryString(context.Request.Url?.Query ?? string.Empty);
        var code = query["code"];
        var error = query["error"];

        const string ResponseHtml = "<html><body><h1>Authentication complete</h1><p>You can close this window now.</p></body></html>";
        byte[] buffer = Encoding.UTF8.GetBytes(ResponseHtml);
        context.Response.ContentLength64 = buffer.Length;
        context.Response.ContentType = "text/html";
        context.Response.OutputStream.Write(buffer, 0, buffer.Length);
        context.Response.Close();

        if (!string.IsNullOrEmpty(error))
        {
            Console.WriteLine($"Auth error: {error}");
            return null;
        }

        if (string.IsNullOrEmpty(code))
        {
            Console.WriteLine("No authorization code received");
            return null;
        }

        Console.WriteLine("Authorization code received successfully.");
        return code;
    }
    catch (Exception ex)
    {
        Console.WriteLine($"Error getting auth code: {ex.Message}");
        return null;
    }
    finally
    {
        if (listener.IsListening)
        {
            listener.Stop();
        }
    }
}

// Opens the specified URL in the default browser.
static void OpenBrowser(Uri url)
{
    try
    {
        var psi = new ProcessStartInfo
        {
            FileName = url.ToString(),
            UseShellExecute = true
        };
        Process.Start(psi);
    }
    catch (Exception ex)
    {
        Console.WriteLine($"Error opening browser. {ex.Message}");
        Console.WriteLine($"Please manually open this URL: {url}");
    }
}
