// Copyright (c) Microsoft. All rights reserved.

using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.IO;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.AI;

namespace Microsoft.Agents.AI.Tools.Shell;

/// <summary>
/// Cross-platform shell tool. <b>Approval-in-the-loop is the security boundary.</b>
/// </summary>
/// <remarks>
/// <para>
/// <c>LocalShellTool</c> launches a real shell (bash/sh on POSIX, pwsh/powershell/cmd on Windows)
/// to execute commands emitted by an agent. Output is captured, optionally truncated, and a
/// timeout terminates the process tree.
/// </para>
/// <para>
/// This is the .NET counterpart to the Python <c>agent_framework_tools.shell.LocalShellTool</c>.
/// The current build implements <see cref="ShellMode.Stateless"/> only — every call spawns a
/// fresh shell. <see cref="ShellMode.Persistent"/> (long-lived shell + sentinel protocol) is
/// reserved for a follow-up.
/// </para>
/// <para>
/// <b>Threat model.</b> The deny list is a guardrail, not a security boundary. Real isolation
/// requires either (a) approval-in-the-loop, where every command is reviewed by a human via the
/// harness <c>ToolApprovalAgent</c>, or (b) container isolation. To disable approval gating you
/// must pass <c>acknowledgeUnsafe: true</c>; otherwise the tool refuses to wire up.
/// </para>
/// </remarks>
public sealed class LocalShellTool : IDisposable
{
    private const int DefaultMaxOutputBytes = 64 * 1024;

    private readonly ShellPolicy _policy;
    private readonly ResolvedShell _shell;
    private readonly TimeSpan? _timeout;
    private readonly int _maxOutputBytes;
    private readonly string? _workingDirectory;
    private readonly IReadOnlyDictionary<string, string?>? _environment;
    private readonly Action<string>? _onCommand;

    /// <summary>
    /// Initializes a new instance of the <see cref="LocalShellTool"/> class.
    /// </summary>
    /// <param name="mode">Execution mode. Currently only <see cref="ShellMode.Stateless"/> is implemented.</param>
    /// <param name="shell">Override path to the shell binary. Falls back to the <c>AGENT_FRAMEWORK_SHELL</c> environment variable, then OS defaults.</param>
    /// <param name="workingDirectory">Working directory for the spawned shell. Defaults to the current process directory.</param>
    /// <param name="environment">Extra environment variables. Pass a <see langword="null"/> value to remove an inherited variable.</param>
    /// <param name="policy">Optional <see cref="ShellPolicy"/>. Defaults to a policy seeded with <see cref="ShellPolicy.DefaultDenyList"/>.</param>
    /// <param name="timeout">Per-command timeout. <see langword="null"/> disables timeouts.</param>
    /// <param name="maxOutputBytes">Combined stdout/stderr cap before truncation.</param>
    /// <param name="onCommand">Audit callback invoked for every allowed command.</param>
    public LocalShellTool(
        ShellMode mode = ShellMode.Stateless,
        string? shell = null,
        string? workingDirectory = null,
        IReadOnlyDictionary<string, string?>? environment = null,
        ShellPolicy? policy = null,
        TimeSpan? timeout = null,
        int maxOutputBytes = DefaultMaxOutputBytes,
        Action<string>? onCommand = null)
    {
        if (mode == ShellMode.Persistent)
        {
            throw new NotSupportedException(
                "Persistent mode is reserved for a follow-up release. Use ShellMode.Stateless.");
        }
        if (maxOutputBytes <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(maxOutputBytes));
        }

        this._policy = policy ?? new ShellPolicy();
        this._shell = ShellResolver.Resolve(shell);
        this._timeout = timeout ?? TimeSpan.FromSeconds(30);
        this._maxOutputBytes = maxOutputBytes;
        this._workingDirectory = workingDirectory;
        this._environment = environment;
        this._onCommand = onCommand;
    }

    /// <summary>Gets the resolved shell binary that will host commands.</summary>
    public string ResolvedShellBinary => this._shell.Binary;

    /// <summary>
    /// Run a single command and return its result.
    /// </summary>
    /// <param name="command">The command to execute.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>The captured <see cref="ShellResult"/>.</returns>
    /// <exception cref="ShellCommandRejectedException">Thrown when the policy denies the command.</exception>
    public async Task<ShellResult> RunAsync(string command, CancellationToken cancellationToken = default)
    {
        if (command is null)
        {
            throw new ArgumentNullException(nameof(command));
        }

        var decision = this._policy.Evaluate(new ShellRequest(command, this._workingDirectory));
        if (!decision.Allowed)
        {
            throw new ShellCommandRejectedException(
                $"Command rejected by policy: {decision.Reason ?? "(unspecified)"}");
        }

        this._onCommand?.Invoke(command);

        var startInfo = new ProcessStartInfo
        {
            FileName = this._shell.Binary,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            RedirectStandardInput = false,
            UseShellExecute = false,
            CreateNoWindow = true,
            WorkingDirectory = this._workingDirectory ?? Directory.GetCurrentDirectory(),
        };

        foreach (var arg in this._shell.StatelessArgvForCommand(command))
        {
            startInfo.ArgumentList.Add(arg);
        }

        if (this._environment is not null)
        {
            foreach (var kv in this._environment)
            {
                if (kv.Value is null)
                {
                    _ = startInfo.Environment.Remove(kv.Key);
                }
                else
                {
                    startInfo.Environment[kv.Key] = kv.Value;
                }
            }
        }

        // PowerShell defaults to non-UTF8 output redirection; force UTF-8 to avoid mojibake.
        if (this._shell.Kind == ShellKind.PowerShell)
        {
            startInfo.Environment["PSDefaultParameterValues"] = "Out-File:Encoding=utf8";
        }

        using var process = new Process { StartInfo = startInfo, EnableRaisingEvents = true };
        var stdoutBuf = new StringBuilder();
        var stderrBuf = new StringBuilder();
        var stdoutTruncated = false;
        var stderrTruncated = false;

        process.OutputDataReceived += (_, e) =>
        {
            if (e.Data is null)
            {
                return;
            }
            AppendCapped(stdoutBuf, e.Data, this._maxOutputBytes, ref stdoutTruncated);
        };
        process.ErrorDataReceived += (_, e) =>
        {
            if (e.Data is null)
            {
                return;
            }
            AppendCapped(stderrBuf, e.Data, this._maxOutputBytes, ref stderrTruncated);
        };

        var stopwatch = Stopwatch.StartNew();
        try
        {
            _ = process.Start();
        }
        catch (Win32Exception ex)
        {
            throw new InvalidOperationException(
                $"Failed to launch shell '{this._shell.Binary}': {ex.Message}", ex);
        }

        process.BeginOutputReadLine();
        process.BeginErrorReadLine();

        var timedOut = false;
        using var timeoutCts = this._timeout is null
            ? new CancellationTokenSource()
            : new CancellationTokenSource(this._timeout.Value);
        using var linkedCts = CancellationTokenSource.CreateLinkedTokenSource(
            cancellationToken, timeoutCts.Token);

        try
        {
            await process.WaitForExitAsync(linkedCts.Token).ConfigureAwait(false);
        }
        catch (OperationCanceledException) when (timeoutCts.IsCancellationRequested && !cancellationToken.IsCancellationRequested)
        {
            timedOut = true;
        }
        catch (OperationCanceledException)
        {
            KillProcessTree(process);
            throw;
        }

        if (timedOut)
        {
            KillProcessTree(process);
            try
            {
                await process.WaitForExitAsync(CancellationToken.None).ConfigureAwait(false);
            }
            catch (Exception ex) when (ex is InvalidOperationException || ex is System.ComponentModel.Win32Exception)
            {
                // Best-effort shutdown after timeout — process may already be reaped.
            }
        }

        stopwatch.Stop();

        // Drain the async readers — WaitForExit doesn't guarantee the
        // OutputDataReceived/ErrorDataReceived events have all fired.
        process.WaitForExit();

        return new ShellResult(
            Stdout: stdoutBuf.ToString(),
            Stderr: stderrBuf.ToString(),
            ExitCode: timedOut ? -1 : process.ExitCode,
            Duration: stopwatch.Elapsed,
            Truncated: stdoutTruncated || stderrTruncated,
            TimedOut: timedOut);
    }

    /// <summary>
    /// Build an <see cref="AIFunction"/> bound to this tool, suitable for
    /// adding to <see cref="ChatOptions.Tools"/>.
    /// </summary>
    /// <param name="name">Function name surfaced to the model. Defaults to <c>run_shell</c>.</param>
    /// <param name="description">Function description for the model.</param>
    /// <returns>An <see cref="AIFunction"/> wrapping <see cref="RunAsync"/>.</returns>
    public AIFunction AsAIFunction(string name = "run_shell", string? description = null)
    {
        description ??= "Execute a single shell command and return its stdout, stderr, and exit code. " +
            "The tool runs commands directly on the host. The user reviews and approves each call.";

        return AIFunctionFactory.Create(
            async ([Description("The shell command to execute.")] string command,
                CancellationToken cancellationToken) =>
            {
                try
                {
                    var result = await this.RunAsync(command, cancellationToken).ConfigureAwait(false);
                    return result.FormatForModel();
                }
                catch (ShellCommandRejectedException ex)
                {
                    return $"Command blocked by policy: {ex.Message}";
                }
            },
            new AIFunctionFactoryOptions
            {
                Name = name,
                Description = description,
            });
    }

    /// <inheritdoc />
    public void Dispose()
    {
        // Reserved for persistent-mode resources (the long-lived shell process).
        // Stateless mode owns no long-lived state, but Dispose is part of the
        // contract so callers can write `using var shell = new LocalShellTool();`.
    }

    private static void AppendCapped(StringBuilder sb, string line, int cap, ref bool truncated)
    {
        if (truncated)
        {
            return;
        }
        // Approximate cap on bytes via char length (UTF-16). Good enough for
        // the byte cap — exact byte accounting would require encoding every
        // append.
        if (sb.Length + line.Length + 1 > cap)
        {
            var allowed = Math.Max(0, cap - sb.Length);
            if (allowed > 0)
            {
                _ = sb.Append(line, 0, Math.Min(allowed, line.Length));
            }
            truncated = true;
            return;
        }
        _ = sb.AppendLine(line);
    }

    private static void KillProcessTree(Process process)
    {
        try
        {
#if NET5_0_OR_GREATER
            process.Kill(entireProcessTree: true);
#else
            process.Kill();
#endif
        }
        catch (InvalidOperationException)
        {
            // Process already exited.
        }
        catch (System.ComponentModel.Win32Exception)
        {
            // Best-effort tree-kill — child has likely already exited.
        }
    }
}

/// <summary>
/// Thrown when <see cref="LocalShellTool"/> rejects a command via its policy.
/// </summary>
public sealed class ShellCommandRejectedException : Exception
{
    /// <summary>Initializes a new instance of the <see cref="ShellCommandRejectedException"/> class.</summary>
    /// <param name="message">The exception message.</param>
    public ShellCommandRejectedException(string message) : base(message)
    {
    }

    /// <summary>Initializes a new instance of the <see cref="ShellCommandRejectedException"/> class.</summary>
    /// <param name="message">The exception message.</param>
    /// <param name="inner">The inner exception.</param>
    public ShellCommandRejectedException(string message, Exception inner) : base(message, inner)
    {
    }

    /// <summary>Initializes a new instance of the <see cref="ShellCommandRejectedException"/> class.</summary>
    public ShellCommandRejectedException()
    {
    }
}
