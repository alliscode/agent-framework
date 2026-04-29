// Copyright (c) Microsoft. All rights reserved.

using System;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using Microsoft.Agents.AI.Tools.Shell;
using Microsoft.Extensions.AI;

namespace Microsoft.Agents.AI.Tools.Shell.UnitTests;

/// <summary>
/// Smoke + behavior tests for <see cref="LocalShellTool"/> and <see cref="ShellPolicy"/>.
/// </summary>
public sealed class LocalShellToolTests
{
    [Fact]
    public void Policy_DenyList_BlocksDestructiveRm()
    {
        var policy = new ShellPolicy();
        var decision = policy.Evaluate(new ShellRequest("rm -rf /"));
        Assert.False(decision.Allowed);
        Assert.Contains("deny pattern", decision.Reason ?? string.Empty, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void Policy_AllowList_OverridesDeny()
    {
        var policy = new ShellPolicy(
            allowList: ["^echo "],
            denyList: ["echo"]);
        var decision = policy.Evaluate(new ShellRequest("echo hello"));
        Assert.True(decision.Allowed);
    }

    [Fact]
    public void Policy_EmptyCommand_Denied()
    {
        var decision = new ShellPolicy().Evaluate(new ShellRequest("   "));
        Assert.False(decision.Allowed);
    }

    [Fact]
    public void Policy_DenyList_IsGuardrailNotBoundary_KnownBypass()
    {
        // This test codifies that the policy is a guardrail — a small change
        // to the command (variable indirection) bypasses the literal `rm -rf /`
        // pattern. Documented as expected behavior; the real boundary is
        // approval-in-the-loop.
        var policy = new ShellPolicy();
        var decision = policy.Evaluate(new ShellRequest("${RM:=rm} -rf /"));
        Assert.True(decision.Allowed, "Policy is intentionally a guardrail; this bypass is documented in ADR 0026.");
    }

    [Fact]
    public async Task RunAsync_EchoCommand_RoundtripsStdoutAndExitCode()
    {
        using var shell = new LocalShellTool();
        // Use an OS-appropriate echo. On Windows the resolved shell is PowerShell.
        var result = await shell.RunAsync("echo hello-from-shell");
        Assert.Equal(0, result.ExitCode);
        Assert.Contains("hello-from-shell", result.Stdout, StringComparison.Ordinal);
        Assert.False(result.TimedOut);
    }

    [Fact]
    public async Task RunAsync_RejectedCommand_ThrowsShellCommandRejected()
    {
        using var shell = new LocalShellTool();
        await Assert.ThrowsAsync<ShellCommandRejectedException>(
            () => shell.RunAsync("rm -rf /"));
    }

    [Fact]
    public async Task RunAsync_NonZeroExit_PropagatesExitCode()
    {
        using var shell = new LocalShellTool();
        // Exit-1 phrasing portable across bash and PowerShell.
        var script = RuntimeInformation.IsOSPlatform(OSPlatform.Windows)
            ? "exit 7"
            : "exit 7";
        var result = await shell.RunAsync(script);
        Assert.Equal(7, result.ExitCode);
    }

    [Fact]
    public async Task RunAsync_Timeout_FlagsTimedOutAndKillsProcess()
    {
        using var shell = new LocalShellTool(timeout: TimeSpan.FromMilliseconds(250));
        var sleepCmd = RuntimeInformation.IsOSPlatform(OSPlatform.Windows)
            ? "Start-Sleep -Seconds 30"
            : "sleep 30";
        var result = await shell.RunAsync(sleepCmd);
        Assert.True(result.TimedOut);
        Assert.Equal(-1, result.ExitCode);
        Assert.True(result.Duration < TimeSpan.FromSeconds(10));
    }

    [Fact]
    public void AsAIFunction_DefaultsToApprovalRequired()
    {
        using var shell = new LocalShellTool();
        var fn = shell.AsAIFunction();
        Assert.IsType<ApprovalRequiredAIFunction>(fn);
        Assert.Equal("run_shell", fn.Name);
        Assert.False(string.IsNullOrWhiteSpace(fn.Description));
    }

    [Fact]
    public void AsAIFunction_OptOut_ReturnsPlainFunction()
    {
        using var shell = new LocalShellTool();
        var fn = shell.AsAIFunction(requireApproval: false);
        Assert.IsNotType<ApprovalRequiredAIFunction>(fn);
        Assert.Equal("run_shell", fn.Name);
    }

    [Fact]
    public void Persistent_Mode_NotImplemented_Throws()
    {
        Assert.Throws<NotSupportedException>(() => new LocalShellTool(mode: ShellMode.Persistent));
    }

    [Fact]
    public async Task OnCommand_HookFiredForAllowedCommandsOnly()
    {
        var calls = new System.Collections.Generic.List<string>();
        using var shell = new LocalShellTool(onCommand: cmd => calls.Add(cmd));
        await Assert.ThrowsAsync<ShellCommandRejectedException>(() => shell.RunAsync("rm -rf /"));
        Assert.Empty(calls);
    }
}
