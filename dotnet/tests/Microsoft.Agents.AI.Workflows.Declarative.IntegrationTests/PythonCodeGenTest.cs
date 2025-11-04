// Copyright (c) Microsoft. All rights reserved.

using System;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using Xunit.Abstractions;

namespace Microsoft.Agents.AI.Workflows.Declarative.IntegrationTests;

/// <summary>
/// Tests Python code generation by <see cref="DeclarativeWorkflowBuilder"/>.
/// </summary>
/// <remarks>
/// These tests verify that Python code can be generated from YAML workflows.
/// They validate syntax correctness but do not execute the Python code.
///
/// IMPORTANT: Many tests are currently marked as Skip because the stub action templates
/// need to be fully implemented. Once the templates are complete (following the pattern
/// in PythonSendActivityTemplate.cs), these tests should be enabled.
/// </remarks>
public sealed class PythonCodeGenTest(ITestOutputHelper output)
{
    private readonly ITestOutputHelper _output = output;

    [Theory]
    [InlineData("SendActivity.yaml", "Test.WorkflowProviders", "Test")]
    public void GeneratePythonCode_SendActivity_Success(string workflowFileName, string workflowNamespace, string workflowPrefix)
    {
        // Arrange
        string workflowPath = Path.Combine(Environment.CurrentDirectory, "Workflows", workflowFileName);

        // Act
        string pythonCode = DeclarativeWorkflowBuilder.Eject(
            workflowPath,
            DeclarativeWorkflowLanguage.Python,
            workflowNamespace,
            workflowPrefix);

        // Assert
        Assert.NotNull(pythonCode);
        Assert.NotEmpty(pythonCode);

        // Verify basic Python structure
        Assert.Contains("class", pythonCode);
        Assert.Contains("def create_workflow", pythonCode);
        Assert.Contains("WorkflowProvider", pythonCode);

        // Output for manual inspection
        this._output.WriteLine("=== Generated Python Code ===");
        this._output.WriteLine(pythonCode);

        // Validate Python syntax (basic checks)
        ValidatePythonSyntax(pythonCode);
    }

    [Theory(Skip = "Requires InvokeAzureAgent template implementation")]
    [InlineData("InvokeAgent.yaml", "Test.WorkflowProviders", "Test")]
    public void GeneratePythonCode_InvokeAgent_Success(string workflowFileName, string workflowNamespace, string workflowPrefix)
    {
        // Arrange
        string workflowPath = Path.Combine(Environment.CurrentDirectory, "Workflows", workflowFileName);

        // Act
        string pythonCode = DeclarativeWorkflowBuilder.Eject(
            workflowPath,
            DeclarativeWorkflowLanguage.Python,
            workflowNamespace,
            workflowPrefix);

        // Assert
        Assert.NotNull(pythonCode);
        Assert.NotEmpty(pythonCode);

        this._output.WriteLine("=== Generated Python Code ===");
        this._output.WriteLine(pythonCode);

        ValidatePythonSyntax(pythonCode);
        Assert.Contains("InvokeAzureAgentExecutor", pythonCode);
    }

    [Theory(Skip = "Requires SetVariable template implementation")]
    [InlineData("CheckSystem.yaml", "Test.WorkflowProviders", "Test")]
    public void GeneratePythonCode_CheckSystem_Success(string workflowFileName, string workflowNamespace, string workflowPrefix)
    {
        // Arrange
        string workflowPath = Path.Combine(Environment.CurrentDirectory, "Workflows", workflowFileName);

        // Act
        string pythonCode = DeclarativeWorkflowBuilder.Eject(
            workflowPath,
            DeclarativeWorkflowLanguage.Python,
            workflowNamespace,
            workflowPrefix);

        // Assert
        Assert.NotNull(pythonCode);
        Assert.NotEmpty(pythonCode);

        this._output.WriteLine("=== Generated Python Code ===");
        this._output.WriteLine(pythonCode);

        ValidatePythonSyntax(pythonCode);
    }

    [Theory(Skip = "Requires Conversation template implementations")]
    [InlineData("ConversationMessages.yaml", "Test.WorkflowProviders", "Test")]
    public void GeneratePythonCode_ConversationMessages_Success(string workflowFileName, string workflowNamespace, string workflowPrefix)
    {
        // Arrange
        string workflowPath = Path.Combine(Environment.CurrentDirectory, "Workflows", workflowFileName);

        // Act
        string pythonCode = DeclarativeWorkflowBuilder.Eject(
            workflowPath,
            DeclarativeWorkflowLanguage.Python,
            workflowNamespace,
            workflowPrefix);

        // Assert
        Assert.NotNull(pythonCode);
        Assert.NotEmpty(pythonCode);

        this._output.WriteLine("=== Generated Python Code ===");
        this._output.WriteLine(pythonCode);

        ValidatePythonSyntax(pythonCode);
        Assert.Contains("CreateConversationExecutor", pythonCode);
    }

    [Theory(Skip = "Requires template implementations for complex scenarios")]
    [InlineData("Marketing.yaml", null, "Marketing")]
    public void GeneratePythonCode_MarketingScenario_Success(string workflowFileName, string? workflowNamespace, string workflowPrefix)
    {
        // Arrange
        string workflowPath = Path.Combine(GetRepoFolder(), "workflow-samples", workflowFileName);

        // Act
        string pythonCode = DeclarativeWorkflowBuilder.Eject(
            workflowPath,
            DeclarativeWorkflowLanguage.Python,
            workflowNamespace,
            workflowPrefix);

        // Assert
        Assert.NotNull(pythonCode);
        Assert.NotEmpty(pythonCode);

        this._output.WriteLine("=== Generated Python Code ===");
        this._output.WriteLine(pythonCode);

        ValidatePythonSyntax(pythonCode);
    }

    [Theory(Skip = "Requires template implementations for complex scenarios")]
    [InlineData("MathChat.yaml", null, "MathChat")]
    public void GeneratePythonCode_MathChatScenario_Success(string workflowFileName, string? workflowNamespace, string workflowPrefix)
    {
        // Arrange
        string workflowPath = Path.Combine(GetRepoFolder(), "workflow-samples", workflowFileName);

        // Act
        string pythonCode = DeclarativeWorkflowBuilder.Eject(
            workflowPath,
            DeclarativeWorkflowLanguage.Python,
            workflowNamespace,
            workflowPrefix);

        // Assert
        Assert.NotNull(pythonCode);
        Assert.NotEmpty(pythonCode);

        this._output.WriteLine("=== Generated Python Code ===");
        this._output.WriteLine(pythonCode);

        ValidatePythonSyntax(pythonCode);
    }

    [Fact]
    public void GeneratePythonCode_WithNamespace_IncludesModuleComment()
    {
        // Arrange
        string workflowPath = Path.Combine(Environment.CurrentDirectory, "Workflows", "SendActivity.yaml");
        const string expectedNamespace = "my.test.workflows";

        // Act
        string pythonCode = DeclarativeWorkflowBuilder.Eject(
            workflowPath,
            DeclarativeWorkflowLanguage.Python,
            expectedNamespace,
            "Sample");

        // Assert
        Assert.Contains($"# Module: {expectedNamespace}", pythonCode);

        this._output.WriteLine("=== Generated Python Code ===");
        this._output.WriteLine(pythonCode);
    }

    [Fact]
    public void GeneratePythonCode_WithPrefix_IncludesInClassName()
    {
        // Arrange
        string workflowPath = Path.Combine(Environment.CurrentDirectory, "Workflows", "SendActivity.yaml");
        const string expectedPrefix = "CustomPrefix";

        // Act
        string pythonCode = DeclarativeWorkflowBuilder.Eject(
            workflowPath,
            DeclarativeWorkflowLanguage.Python,
            null,
            expectedPrefix);

        // Assert
        Assert.Contains($"class {expectedPrefix}WorkflowProvider:", pythonCode);

        this._output.WriteLine("=== Generated Python Code ===");
        this._output.WriteLine(pythonCode);
    }

    [Fact]
    public void GeneratePythonCode_VerifyPythonNamingConventions()
    {
        // Arrange
        string workflowPath = Path.Combine(Environment.CurrentDirectory, "Workflows", "SendActivity.yaml");

        // Act
        string pythonCode = DeclarativeWorkflowBuilder.Eject(
            workflowPath,
            DeclarativeWorkflowLanguage.Python,
            null,
            "Test");

        // Assert - Verify Python naming conventions
        Assert.Contains("def create_workflow(", pythonCode); // snake_case for functions
        Assert.Contains("def __init__(", pythonCode); // Python constructor
        Assert.Contains("async def execute_async(", pythonCode); // async methods
        Assert.Matches(@"class \w+Executor\(", pythonCode); // PascalCase for classes

        // Verify Python keywords are NOT C# keywords (at least one check)
        Assert.DoesNotContain("true", pythonCode); // Python uses True, not true
        Assert.DoesNotContain("false", pythonCode); // Python uses False, not false
        Assert.DoesNotContain("null", pythonCode); // Python uses None, not null

        this._output.WriteLine("=== Generated Python Code ===");
        this._output.WriteLine(pythonCode);
    }

    [Fact]
    public void GeneratePythonCode_VerifyImports()
    {
        // Arrange
        string workflowPath = Path.Combine(Environment.CurrentDirectory, "Workflows", "SendActivity.yaml");

        // Act
        string pythonCode = DeclarativeWorkflowBuilder.Eject(
            workflowPath,
            DeclarativeWorkflowLanguage.Python,
            null,
            "Test");

        // Assert - Verify required imports
        Assert.Contains("from typing import", pythonCode);
        Assert.Contains("from datetime import", pythonCode);
        Assert.Contains("from agents.workflows import", pythonCode);
        Assert.Contains("from agents.workflows.declarative import DeclarativeWorkflowOptions", pythonCode);
        Assert.Contains("import asyncio", pythonCode);

        this._output.WriteLine("=== Generated Python Code ===");
        this._output.WriteLine(pythonCode);
    }

    [Fact]
    public void GeneratePythonCode_VerifyAutoGeneratedComment()
    {
        // Arrange
        string workflowPath = Path.Combine(Environment.CurrentDirectory, "Workflows", "SendActivity.yaml");

        // Act
        string pythonCode = DeclarativeWorkflowBuilder.Eject(
            workflowPath,
            DeclarativeWorkflowLanguage.Python,
            null,
            "Test");

        // Assert
        Assert.Contains("# <auto-generated>", pythonCode);
        Assert.Contains("#     This code was generated by a tool.", pythonCode);
        Assert.Contains("# </auto-generated>", pythonCode);

        this._output.WriteLine("=== Generated Python Code ===");
        this._output.WriteLine(pythonCode);
    }

    /// <summary>
    /// Performs basic Python syntax validation checks.
    /// </summary>
    private static void ValidatePythonSyntax(string pythonCode)
    {
        // Check for balanced parentheses
        int openParens = pythonCode.Count(c => c == '(');
        int closeParens = pythonCode.Count(c => c == ')');
        Assert.Equal(openParens, closeParens);

        // Check for balanced brackets
        int openBrackets = pythonCode.Count(c => c == '[');
        int closeBrackets = pythonCode.Count(c => c == ']');
        Assert.Equal(openBrackets, closeBrackets);

        // Check for balanced braces
        int openBraces = pythonCode.Count(c => c == '{');
        int closeBraces = pythonCode.Count(c => c == '}');
        Assert.Equal(openBraces, closeBraces);

        // Verify no C# specific syntax leaked through
        Assert.DoesNotContain("public ", pythonCode);
        Assert.DoesNotContain("private ", pythonCode);
        Assert.DoesNotContain("protected ", pythonCode);
        Assert.DoesNotContain(" void ", pythonCode);
        Assert.DoesNotContain("namespace ", pythonCode);
        Assert.DoesNotContain("using ", pythonCode);

        // Verify Python-specific syntax is present
        Assert.Contains("def ", pythonCode);
        Assert.Contains("class ", pythonCode);

        // Check indentation is spaces (Python convention, usually 4 spaces)
        foreach (var line in pythonCode.Split('\n'))
        {
            if (line.Length > 0 && line[0] == ' ')
            {
                // If indented, should not contain tabs
                Assert.DoesNotContain("\t", line);
            }
        }
    }

    private static string GetRepoFolder()
    {
        DirectoryInfo? current = new(Directory.GetCurrentDirectory());

        while (current is not null)
        {
            if (Directory.Exists(Path.Combine(current.FullName, ".git")))
            {
                return current.FullName;
            }

            current = current.Parent;
        }

        throw new InvalidOperationException("Unable to locate repository root folder.");
    }
}
