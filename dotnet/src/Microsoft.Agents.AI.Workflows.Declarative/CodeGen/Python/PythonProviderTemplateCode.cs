// Copyright (c) Microsoft. All rights reserved.

using System.Collections.Generic;
using Microsoft.Agents.AI.Workflows.Declarative.Extensions;

namespace Microsoft.Agents.AI.Workflows.Declarative.CodeGen.Python;

internal partial class PythonProviderTemplate
{
    public PythonProviderTemplate(
        string workflowId,
        IEnumerable<string> executors,
        IEnumerable<string> instances,
        IEnumerable<string> edges)
    {
        this.Executors = executors;
        this.Instances = instances;
        this.Edges = edges;
        this.RootInstance = PythonCodeTemplate.ToPythonName(workflowId);
        this.RootExecutorType = ToPythonClassName(workflowId);
    }

    public string? Namespace { get; init; }
    public string? Prefix { get; init; }

    public string RootInstance { get; }
    public string RootExecutorType { get; }

    public IEnumerable<string> Executors { get; }
    public IEnumerable<string> Instances { get; }
    public IEnumerable<string> Edges { get; }

    public static IEnumerable<string> ByLine(IEnumerable<string> templates, bool formatGroup = false)
    {
        foreach (string template in templates)
        {
            foreach (string line in template.ByLine())
            {
                yield return line;
            }

            if (formatGroup)
            {
                yield return string.Empty;
            }
        }
    }

    /// <summary>
    /// Convert a workflow ID to a Python class name (PascalCase)
    /// </summary>
    private static string ToPythonClassName(string workflowId)
    {
        var parts = workflowId.Split('_');
        var result = new System.Text.StringBuilder();
        foreach (var part in parts)
        {
            if (part.Length > 0)
            {
                result.Append(char.ToUpper(part[0], System.Globalization.CultureInfo.InvariantCulture));
                if (part.Length > 1)
                {
                    result.Append(part.Substring(1));
                }
            }
        }
        return result.ToString();
    }
}
