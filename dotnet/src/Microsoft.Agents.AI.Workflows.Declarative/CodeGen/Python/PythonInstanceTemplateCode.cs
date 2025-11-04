// Copyright (c) Microsoft. All rights reserved.

using Microsoft.Agents.AI.Workflows.Declarative.Extensions;

namespace Microsoft.Agents.AI.Workflows.Declarative.CodeGen.Python;

internal partial class PythonInstanceTemplate
{
    public PythonInstanceTemplate(string executorId, string rootId, bool hasProvider = false)
    {
        this.InstanceVariable = PythonCodeTemplate.ToPythonName(executorId);
        this.ExecutorType = ToPythonClassName(executorId);
        this.RootVariable = PythonCodeTemplate.ToPythonName(rootId);
        this.HasProvider = hasProvider;
    }

    public string InstanceVariable { get; }
    public string ExecutorType { get; }
    public string RootVariable { get; }
    public bool HasProvider { get; }

    /// <summary>
    /// Convert an executor ID to a Python class name (PascalCase)
    /// </summary>
    private static string ToPythonClassName(string executorId)
    {
        var parts = executorId.Split('_');
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
