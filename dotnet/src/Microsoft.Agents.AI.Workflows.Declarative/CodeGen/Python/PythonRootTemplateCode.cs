// Copyright (c) Microsoft. All rights reserved.

using Microsoft.Agents.AI.Workflows.Declarative.Extensions;
using Microsoft.Agents.AI.Workflows.Declarative.PowerFx;

namespace Microsoft.Agents.AI.Workflows.Declarative.CodeGen.Python;

internal partial class PythonRootTemplate
{
    internal PythonRootTemplate(
        string workflowId,
        WorkflowTypeInfo typeInfo)
    {
        this.Id = workflowId;
        this.TypeInfo = typeInfo;
        this.TypeName = ToPythonClassName(workflowId);
    }

    public string Id { get; }
    public WorkflowTypeInfo TypeInfo { get; }
    public string TypeName { get; }

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
