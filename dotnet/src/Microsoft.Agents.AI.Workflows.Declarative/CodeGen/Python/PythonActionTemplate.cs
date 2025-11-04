// Copyright (c) Microsoft. All rights reserved.

using Microsoft.Agents.AI.Workflows.Declarative.Extensions;
using Microsoft.Agents.AI.Workflows.Declarative.Interpreter;
using Microsoft.Bot.ObjectModel;

namespace Microsoft.Agents.AI.Workflows.Declarative.CodeGen.Python;

internal abstract class PythonActionTemplate : PythonCodeTemplate, IModeledAction
{
    public string Id { get; private set; } = string.Empty;

    public string Name { get; private set; } = string.Empty;

    public string ParentId { get; private set; } = string.Empty;

    public bool UseAgentProvider { get; init; }

    protected TAction Initialize<TAction>(TAction model) where TAction : DialogAction
    {
        this.Id = model.GetId();
        this.ParentId = model.GetParentId() ?? WorkflowActionVisitor.Steps.Root();
        this.Name = ToPythonClassName(this.Id);

        return model;
    }

    /// <summary>
    /// Convert an action ID to a Python class name (PascalCase)
    /// </summary>
    private static string ToPythonClassName(string actionId)
    {
        // Convert action_id to ActionId for class names
        var parts = actionId.Split('_');
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
