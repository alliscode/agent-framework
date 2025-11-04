// Copyright (c) Microsoft. All rights reserved.

using Microsoft.Agents.AI.Workflows.Declarative.Extensions;

namespace Microsoft.Agents.AI.Workflows.Declarative.CodeGen.Python;

internal partial class PythonEmptyTemplate
{
    public PythonEmptyTemplate(string actionId, string rootId, string? action = null)
    {
        this.Id = actionId;
        this.Name = PythonCodeTemplate.ToPythonName(this.Id);
        this.InstanceVariable = PythonCodeTemplate.ToPythonName(this.Id);
        this.RootVariable = PythonCodeTemplate.ToPythonName(rootId);
        this.Action = action;
    }

    public string Id { get; }
    public string Name { get; }
    public string InstanceVariable { get; }
    public string RootVariable { get; }
    public string? Action { get; }
}
