// Copyright (c) Microsoft. All rights reserved.

using Microsoft.Agents.AI.Workflows.Declarative.Extensions;

namespace Microsoft.Agents.AI.Workflows.Declarative.CodeGen.Python;

internal partial class PythonEdgeTemplate
{
    public PythonEdgeTemplate(string sourceId, string targetId, string? condition)
    {
        this.SourceId = PythonCodeTemplate.ToPythonName(sourceId);
        this.TargetId = PythonCodeTemplate.ToPythonName(targetId);
        this.Condition = condition;
    }

    public string SourceId { get; }
    public string TargetId { get; }
    public string? Condition { get; }
}
