// Copyright (c) Microsoft. All rights reserved.

using Microsoft.Bot.ObjectModel;

namespace Microsoft.Agents.AI.Workflows.Declarative.CodeGen.Python;

internal partial class PythonConditionGroupTemplate : PythonActionTemplate
{
    public PythonConditionGroupTemplate(ConditionGroup model)
    {
        this.Model = this.Initialize(model);
    }

    public ConditionGroup Model { get; }
}
