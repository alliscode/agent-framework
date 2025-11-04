// Copyright (c) Microsoft. All rights reserved.

using Microsoft.Bot.ObjectModel;

namespace Microsoft.Agents.AI.Workflows.Declarative.CodeGen.Python;

internal partial class PythonSetTextVariableTemplate : PythonActionTemplate
{
    public PythonSetTextVariableTemplate(SetTextVariable model)
    {
        this.Model = this.Initialize(model);
    }

    public SetTextVariable Model { get; }
}
