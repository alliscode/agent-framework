// Copyright (c) Microsoft. All rights reserved.

using Microsoft.Bot.ObjectModel;

namespace Microsoft.Agents.AI.Workflows.Declarative.CodeGen.Python;

internal partial class PythonSetVariableTemplate : PythonActionTemplate
{
    public PythonSetVariableTemplate(SetVariable model)
    {
        this.Model = this.Initialize(model);
    }

    public SetVariable Model { get; }
}
