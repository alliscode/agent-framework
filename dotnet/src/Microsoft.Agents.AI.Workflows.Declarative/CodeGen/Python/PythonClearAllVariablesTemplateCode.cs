// Copyright (c) Microsoft. All rights reserved.

using Microsoft.Bot.ObjectModel;

namespace Microsoft.Agents.AI.Workflows.Declarative.CodeGen.Python;

internal partial class PythonClearAllVariablesTemplate : PythonActionTemplate
{
    public PythonClearAllVariablesTemplate(ClearAllVariables model)
    {
        this.Model = this.Initialize(model);
    }

    public ClearAllVariables Model { get; }
}
