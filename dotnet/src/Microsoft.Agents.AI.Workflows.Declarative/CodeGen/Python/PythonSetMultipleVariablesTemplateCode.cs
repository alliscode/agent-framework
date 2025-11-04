// Copyright (c) Microsoft. All rights reserved.

using Microsoft.Bot.ObjectModel;

namespace Microsoft.Agents.AI.Workflows.Declarative.CodeGen.Python;

internal partial class PythonSetMultipleVariablesTemplate : PythonActionTemplate
{
    public PythonSetMultipleVariablesTemplate(SetMultipleVariables model)
    {
        this.Model = this.Initialize(model);
    }

    public SetMultipleVariables Model { get; }
}
