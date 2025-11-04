// Copyright (c) Microsoft. All rights reserved.

using Microsoft.Bot.ObjectModel;

namespace Microsoft.Agents.AI.Workflows.Declarative.CodeGen.Python;

internal partial class PythonForeachTemplate : PythonActionTemplate
{
    public PythonForeachTemplate(Foreach model)
    {
        this.Model = this.Initialize(model);
    }

    public Foreach Model { get; }
}
