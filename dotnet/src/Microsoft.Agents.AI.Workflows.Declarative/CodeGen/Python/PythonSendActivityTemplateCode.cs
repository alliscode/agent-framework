// Copyright (c) Microsoft. All rights reserved.

using Microsoft.Bot.ObjectModel;

namespace Microsoft.Agents.AI.Workflows.Declarative.CodeGen.Python;

internal partial class PythonSendActivityTemplate : PythonActionTemplate
{
    public PythonSendActivityTemplate(SendActivity model)
    {
        this.Model = this.Initialize(model);
    }

    public SendActivity Model { get; }
}
