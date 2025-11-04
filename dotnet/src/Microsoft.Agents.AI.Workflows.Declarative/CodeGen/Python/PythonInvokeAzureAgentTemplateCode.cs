// Copyright (c) Microsoft. All rights reserved.

using Microsoft.Bot.ObjectModel;

namespace Microsoft.Agents.AI.Workflows.Declarative.CodeGen.Python;

internal partial class PythonInvokeAzureAgentTemplate : PythonActionTemplate
{
    public PythonInvokeAzureAgentTemplate(InvokeAzureAgent model)
    {
        this.Model = this.Initialize(model);
        this.UseAgentProvider = true;
    }

    public InvokeAzureAgent Model { get; }
}
