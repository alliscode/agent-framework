// Copyright (c) Microsoft. All rights reserved.

using Microsoft.Agents.AI.Workflows.Declarative.Extensions;
using Microsoft.Bot.ObjectModel;

namespace Microsoft.Agents.AI.Workflows.Declarative.CodeGen.Python;

internal partial class PythonDefaultTemplate : PythonActionTemplate
{
    public PythonDefaultTemplate(DialogAction model, string rootId)
    {
        this.Initialize(model);
        this.InstanceVariable = PythonCodeTemplate.ToPythonName(this.Id);
        this.RootVariable = PythonCodeTemplate.ToPythonName(rootId);
        this.Action = null;
    }

    public string InstanceVariable { get; }
    public string RootVariable { get; }
    public string? Action { get; }
}
