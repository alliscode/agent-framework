// Copyright (c) Microsoft. All rights reserved.

using Microsoft.Bot.ObjectModel;

namespace Microsoft.Agents.AI.Workflows.Declarative.CodeGen.Python;

internal partial class PythonAddConversationMessageTemplate : PythonActionTemplate
{
    public PythonAddConversationMessageTemplate(AddConversationMessage model)
    {
        this.Model = this.Initialize(model);
    }

    public AddConversationMessage Model { get; }
}
