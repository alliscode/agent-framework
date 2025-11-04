// Copyright (c) Microsoft. All rights reserved.

using Microsoft.Bot.ObjectModel;

namespace Microsoft.Agents.AI.Workflows.Declarative.CodeGen.Python;

internal partial class PythonRetrieveConversationMessageTemplate : PythonActionTemplate
{
    public PythonRetrieveConversationMessageTemplate(RetrieveConversationMessage model)
    {
        this.Model = this.Initialize(model);
    }

    public RetrieveConversationMessage Model { get; }
}
