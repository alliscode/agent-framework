// Copyright (c) Microsoft. All rights reserved.

using Microsoft.Bot.ObjectModel;

namespace Microsoft.Agents.AI.Workflows.Declarative.CodeGen.Python;

internal partial class PythonRetrieveConversationMessagesTemplate : PythonActionTemplate
{
    public PythonRetrieveConversationMessagesTemplate(RetrieveConversationMessages model)
    {
        this.Model = this.Initialize(model);
    }

    public RetrieveConversationMessages Model { get; }
}
