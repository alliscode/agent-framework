// Copyright (c) Microsoft. All rights reserved.

using Microsoft.Bot.ObjectModel;

namespace Microsoft.Agents.AI.Workflows.Declarative.CodeGen.Python;

internal partial class PythonCopyConversationMessagesTemplate : PythonActionTemplate
{
    public PythonCopyConversationMessagesTemplate(CopyConversationMessages model)
    {
        this.Model = this.Initialize(model);
    }

    public CopyConversationMessages Model { get; }
}
